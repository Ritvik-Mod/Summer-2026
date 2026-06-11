"""
Batched self-play: runs many games concurrently and evaluates all their MCTS
leaf nodes in single batched network forward passes, instead of one position
at a time. This is what actually feeds the GPU.

Corrections over the first draft of this file:
  1. DIRICHLET NOISE — the draft never injected root noise, so self-play had
     no exploration pressure beyond temperature sampling. Noise (alpha, eps)
     is now applied to every game's root at the START OF EVERY MOVE, exactly
     like the sequential pipeline (mcts.run_mcts with add_Noise=True).
  2. PRIOR MASKING — priors are now computed by masking ILLEGAL LOGITS to
     -inf BEFORE softmax (mathematically equivalent to softmax-then-renorm,
     but avoids the sum==0 edge case entirely and matches mcts.expand()).
  3. TREE REUSE + NOISE — when the root is re-rooted onto the chosen child,
     the child's priors are clean (noise is only ever applied at the root),
     and fresh noise is applied to it on the next move. Visit statistics of
     the kept subtree are reused, which is standard.
  4. FORCED-PASS POSITIONS are stepped through but NOT recorded as training
     examples (pi would be a degenerate one-hot on PASS), matching the
     sequential self_play.py behaviour.
  5. select_action uses sqrt(total_N + 1) so that priors (incl. noise)
     break the tie on the very first simulation instead of dict order.
  6. `sims` now means: simulations run AFTER the root is expanded and noise
     is applied — every one of them is a real search.
"""

import math
import torch
import numpy as np
import othello_env


class Node:
    __slots__ = ("env", "parent", "action_taken", "is_expanded",
                 "children", "N", "W", "Q", "P")

    def __init__(self, env, parent=None, action_taken=None):
        self.env = env
        self.parent = parent
        self.action_taken = action_taken
        self.is_expanded = False
        self.children = {}   # action -> Node
        self.N = {}          # visit count per action
        self.W = {}          # total value per action
        self.Q = {}          # mean value per action
        self.P = {}          # prior per action


def select_action(node, c_puct=1.0):
    total_N = sum(node.N.values())
    sqrt_total = math.sqrt(total_N + 1)   # +1: priors matter on first selection
    best_score = -math.inf
    best_action = None
    for a, P in node.P.items():
        Q = node.Q[a]
        N = node.N[a]
        score = Q + c_puct * P * sqrt_total / (1 + N)
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def traverse_to_leaf(node, c_puct=1.0):
    """Descend with PUCT until an unexpanded node or a terminal state."""
    while node.is_expanded and not node.env.done:
        a = select_action(node, c_puct)
        if a not in node.children:
            child_env = node.env.clone()
            child_env.step(a)
            node.children[a] = Node(child_env, parent=node, action_taken=a)
        node = node.children[a]
    return node


def backpropagate(node, value):
    """Walk up the tree updating edge stats. `value` must be from the
    perspective of the PARENT of `node`; it is negated at each level up."""
    while node.parent is not None:
        a = node.action_taken
        node = node.parent
        node.N[a] += 1
        node.W[a] += value
        node.Q[a] = node.W[a] / node.N[a]
        value = -value


def batched_expand(leaves, net, device):
    """Expand a list of distinct unexpanded leaves with ONE forward pass.
    Returns the network values, one per leaf, from each leaf's own
    player-to-move perspective."""
    states = np.array([lf.env.state() for lf in leaves])          # (B,2,n,n)
    masks  = np.array([lf.env.legal_mask() for lf in leaves])     # (B,A) bool

    states_t = torch.from_numpy(states).to(device)
    masks_t  = torch.from_numpy(masks).to(device)

    with torch.no_grad():
        logits, values = net(states_t)
        logits = logits.masked_fill(~masks_t, -math.inf)   # mask BEFORE softmax
        priors = torch.softmax(logits, dim=1).cpu().numpy()
    values = values.cpu().numpy()[:, 0]

    for i, lf in enumerate(leaves):
        for a in np.flatnonzero(masks[i]):
            a = int(a)
            lf.P[a] = float(priors[i, a])
            lf.N[a] = 0
            lf.W[a] = 0.0
            lf.Q[a] = 0.0
        lf.is_expanded = True
    return values


def _add_root_noise(root, alpha, eps):
    actions = list(root.P.keys())
    if len(actions) <= 1:
        return
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, nz in zip(actions, noise):
        root.P[a] = (1 - eps) * root.P[a] + eps * nz


def generate_batched_selfplay_data(net, num_games, n=8, sims=200,
                                   temp_moves=15, c_puct=1.0,
                                   alpha=0.3, eps=0.25):
    """Play `num_games` self-play games concurrently. Returns a list of
    (state, pi, z) training triples, identical format to self_play.py."""
    device = next(net.parameters()).device
    net.eval()

    envs         = [othello_env.OthelloEnv(n) for _ in range(num_games)]
    roots        = [Node(e.clone()) for e in envs]
    trajectories = [[] for _ in range(num_games)]
    moves_played = [0] * num_games
    active       = list(range(num_games))

    finished_examples = []

    while active:
        # ---- Phase 0: expand any unexpanded roots (batched), inject noise ----
        unexpanded = [roots[i] for i in active if not roots[i].is_expanded]
        if unexpanded:
            batched_expand(unexpanded, net, device)   # root values unused
        for i in active:
            _add_root_noise(roots[i], alpha, eps)

        # ---- Phase 1: simulations (one leaf per active game per sim) ----
        for _ in range(sims):
            eval_leaves = []
            for i in active:
                leaf = traverse_to_leaf(roots[i], c_puct)
                if leaf.env.done:
                    # true outcome, converted to the PARENT's perspective
                    backpropagate(leaf, -leaf.env.outcome_for_current_player())
                else:
                    eval_leaves.append(leaf)
            if eval_leaves:
                values = batched_expand(eval_leaves, net, device)
                for lf, v in zip(eval_leaves, values):
                    # network value is from the leaf's perspective;
                    # negate for the parent before backing up
                    backpropagate(lf, -float(v))

        # ---- Phase 2: pick a move in every active game ----
        for i in reversed(active):          # reversed: safe removal
            root, env = roots[i], envs[i]

            forced_pass = (len(root.P) == 1 and env.PASS in root.P)

            counts = root.N
            total = sum(counts.values())
            pi = np.zeros(env.action_size)
            for a, cnt in counts.items():
                pi[a] = cnt / total

            # record BEFORE stepping; skip degenerate forced-pass positions
            if not forced_pass:
                trajectories[i].append((env.state(), pi, env.to_move))

            if forced_pass:
                action = env.PASS
            elif moves_played[i] < temp_moves:
                pi_s = pi / pi.sum()
                action = int(np.random.choice(len(pi_s), p=pi_s))
            else:
                action = int(np.argmax(pi))

            env.step(action)
            moves_played[i] += 1

            # re-root onto the chosen child, keep its subtree
            child = root.children.get(action)
            if child is None:                     # defensive; visited actions
                child = Node(env.clone())         # always have children
            else:
                child.parent = None
                child.action_taken = None
            roots[i] = child

            if env.done:
                result_current = env.outcome_for_current_player()
                absolute_result = result_current * env.to_move
                for (t_state, t_pi, t_to_move) in trajectories[i]:
                    z = absolute_result * t_to_move
                    finished_examples.append((t_state, t_pi, z))
                active.remove(i)

    return finished_examples


if __name__ == "__main__":
    # smoke test + timing
    import time
    import network

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    net = network.Othello_Net(8).to(device)

    t0 = time.time()
    examples = generate_batched_selfplay_data(net, num_games=10, n=8, sims=100)
    dt = time.time() - t0

    zs = [z for (_, _, z) in examples]
    print(f"{len(examples)} examples from 10 games in {dt:.1f}s on {device}")
    print(f"z distribution: +1: {zs.count(1.0)}  -1: {zs.count(-1.0)}  0: {zs.count(0.0)}")
    # sanity: every example's pi sums to 1 and only on legal-shaped support
    for (s, pi, z) in examples[:5]:
        assert abs(pi.sum() - 1.0) < 1e-6
    print("pi normalisation OK")
