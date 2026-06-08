"""
Stronger evaluation for trained Othello agents.

Two pieces:
  1. AlphaBetaAgent — a real heuristic opponent (corners, mobility, disc count)
     with alpha-beta pruning. Far harder than the random player; this is the
     benchmark that actually tells you if the agent is GOOD, not just "not broken".
  2. evaluate_vs_alphabeta — mirrors evaluate_vs_random but plays the trained
     MCTS agent against alpha-beta, alternating colours, over many games.

IMPORTANT — perspective:
  OthelloEnv stores the board ALWAYS from the player-to-move's view (+1 = me).
  So alpha-beta, which works on env clones, also reasons in that frame. The
  negamax recursion negates the value at each ply, exactly like the MCTS search.
"""

import numpy as np
import othello_env
import mcts


# ----- positional weights (corners huge, X/C-squares penalised) -----
# Built for 8x8; for 6x6 a simpler corner-weighted grid is generated.

def _weights(n):
    W = np.zeros((n, n), dtype=np.float32)
    # base
    W[:] = 3.0
    # edges
    W[0, :] = W[-1, :] = W[:, 0] = W[:, -1] = 8.0
    # corners (very good)
    for (r, c) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
        W[r, c] = 100.0
    # squares diagonally adjacent to corners (X-squares: very bad)
    for (r, c) in [(1, 1), (1, n - 2), (n - 2, 1), (n - 2, n - 2)]:
        W[r, c] = -25.0
    # squares orthogonally adjacent to corners (C-squares: bad)
    for (cr, cc) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
        for (dr, dc) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = cr + dr, cc + dc
            if 0 <= r < n and 0 <= c < n and W[r, c] not in (100.0,):
                if abs(r - cr) + abs(c - cc) == 1:
                    W[r, c] = -8.0
    return W


def _heuristic(env):
    """Score the position from the CURRENT player's perspective.
    board is +1 = me, -1 = opponent (env invariant)."""
    n = env.n
    W = _weights(n)
    b = env.board
    positional = float((b * W).sum())          # my good squares minus opp's
    # mobility: how many moves I have vs opponent
    my_moves = len(env.legal_actions())
    opp_moves = len(env.legal_actions(board=-b))
    mobility = 0.0
    if my_moves + opp_moves != 0:
        mobility = 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)
    return positional + 2.0 * mobility


def _alphabeta(env, depth, alpha, beta):
    """Negamax with alpha-beta. Returns (value, best_action) from the
    perspective of env's current player."""
    if env.done:
        # terminal: actual outcome from current player's view, scaled big
        return 1e6 * env.outcome_for_current_player(), None
    if depth == 0:
        return _heuristic(env), None

    legal = env.legal_actions()
    if not legal:
        # forced pass — recurse with negation (opponent's turn)
        child = env.clone()
        child.step(child.PASS)
        val, _ = _alphabeta(child, depth - 1, -beta, -alpha)
        return -val, child.PASS

    best_val, best_a = -1e18, legal[0]
    for a in legal:
        child = env.clone()
        child.step(a)
        val, _ = _alphabeta(child, depth - 1, -beta, -alpha)
        val = -val                     # negamax flip
        if val > best_val:
            best_val, best_a = val, a
        alpha = max(alpha, val)
        if alpha >= beta:
            break                      # prune
    return best_val, best_a


class AlphaBetaAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def __call__(self, env):
        _, a = _alphabeta(env.clone(), self.depth, -1e18, 1e18)
        if a is None:
            return env.PASS
        return int(a)


def evaluate_vs_alphabeta(net, sims, games=50, depth=3, n=8):
    """Trained MCTS agent vs alpha-beta(depth). Alternates colours.
    Returns win rate of the trained agent (draws count as half)."""
    ab = AlphaBetaAgent(depth)
    mcts_fn = lambda env: mcts.mcts_player(env, net, sims)

    wins = draws = 0
    for i in range(games):
        if i % 2 == 0:                 # MCTS black, AB white
            result = mcts.play_game(mcts_fn, ab, n)
            if result == 1: wins += 1
            elif result == 0: draws += 1
        else:                          # MCTS white, AB black
            result = mcts.play_game(ab, mcts_fn, n)
            if result == -1: wins += 1
            elif result == 0: draws += 1
    return (wins + 0.5 * draws) / games


if __name__ == "__main__":
    import sys, torch, network
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    sims  = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    n = 8                                    # <-- board size, match your model

    net = network.Othello_Net(n)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ckpt["net"])
        print(f"loaded {ckpt_path}")
    net.eval()

    wr_rand = mcts.evaluate_vs_random(net, sims, games=games, n=n)
    wr_ab   = evaluate_vs_alphabeta(net, sims, games=games, depth=depth, n=n)
    print(f"sims={sims}  depth={depth}  games={games}")
    print(f"  win rate vs random:      {wr_rand:.2f}")
    print(f"  win rate vs alpha-beta:  {wr_ab:.2f}")
