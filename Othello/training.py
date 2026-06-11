"""
Training loop, v2.

Changes over v1:
  1. REPLAY BUFFER — examples accumulate in a rolling deque (default 60k
     positions ~ last 15-20 iterations) instead of being discarded after one
     iteration. This is the fix for the flat-loss / thrashing plateau.
  2. BATCHED SELF-PLAY — uses batched_self_play.generate_batched_selfplay_data
     (all games concurrent, batched GPU inference) instead of sequential
     one-position-at-a-time self_play.py.
  3. STEPS, NOT EPOCHS — each iteration runs a fixed number of SGD steps on
     batches sampled uniformly from the buffer. Compute per iteration stays
     constant as the buffer grows (epochs over a growing buffer wouldn't).
  4. SYMMETRY AUGMENTATION — Othello is invariant under the 8 dihedral board
     symmetries. Each sampled example gets a random rotation/reflection
     applied to (state, board-part-of-pi); PASS prob is untouched. Free 8x
     effective data, big deal at 50 games/iteration.
  5. RESUME-PROOF — the buffer is saved to disk every iteration alongside the
     checkpoint, so a 12h-walltime kill loses nothing: resume restores net,
     optimizer, iteration AND the replay buffer.
  6. CONSISTENT NUMBERING — the log prints the same `iteration` number used
     in the checkpoint filename. No more +1 mismatch.

Run:        python training.py
Resume:     set resume_from="checkpoint_iterNN.pt" at the bottom.
"""

import os
import time
import collections

import torch
import numpy as np

import network
import mcts
import evaluate
from batched_self_play import generate_batched_selfplay_data

BUFFER_PATH = "replay_buffer.pt"


# ----------------------------------------------------------- augmentation
def transform_example(state, pi, n, k, flip):
    """Apply one of the 8 dihedral symmetries (k quarter-rotations, optional
    horizontal flip) consistently to the state planes and the board part of
    pi. The PASS entry is position-independent and is left alone."""
    s = state
    p_board = pi[:n * n].reshape(n, n)
    if k:
        s = np.rot90(s, k, axes=(1, 2))
        p_board = np.rot90(p_board, k)
    if flip:
        s = np.flip(s, axis=2)
        p_board = np.flip(p_board, axis=1)
    p = np.concatenate([p_board.reshape(-1), pi[n * n:]])
    return np.ascontiguousarray(s), np.ascontiguousarray(p)


def sample_batch(buffer, batch_size, n, augment=True):
    idx = np.random.randint(len(buffer), size=batch_size)
    states, pis, zs = [], [], []
    for j in idx:
        s, p, z = buffer[j]
        if augment:
            k = np.random.randint(4)
            flip = bool(np.random.randint(2))
            if k or flip:
                s, p = transform_example(s, p, n, k, flip)
        states.append(s)
        pis.append(p)
        zs.append(z)
    return (np.array(states, dtype=np.float32),
            np.array(pis, dtype=np.float32),
            np.array(zs, dtype=np.float32))


# ----------------------------------------------------------------- training
def train_steps(net, buffer, optimizer, batch_size, steps, n, augment=True):
    """Run `steps` SGD steps on batches sampled uniformly from the buffer.
    Returns mean loss over the steps."""
    device = next(net.parameters()).device
    net.train()
    loss_sum = 0.0
    for _ in range(steps):
        s, p, z = sample_batch(buffer, batch_size, n, augment)
        states = torch.from_numpy(s).to(device)
        pis    = torch.from_numpy(p).to(device)
        zs     = torch.from_numpy(z).unsqueeze(1).to(device)

        policy_logits, value = net(states)
        loss = network.compute_loss(policy_logits, value, pis, zs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / steps


def train(num_iterations=60,
          games_per_iteration=50,
          sims=200,
          n=8,
          batch_size=128,
          steps_per_iteration=250,
          buffer_size=60000,
          temp_moves=15,
          eval_sims=100,
          eval_games=20,
          resume_from=None):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    net = network.Othello_Net(n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"Training on {device}", flush=True)

    buffer = collections.deque(maxlen=buffer_size)
    start_iter = 0

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"] + 1
        print(f"resumed from {resume_from}, starting at iteration {start_iter}",
              flush=True)
        if os.path.exists(BUFFER_PATH):
            saved = torch.load(BUFFER_PATH, weights_only=False)
            buffer.extend(saved)
            print(f"restored replay buffer: {len(buffer)} examples", flush=True)
        else:
            print("no saved replay buffer found, starting empty", flush=True)

    for iteration in range(start_iter, num_iterations):
        t0 = time.time()

        # 1. self-play (batched, concurrent games)
        examples = generate_batched_selfplay_data(
            net, games_per_iteration, n=n, sims=sims, temp_moves=temp_moves)
        buffer.extend(examples)
        t_sp = time.time() - t0

        # 2. train on samples from the whole buffer
        t1 = time.time()
        avg_loss = train_steps(net, buffer, optimizer,
                               batch_size, steps_per_iteration, n)
        t_tr = time.time() - t1

        # 3. evaluate
        if iteration % 5 == 0:
            t2 = time.time()
            wr_rand = mcts.evaluate_vs_random(net, eval_sims, games=eval_games, n=n)
            wr_ab = evaluate.evaluate_vs_alphabeta(net, eval_sims,
                                                games=eval_games, depth=2, n=n)
            t_ev = time.time() - t2
        else:
            wr_rand = -1.0
            wr_ab = -1.0
            t_ev = -1.0

        print(f"Iter {iteration:02d}  Loss {avg_loss:.4f}  "
              f"buffer {len(buffer)}  "
              f"vsRandom {wr_rand:.2f}  vsAB(d2) {wr_ab:.2f}  "
              f"[selfplay {t_sp/60:.1f}m  train {t_tr/60:.1f}m  "
              f"eval {t_ev/60:.1f}m]", flush=True)

        # 4. checkpoint net+optimizer AND the buffer (walltime-proof)
        torch.save({"iteration": iteration,
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   f"checkpoint_iter{iteration}.pt")
        torch.save(list(buffer), BUFFER_PATH)


if __name__ == "__main__":
    train(
        num_iterations=120,
        games_per_iteration=50,
        sims=200,
        n=8,
        batch_size=128,
        steps_per_iteration=1000,
        buffer_size=60000,
        temp_moves=15,
        eval_sims=50,
        eval_games=10,
        # Warm start from your strongest existing checkpoint, or None for fresh:
        resume_from="checkpoint_iter59.pt",          # e.g. "checkpoint_iter15.pt"
    )
