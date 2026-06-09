# AlphaZero Othello

An AlphaZero-style reinforcement learning agent for Othello, built from scratch in PyTorch.

The agent learns to play Othello entirely through self-play — no human game data, no handcrafted evaluation functions during training. A neural network provides move probabilities and position evaluation, which Monte Carlo Tree Search uses to generate training targets. The network then trains on those targets, plays stronger games, generates better targets, and so on.

## Architecture

**Neural network** (`network.py`): Two-headed ResNet — an initial conv block followed by 4 residual blocks (64 channels). The policy head outputs logits over all board positions + pass. The value head outputs a scalar ∈ [−1, +1] via tanh. Board-size agnostic.

**MCTS** (`mcts.py`): PUCT-based tree search with negamax sign flipping. The network provides prior probabilities at leaf nodes; visit counts after search form the improved policy target π. Dirichlet noise (α=0.3, ε=0.25) is injected at the root during self-play for exploration.

**Environment** (`othello_env.py`): The board is always stored from the perspective of the player to move (+1 = my discs, −1 = opponent). This means a single network handles both colours without any colour-specific logic. Action space is n² + 1 (last index = pass). Verified against a separate reference implementation over random game rollouts.

**Self-play** (`self_play.py`): Generates (state, π, z) training triples. Moves are sampled from π during play; z is back-filled with the actual game outcome (from each position's player's perspective) once the game ends.

**Training loop** (`training.py`): Each iteration: generate self-play games → train network (value MSE + policy cross-entropy) → evaluate → checkpoint. Supports resume from checkpoint with optimizer state preserved.

**Evaluation** (`evaluate.py`): Alpha-beta agent with a positional heuristic (corner weights, X/C-square penalties, mobility) and negamax pruning. This is the real benchmark — win rate vs random saturates early and stops being informative.

**GUI** (`gui.py`): Pygame interface where either side can be human, a trained model (any checkpoint), or the alpha-beta agent. Animated disc flips, legal-move hints (H key), auto-play toggle (A key). Useful for visually comparing checkpoints.

## How to run

### Requirements

```
python >= 3.10
torch >= 2.0
numpy
pygame  # for the GUI only
```

### Train from scratch

```bash
python training.py
```

Default config in `training.py` trains on 6×6. For 8×8 training (as used on the HPC), edit the call at the bottom or run with a modified config:

```python
train(num_iterations=40, games_per_iteration=50, sims=100, epochs=4, n=8, batch_size=64)
```

### Resume from checkpoint

```python
train(num_iterations=40, games_per_iteration=50, sims=100, epochs=4, n=8, batch_size=64,
      resume_from="checkpoint_iter11.pt")
```

### Evaluate a checkpoint

```bash
python evaluate.py checkpoint_iter15.pt 100 3 30
#                   ^checkpoint          ^sims ^AB-depth ^games
```

### Play in the GUI

Edit the config block at the top of `gui.py` to set the matchup and checkpoint paths, then:

```bash
python gui.py
```

Controls: **Space** = step one move, **A** = toggle auto-play, **H** = toggle hints, **R** = restart, **Esc** = quit.

## Training progress

Training on 8×8, single GPU (college HPC, PyTorch 2.4.1+cu121), sims=100, 50 games/iteration.

| Iteration | Loss   | vs Random | vs AB (d2) |
|-----------|--------|-----------|------------|
| 2         | 2.3124 | 0.80      | —          |
| 5         | 1.4990 | 0.65      | —          |
| 8         | 1.4770 | 1.00      | —          |
| 11        | 1.4613 | 0.90      | —          |
| 13        | 1.5683 | 0.80      | 0.50       |
| 14        | 1.5698 | 0.95      | 0.00       |
| 16        | 1.5466 | 0.90      | 0.00       |

The agent beats a random player consistently. It does not yet beat alpha-beta at depth 2 — the loss plateau and weak alpha-beta performance are diagnosed as a training data problem: each iteration trains only on its own 50 games and then discards them.

## What's next

- **Replay buffer**: Accumulate training examples across iterations (rolling window of ~10–20 iterations) instead of discarding after each. This is the primary fix for the loss plateau.
- **Bump simulations to 200+** for sharper MCTS policy targets.
- **Temperature schedule**: Sample moves proportionally in the opening, switch to argmax in the endgame.
- **Batched self-play**: Run multiple games concurrently with batched GPU inference to reduce iteration time. Profiling shows the bottleneck is the pure-Python environment (`_flips` accounts for ~44% of self-play time at 25M calls), not the network forward pass.

## File overview

```
othello_env.py   — RL environment (perspective-based board, legal moves, step)
network.py       — Two-headed ResNet (policy + value)
mcts.py          — PUCT tree search, play_game, evaluate_vs_random
self_play.py     — Self-play data generation (state, π, z)
training.py      — Training loop with checkpoint save/resume
evaluate.py      — Alpha-beta agent + evaluation harness
gui.py           — Pygame GUI for human/model/alphabeta matchups
```
