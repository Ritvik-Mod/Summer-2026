"""
Step 1 — RL interface for Othello.

Why this file exists separately from game.py:
  game.py is correct but built for a human game: N is hardcoded to 8, the board
  is from an absolute black/white view, moves are (r,c) tuples, and there's no
  fixed-size action vector or legal mask. MCTS and the neural net need a
  stricter contract. This file provides that contract WITHOUT modifying game.py.

The contract (everything MCTS / the network will call):
  - state is ALWAYS from the perspective of the player to move:
        +1 = my disc, -1 = opponent disc, 0 = empty.
    So a single network handles both players. This is the key trick.
  - action space is fixed size: n*n move-cells + 1 PASS action (the last index).
  - legal_mask() is a boolean vector over that fixed action space.
  - step(action) returns the next state already flipped to the NEXT player's
    perspective, so "the player to move always sees +1 = me" stays invariant.
  - outcome is reported from the perspective of the player who just moved.

The flip/legal rules themselves are identical to game.py, only re-expressed
to be board size parametric and perspective based.
"""

from __future__ import annotations
import numpy as np

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]


class OthelloEnv:
    def __init__(self, n: int = 8):
        assert n % 2 == 0 and n >= 4, "board size must be even and >= 4"
        self.n = n
        self.size = n * n          # number of board cells
        self.action_size = self.size + 1   # + 1 for PASS
        self.PASS = self.size      # PASS is the last action index
        self.reset()

    # ----------------------------------------------------------------- core
    def reset(self):
        """Standard 4 disc start. Board stored from the MOVER's perspective.

        At reset the mover is 'black'. We store +1 = mover. We also track
        `self.to_move` in absolute terms (+1 black / -1 white) only so we can
        render and reason about who actually moved; the network never sees it.
        """
        n = self.n
        b = np.zeros((n, n), dtype=np.int8)
        c = n // 2
        # standard cross: two of each colour. From mover(+1=black) perspective:
        b[c - 1, c - 1] = -1   # white
        b[c, c]         = -1   # white
        b[c - 1, c]     = 1    # black (mover)
        b[c, c - 1]     = 1    # black (mover)
        self.board = b              # always +1 = player to move
        self.to_move = 1            # absolute: +1 black, -1 white
        self.consecutive_passes = 0
        self.done = False
        return self.state()

    # ------------------------------------------------------ flip / legality
    def _flips(self, board, r, c):
        """Discs flipped if the MOVER (+1) plays (r,c). [] => illegal.
        Identical rule to game.py.discs_to_flip, but perspective-based."""
        n = self.n
        if board[r, c] != 0:
            return []
        out = []
        for dr, dc in DIRECTIONS:
            line = []
            rr, cc = r + dr, c + dc
            while 0 <= rr < n and 0 <= cc < n and board[rr, cc] == -1:  # opp
                line.append((rr, cc))
                rr, cc = rr + dr, cc + dc
            if line and 0 <= rr < n and 0 <= cc < n and board[rr, cc] == 1:  # mine
                out.extend(line)
        return out

    def legal_actions(self, board=None):
        """List of legal action indices for the MOVER. Empty => must PASS."""
        b = self.board if board is None else board
        n = self.n
        acts = []
        for r in range(n):
            for c in range(n):
                if b[r, c] == 0 and self._flips(b, r, c):
                    acts.append(r * n + c)
        return acts

    def legal_mask(self, board=None):
        """Boolean vector length action_size. If no moves, only PASS is legal."""
        b = self.board if board is None else board
        mask = np.zeros(self.action_size, dtype=bool)
        acts = self.legal_actions(b)
        if acts:
            mask[acts] = True
        else:
            mask[self.PASS] = True
        return mask

    # ----------------------------------------------------------- transition
    def step(self, action: int):
        """Apply `action` for the current mover. Returns
        (next_state, reward, done, info).

        reward is from the perspective of the player who JUST moved:
            +1 win / -1 loss / 0 draw or non-terminal.
        After the move the board is negated so the NEXT mover sees +1 = me.
        """
        assert not self.done, "game is over; call reset()"
        legal = self.legal_actions()

        if action == self.PASS:
            assert not legal, "illegal PASS: real moves exist"
            self.consecutive_passes += 1
        else:
            assert action in legal, f"illegal action {action}"
            r, c = divmod(action, self.n)
            flips = self._flips(self.board, r, c)
            self.board[r, c] = 1
            for (fr, fc) in flips:
                self.board[fr, fc] = 1
            self.consecutive_passes = 0

        # hand turn to opponent: flip perspective so mover is again +1
        self.board = -self.board
        self.to_move = -self.to_move

        # terminal: two passes in a row, or board full, or neither side can move
        if self.consecutive_passes >= 2 or np.count_nonzero(self.board == 0) == 0:
            self.done = True
        elif not self.legal_actions() and not self.legal_actions(board=-self.board):
            self.done = True

        reward = 0.0
        if self.done:
            # board is from NEW mover's view; the just-moved player is the other side
            mover_disc_diff = int(self.board.sum())       # >0 favors new mover
            just_moved_diff = -mover_disc_diff
            reward = float(np.sign(just_moved_diff))      # +1 / -1 / 0
        return self.state(), reward, self.done, {"legal": legal}
    
    def outcome_for_current_player(self):
        #valid only when self.done is true.
        #returns +1/-1/0 from the perspective of the player to move at this state
        diff = int(self.board.sum())
        return float(np.sign(diff))

    # ------------------------------------------------------- representations
    def state(self):
        """(2, n, n) float32 planes from the MOVER's perspective:
        plane 0 = my discs, plane 1 = opponent discs. This is the network input."""
        b = self.board
        return np.stack([(b == 1), (b == -1)], axis=0).astype(np.float32)

    def clone(self):
        e = OthelloEnv(self.n)
        e.board = self.board.copy()
        e.to_move = self.to_move
        e.consecutive_passes = self.consecutive_passes
        e.done = self.done
        return e

    def render(self):
        """ASCII, absolute view: black=X white=O."""
        absb = self.board if self.to_move == 1 else -self.board
        sym = {1: "X", -1: "O", 0: "."}
        n = self.n
        rows = ["   " + " ".join(str(i) for i in range(n))]
        for r in range(n):
            rows.append(f"{r}  " + " ".join(sym[int(absb[r, c])] for c in range(n)))
        b = int((absb == 1).sum()); w = int((absb == -1).sum())
        side = "X(black)" if self.to_move == 1 else "O(white)"
        return "\n".join(rows) + f"\n   X={b} O={w}  to move: {side}"
