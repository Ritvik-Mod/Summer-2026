"""
Othello — pure game logic.

Board: 8x8 list of lists.
  1  = black
 -1  = white
  0  = empty

Black (1) moves first.
"""

EMPTY, BLACK, WHITE = 0, 1, -1
N = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]


def new_board():
    """Return a fresh board with the standard 4 disc starting position."""
    b = [[EMPTY] * N for _ in range(N)]
    b[3][3] = WHITE
    b[3][4] = BLACK
    b[4][3] = BLACK
    b[4][4] = WHITE
    return b


def on_board(r, c):
    return 0 <= r < N and 0 <= c < N


def discs_to_flip(board, player, r, c):
    """
    If player plays at (r, c), return the list of opponent discs that get
    flipped. Empty list means the move is illegal.
    """
    if board[r][c] != EMPTY:
        return []
    opponent = -player
    flips = []
    for dr, dc in DIRECTIONS:
        line = []
        rr, cc = r + dr, c + dc
        while on_board(rr, cc) and board[rr][cc] == opponent:
            line.append((rr, cc))
            rr, cc = rr + dr, cc + dc
        # the line must end on one of the player's own discs to be valid
        if line and on_board(rr, cc) and board[rr][cc] == player:
            flips.extend(line)
    return flips


def is_legal(board, player, r, c):
    return len(discs_to_flip(board, player, r, c)) > 0


def legal_moves(board, player):
    """Return list of (r, c) cells where `player` can legally move."""
    moves = []
    for r in range(N):
        for c in range(N):
            if board[r][c] == EMPTY and discs_to_flip(board, player, r, c):
                moves.append((r, c))
    return moves


def apply_move(board, player, r, c):
    """
    Play (r, c) for `player` on `board` (mutates it). Places the disc and
    flips all captured discs. Assumes the move is legal.
    """
    flips = discs_to_flip(board, player, r, c)
    board[r][c] = player
    for (fr, fc) in flips:
        board[fr][fc] = player
    return board


def has_any_move(board, player):
    return len(legal_moves(board, player)) > 0


def is_game_over(board):
    """Game ends when neither player has a legal move."""
    return not has_any_move(board, BLACK) and not has_any_move(board, WHITE)


def score(board):
    """Return (black_count, white_count)."""
    black = sum(row.count(BLACK) for row in board)
    white = sum(row.count(WHITE) for row in board)
    return black, white


def winner(board):
    """Return BLACK, WHITE, or 0 for a draw."""
    b, w = score(board)
    if b > w:
        return BLACK
    if w > b:
        return WHITE
    return 0


# ----------------------------- display + play -----------------------------

def print_board(board):
    symbols = {BLACK: "X", WHITE: "O", EMPTY: "."}
    print("   " + " ".join(str(c) for c in range(N)))
    for r in range(N):
        print(f"{r}  " + " ".join(symbols[board[r][c]] for c in range(N)))
    b, w = score(board)
    print(f"   black(X)={b}  white(O)={w}")


def play():
    board = new_board()
    player = BLACK

    while not is_game_over(board):
        moves = legal_moves(board, player)
        name = "Black (X)" if player == BLACK else "White (O)"

        if not moves:
            print(f"{name} has no moves — passing.")
            player = -player
            continue

        print_board(board)
        print(f"{name} to move. Legal: {moves}")
        try:
            raw = input("enter row col (e.g. '2 3'): ").split()
            r, c = int(raw[0]), int(raw[1])
        except (ValueError, IndexError):
            print("bad input, try again\n")
            continue

        if not is_legal(board, player, r, c):
            print("illegal move, try again\n")
            continue

        apply_move(board, player, r, c)
        player = -player

    print_board(board)
    w = winner(board)
    if w == BLACK:
        print("Black (X) wins!")
    elif w == WHITE:
        print("White (O) wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    play()