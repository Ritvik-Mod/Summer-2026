"""
Othello GUI -- watch or play against your trained agents.

Runs your REAL code (othello_env, network, mcts, evaluate). A clean clickable
board where EITHER side can be: a human, a trained model (any checkpoint), or
the alpha-beta agent. Set the matchup at the top and run:  python gui.py

MATCHUP EXAMPLES (edit BLACK_PLAYER / WHITE_PLAYER):
  - you vs model:        BLACK="human",  WHITE="model"
  - model vs alpha-beta: BLACK="model",  WHITE="alphabeta"
  - model vs model:      BLACK="model",  WHITE="model"   (use BLACK_CKPT/WHITE_CKPT
                                                          to compare two checkpoints)

Controls:
  SPACE = step one agent move (when AUTO_PLAY off)
  A     = toggle auto-play
  H     = toggle legal-move hints (for human players)
  R     = restart
  Esc   = quit

Requires: pip install pygame
"""

import sys
import numpy as np
import pygame

import othello_env
import network
import mcts

# ============================ CONFIG -- edit here ===========================
N         = 8
SIMS      = 2000
AB_DEPTH  = 3

# who plays each colour: "human", "model", or "alphabeta"
BLACK_PLAYER = "human"
WHITE_PLAYER = "model"

# checkpoint file used by each side IF that side is "model".
# set them to different files to watch one model play another.
BLACK_CKPT = "checkpoint_iter10.pt"
WHITE_CKPT = "checkpoint_iter12.pt"

AUTO_PLAY      = True     # auto-advance agent moves (else press SPACE)
MOVE_DELAY_MS  = 600      # pause between agent moves in auto mode
SHOW_HINTS     = True     # show legal moves for humans (toggle in-game with H)

FLIP_ANIM      = True     # animate disc flips one-by-one
FLIP_STEP_MS   = 110      # delay between each disc flipping
PLACE_HOLD_MS  = 280      # pause after placing a disc, before flips start
# ===========================================================================

BG       = (22, 26, 33)
FELT     = (34, 110, 86)
FELT_DK  = (29, 96, 75)
GRID     = (18, 60, 47)
EBONY    = (26, 28, 34)
IVORY    = (240, 240, 233)
IVORY_RIM= (200, 202, 196)
HINT     = (243, 224, 130)
LASTMV   = (224, 130, 86)
TEXT     = (230, 234, 242)
SUBTLE   = (146, 158, 176)
GOOD     = (122, 204, 172)

CELL   = 70
MARGIN = 36
TOPBAR = 96
W = N * CELL + 2 * MARGIN
H = N * CELL + 2 * MARGIN + TOPBAR


# --------------------------------------------------------- model / players
def load_net(ckpt_path):
    net = network.Othello_Net(N)
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
    net.eval()
    print(f"loaded {ckpt_path}")
    return net


def make_player(kind, net):
    if kind == "human":
        return None
    if kind == "model":
        return lambda env: mcts.mcts_player(env, net, SIMS)
    if kind == "alphabeta":
        from evaluate import AlphaBetaAgent
        return AlphaBetaAgent(AB_DEPTH)
    raise ValueError(f"unknown player kind: {kind}")


def label(kind, side):
    if kind == "model":
        f = BLACK_CKPT if side == 1 else WHITE_CKPT
        return f"Model ({f})"
    return {"human": "Human", "alphabeta": f"Alpha-Beta d{AB_DEPTH}"}[kind]


# --------------------------------------------------------------- rendering
def absolute_board(env):
    return env.board if env.to_move == 1 else -env.board


def draw(screen, env, fonts, status, last_move, show_hints, anim):
    big, mid, small = fonts
    screen.fill(BG)

    absb = absolute_board(env).copy()

    # mid-animation: env not yet stepped; render placed disc + flips-so-far
    if anim is not None:
        pr, pc, pcolor = anim["placed"]
        absb[pr, pc] = pcolor
        if anim["held"]:
            for i in range(anim["shown"]):
                fr, fc = anim["pending"][i]
                absb[fr, fc] = pcolor

    screen.blit(big.render("OTHELLO", True, TEXT), (MARGIN, 20))
    matchup = f"B: {label(BLACK_PLAYER,1)}   W: {label(WHITE_PLAYER,-1)}"
    screen.blit(small.render(matchup, True, SUBTLE), (MARGIN, 62))

    black = int((absb == 1).sum()); white = int((absb == -1).sum())
    sc = mid.render(f"B {black}   W {white}", True, TEXT)
    screen.blit(sc, (W - MARGIN - sc.get_width(), 26))
    st = small.render(status, True, GOOD if env.done else HINT)
    screen.blit(st, (W - MARGIN - st.get_width(), 60))
    hint_lbl = small.render(f"hints: {'on' if show_hints else 'off'} (H)", True, SUBTLE)
    screen.blit(hint_lbl, (W - MARGIN - hint_lbl.get_width(), 76))

    bx, by = MARGIN, MARGIN + TOPBAR
    pygame.draw.rect(screen, FELT, (bx-6, by-6, N*CELL+12, N*CELL+12), border_radius=12)
    for r in range(N):
        for c in range(N):
            x, y = bx + c*CELL, by + r*CELL
            pygame.draw.rect(screen, FELT if (r+c) % 2 == 0 else FELT_DK, (x, y, CELL, CELL))
            pygame.draw.rect(screen, GRID, (x, y, CELL, CELL), 1)

    to_move_kind = BLACK_PLAYER if env.to_move == 1 else WHITE_PLAYER
    if show_hints and to_move_kind == "human" and not env.done and anim is None:
        for a in env.legal_actions():
            r, c = divmod(a, N)
            pygame.draw.circle(screen, HINT,
                               (bx + c*CELL + CELL//2, by + r*CELL + CELL//2), 6)

    if last_move is not None and last_move != env.PASS and anim is None:
        r, c = divmod(last_move, N)
        pygame.draw.rect(screen, LASTMV, (bx + c*CELL, by + r*CELL, CELL, CELL), 3)

    rad = CELL//2 - 7
    for r in range(N):
        for c in range(N):
            v = absb[r, c]
            if v == 0:
                continue
            cx, cy = bx + c*CELL + CELL//2, by + r*CELL + CELL//2
            pygame.draw.circle(screen, (0, 0, 0), (cx, cy+2), rad)
            pygame.draw.circle(screen, EBONY if v == 1 else IVORY, (cx, cy), rad)
            if v == -1:
                pygame.draw.circle(screen, IVORY_RIM, (cx, cy), rad, 2)

    pygame.display.flip()


def cell_at(pos):
    bx, by = MARGIN, MARGIN + TOPBAR
    c = (pos[0] - bx) // CELL
    r = (pos[1] - by) // CELL
    if 0 <= r < N and 0 <= c < N:
        return int(r*N + c)
    return None


# --------------------------------------------------------------- main loop
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Othello")
    fonts = (pygame.font.SysFont("Georgia", 32, bold=True),
             pygame.font.SysFont("Georgia", 22),
             pygame.font.SysFont("Helvetica", 14))

    nets = {}
    def net_for(kind, ckpt):
        if kind != "model":
            return None
        if ckpt not in nets:
            nets[ckpt] = load_net(ckpt)
        return nets[ckpt]

    players = {1: make_player(BLACK_PLAYER, net_for(BLACK_PLAYER, BLACK_CKPT)),
               -1: make_player(WHITE_PLAYER, net_for(WHITE_PLAYER, WHITE_CKPT))}

    env = othello_env.OthelloEnv(N)
    last_move = None
    auto = AUTO_PLAY
    show_hints = SHOW_HINTS
    last_agent_time = 0

    anim = None
    last_flip_time = 0

    def status():
        if env.done:
            absb = absolute_board(env)
            b, w = int((absb == 1).sum()), int((absb == -1).sum())
            if b == w: return "draw"
            return "B wins" if b > w else "W wins"
        kind = BLACK_PLAYER if env.to_move == 1 else WHITE_PLAYER
        return f"{kind} to move"

    def begin_move(a):
        nonlocal last_move
        if a == env.PASS:
            env.step(a); last_move = a
            return None
        r, c = divmod(a, N)
        flips = env._flips(env.board, r, c)
        pcolor = env.to_move
        if not FLIP_ANIM or not flips:
            env.step(a); last_move = a
            return None
        return {"action": a, "placed": (r, c, pcolor),
                "pending": list(flips), "shown": 0, "color": pcolor, "held": False}

    clock = pygame.time.Clock()
    running = True
    while running:
        now = pygame.time.get_ticks()

        # ---- advance flip animation if active ----
        if anim is not None:
            if not anim["held"]:
                if now - last_flip_time >= PLACE_HOLD_MS:
                    anim["held"] = True
                    last_flip_time = now
            elif now - last_flip_time >= FLIP_STEP_MS:
                anim["shown"] += 1
                last_flip_time = now
                if anim["shown"] >= len(anim["pending"]):
                    env.step(anim["action"])
                    last_move = anim["action"]
                    anim = None
            draw(screen, env, fonts, status(), last_move, show_hints, anim)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            clock.tick(60)
            continue

        to_move_fn = players[env.to_move]
        is_human = to_move_fn is None

        draw(screen, env, fonts, status(), last_move, show_hints, None)

        if not env.done and not env.legal_actions():
            env.step(env.PASS); last_move = env.PASS
            continue

        if not env.done and not is_human:
            if auto and (now - last_agent_time) >= MOVE_DELAY_MS:
                a = to_move_fn(env)
                anim = begin_move(a)
                last_agent_time = now
                last_flip_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env = othello_env.OthelloEnv(N); last_move = None; anim = None
                elif event.key == pygame.K_a:
                    auto = not auto
                elif event.key == pygame.K_h:
                    show_hints = not show_hints
                elif event.key == pygame.K_SPACE and not is_human and not env.done:
                    a = to_move_fn(env)
                    anim = begin_move(a); last_flip_time = now
            elif event.type == pygame.MOUSEBUTTONDOWN and is_human and not env.done:
                a = cell_at(event.pos)
                if a is not None and a in env.legal_actions():
                    anim = begin_move(a); last_flip_time = now

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
