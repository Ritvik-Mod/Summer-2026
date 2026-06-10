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
SIMS      = 1000
AB_DEPTH  = 3

# who plays each colour: "human", "model", or "alphabeta"
BLACK_PLAYER = "model"
WHITE_PLAYER = "model"

# checkpoint file used by each side IF that side is "model".
# set them to different files to watch one model play another.
BLACK_CKPT = "checkpoint_iter102.pt"
WHITE_CKPT = "checkpoint_iter104.pt"

AUTO_PLAY      = True     # auto-advance agent moves (else press SPACE)
MOVE_DELAY_MS  = 600      # pause between agent moves in auto mode
SHOW_HINTS     = True     # show legal moves for humans (toggle in-game with H)

FLIP_ANIM      = False     # animate disc flips one-by-one
FLIP_STEP_MS   = 300      # delay between each disc flipping
PLACE_HOLD_MS  = 350      # pause after placing a disc, before flips start

AGENT_PASS_MS  = 700      # how long to show "<side> passes" before auto-passing
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


def draw(screen, env, fonts, status, last_move, show_hints, anim, pass_prompt=None):
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

    # centered overlay banner when a side has no legal move and must pass
    if pass_prompt is not None:
        line1, line2 = pass_prompt
        panel_w, panel_h = int(N*CELL*0.78), 96
        px = (W - panel_w) // 2
        py = by + (N*CELL - panel_h) // 2
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((18, 22, 28, 232))
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, HINT, (px, py, panel_w, panel_h), 2, border_radius=10)
        t1 = mid.render(line1, True, TEXT)
        t2 = small.render(line2, True, HINT)
        screen.blit(t1, (W//2 - t1.get_width()//2, py + 24))
        screen.blit(t2, (W//2 - t2.get_width()//2, py + 58))

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

    # pass-handling state:
    #   pass_prompt holds the banner text while we wait/pace a forced pass.
    #   pass_kind   is "human" (wait for input) or "agent" (timed auto-pass).
    #   pass_since  is the tick the agent-pass banner appeared.
    pass_prompt = None
    pass_kind = None
    pass_since = 0

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
        print(f"\n--- {'B' if env.to_move==1 else 'W'} plays ({r},{c}) ---")
        print(env.render())
        print(f"flips {len(flips)} disc(s): {flips}")
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

        # ---- forced-pass handling (no legal move for side to move) ----
        if pass_prompt is not None:
            # a pass is pending; draw the banner and wait/pace before applying.
            draw(screen, env, fonts, status(), last_move, show_hints, None, pass_prompt)
            do_pass = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif pass_kind == "human" and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        do_pass = True
                elif event.type == pygame.MOUSEBUTTONDOWN and pass_kind == "human":
                    do_pass = True
            if pass_kind == "agent" and now - pass_since >= AGENT_PASS_MS:
                do_pass = True
            if do_pass:
                env.step(env.PASS); last_move = env.PASS
                pass_prompt = pass_kind = None
                last_agent_time = now   # pace the next agent move from here
            clock.tick(60)
            continue

        draw(screen, env, fonts, status(), last_move, show_hints, None)

        # detect a forced pass and ARM the banner (instead of passing instantly)
        if not env.done and not env.legal_actions():
            side = "Black" if env.to_move == 1 else "White"
            if is_human:
                pass_prompt = (f"No legal move for {side}.",
                               "You must pass — press SPACE or click.")
                pass_kind = "human"
            else:
                pass_prompt = (f"{side} has no legal move.", "Passing\u2026")
                pass_kind = "agent"
                pass_since = now
            continue

        if not env.done and not is_human:
            if auto and (now - last_agent_time) >= MOVE_DELAY_MS:
                a = to_move_fn(env)               # MCTS think-time burns wall clock
                anim = begin_move(a)
                last_agent_time = pygame.time.get_ticks()
                last_flip_time  = pygame.time.get_ticks()  # re-read AFTER thinking
                                                            # so flip hold isn't pre-elapsed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env = othello_env.OthelloEnv(N); last_move = None; anim = None
                    pass_prompt = pass_kind = None
                elif event.key == pygame.K_a:
                    auto = not auto
                elif event.key == pygame.K_h:
                    show_hints = not show_hints
                elif event.key == pygame.K_SPACE and not is_human and not env.done:
                    a = to_move_fn(env)
                    anim = begin_move(a); last_flip_time = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEBUTTONDOWN and is_human and not env.done:
                a = cell_at(event.pos)
                if a is not None and a in env.legal_actions():
                    anim = begin_move(a); last_flip_time = now

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
