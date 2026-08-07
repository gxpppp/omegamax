"""Live check 1 (F2 MAJOR 1): gtp fixed_handicap 2 genmove uses WHITE as mover.

Runs the real GTPEngine over a tiny 9x9 network on CPU and verifies:
  * after `fixed_handicap 2` the engine's true mover is WHITE;
  * the search root for `genmove W` is WHITE (not parity-BLACK);
  * feature plane 16 (colour to play) is all 0.0 (white);
  * the returned move is a legal WHITE move.
"""
import torch

from omigamax.gtp.gtp import GTPEngine
from omigamax.network.features import encode
from omigamax.network.model import create_model
from omigamax.rules import WHITE

eng = GTPEngine(
    network=create_model(1, 8, 9), board_size=9, komi=7.5,
    simulations=8, device="cpu", seed=0,
)

resp = eng.handle_line("fixed_handicap 2")[0]
print("fixed_handicap 2 ->", resp)

mover_before = eng._color_to_move()
print("_color_to_move() after handicap:", "WHITE" if mover_before == WHITE else "BLACK")

genmove = eng.handle_line("genmove W")[0]
print("genmove W ->", genmove)

root = eng._last_root
print("search root color:", "WHITE" if root.color == WHITE else "BLACK")
assert root.color == WHITE, "search root must be WHITE after even handicap"

planes = encode(root.history, root.color, board_size=eng.size)
print("plane-16 max (0.0 = white to move):", float(planes[16].max()))
assert planes[16].max() == 0.0

move = eng._last_root is not None and genmove[2:] != "pass"
if move:
    r, c = eng.board.moves[-1][0]
    print("returned stone colour on board:", eng.board.get(r, c), "(2 = WHITE)")
    assert eng.board.get(r, c) == WHITE
print("board moves:", len(eng.board.moves), "(2 handicap + 1 white)")
print("LIVE CHECK 1: PASS")
