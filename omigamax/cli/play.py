"""Todo 20: human-vs-engine CLI play with terminal board rendering.

Plan (todo 20, authoritative): ``play`` subcommand -- 人机 CLI 对弈（终端棋盘
渲染）. The human reads GTP coordinates from stdin (``D4``, ``pass``, ``quit``);
an illegal / unparseable move is rejected and re-prompted. The engine replies
via MCTS (``--vs omigamax``, default, at ``--sims`` simulations under the
todo-15 evaluation discipline -- no noise, tau=0) or via a seeded
uniform-random-legal engine (``--vs random``, the acceptance's fast path).
The game ends at two consecutive passes (Tromp-Taylor); the score is reported
from the final position with komi on white.

Terminal rendering: a plain monospace text board (column letters A-T skipping
I, row numbers 1..N with row 1 at the bottom edge, GTP orientation), last
move shown lower-case, every move echoed, and the final result printed.

:func:`play_session` is the reusable driver (the CLI wraps sys.std*). It
accepts an injected ``engine`` (any ``command()``-interface object) as a test
seam for deterministic scripted-session tests.

Usage::

    uv run python -m omigamax.cli.play --model models/best.pt
    uv run python -m omigamax.cli.play --model models/best.pt --vs random
    uv run python -m omigamax.cli.match play --model models/best.pt --vs random
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omigamax.config import load_config
from omigamax.gtp.gtp import GTPCommandError, GTPEngine, parse_vertex
from omigamax.rules import BLACK, WHITE, Board
from omigamax.rules.sgf import export_sgf
from omigamax.cli.match import GTPEngineInProcess, RandomEngine

DEFAULT_MODEL = "models/best.pt"
DEFAULT_KOMI = 7.5
DEFAULT_SIZE = 19
DEFAULT_MAX_MOVES = 1000

_COLUMNS = "ABCDEFGHJKLMNOPQRST"
_COLOR_NAME = {BLACK: "Black", WHITE: "White"}
_COLOR_TOK = {BLACK: "b", WHITE: "w"}


# ---------------------------------------------------------------------------
# text board rendering
# ---------------------------------------------------------------------------

def render_board(board: Board, last_move=None) -> str:
    """A monospace text board, GTP orientation (row 1 at the bottom).

    Stones are ``B``/``W``, the last move is lower-case ``b``/``w``, empty
    intersections are ``.``. Column letters A-T skipping I.
    """
    size = board.size
    cols = _COLUMNS[:size]
    rows = [f"   {' '.join(cols)}"]
    for r in range(size):
        cells = []
        for c in range(size):
            v = board.get(r, c)
            if v == BLACK:
                ch = "b" if last_move == (r, c) else "B"
            elif v == WHITE:
                ch = "w" if last_move == (r, c) else "W"
            else:
                ch = "*" if last_move == (r, c) else "."
            cells.append(ch)
        rows.append(f"{size - r:2d} {' '.join(cells)} {size - r:2d}")
    rows.append(f"   {' '.join(cols)}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# the session driver
# ---------------------------------------------------------------------------

def play_session(
    model_path: "str | Path",
    *,
    vs: str = "omigamax",
    sims: "int | None" = None,
    komi: float = DEFAULT_KOMI,
    size: int = DEFAULT_SIZE,
    human_color: str = "B",
    seed: int = 0,
    max_moves: int = DEFAULT_MAX_MOVES,
    stdin=None,
    stdout=None,
    engine=None,
    sgf_dir=None,
) -> dict:
    """Run one human-vs-engine session; return a per-session record dict.

    ``stdin``/``stdout`` default to ``sys.std*`` (any file-like object works
    for scripted sessions). ``engine`` is an optional ``command()``-interface
    object (test seam); when omitted one is built from ``vs``. ``sgf_dir``
    optionally writes the finished game's SGF (``logs/matches/play_*.sgf``).
    """
    inp = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout
    cfg = load_config(None)
    sims = int(sims if sims is not None else cfg.get("simulations", 200))
    size = int(size)
    komi = float(komi)
    max_moves = int(max_moves)
    human = BLACK if str(human_color).upper() == "B" else WHITE

    if engine is None:
        if vs == "random":
            engine = RandomEngine(size=size, komi=komi, seed=seed)
        else:  # omigamax
            engine = GTPEngineInProcess(
                GTPEngine(
                    model_path=model_path, board_size=size, komi=komi,
                    simulations=sims, seed=seed,
                ),
                name="omigamax",
            )

    board = Board(size)
    move_list: "list[tuple]" = []
    last = None
    quit_requested = False
    while not board.is_terminal() and len(move_list) < max_moves:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        tok = _COLOR_TOK[color]
        out.write("\n" + render_board(board, last_move=last) + "\n")
        out.write(
            f"Move {len(board.moves) + 1} -- {_COLOR_NAME[color]} to play "
            f"(komi {komi:g})\n"
        )
        out.flush()
        if color == human:
            move_gtp = _read_human_move(inp, out, board, size, color)
            if move_gtp is None:  # quit requested
                quit_requested = True
                break
            ok, resp = engine.command(f"play {tok} {move_gtp}")
            if not ok:
                raise RuntimeError(f"engine rejected human move "
                                   f"{move_gtp!r}: {resp!r}")
        else:
            ok, resp = engine.command(f"genmove {tok}")
            if not ok:
                raise RuntimeError(f"engine genmove failed: {resp!r}")
            move_gtp = resp.strip()
            if move_gtp.lower() == "resign":
                move_gtp = "pass"
        try:
            move = parse_vertex(move_gtp, size)
        except GTPCommandError as exc:
            raise RuntimeError(f"engine produced unparseable move "
                               f"{move_gtp!r}: {exc}") from exc
        if not board.is_legal(move, color):
            raise RuntimeError(f"engine produced illegal move {move_gtp!r} "
                               f"for {_COLOR_NAME[color]}")
        board.play(move, color)
        move_list.append((move, color))
        last = move

    winner = board.winner(komi)
    result = board.result_string(komi) if not quit_requested else None
    out.write("\n" + render_board(board, last_move=last) + "\n")
    if quit_requested:
        out.write("Quit -- game not finished.\n")
    else:
        out.write(f"Game over: {result} (winner {winner}, komi {komi:g})\n")
    out.flush()

    sgf_path = None
    if sgf_dir is not None and not quit_requested:
        sgf_dir = Path(sgf_dir)
        sgf_dir.mkdir(parents=True, exist_ok=True)
        sgf_path = sgf_dir / f"play_{seed:04d}.sgf"
        sgf_path.write_text(
            export_sgf(board, komi=komi, player_black="human"
                       if human == BLACK else "engine",
                       player_white="engine" if human == BLACK else "human"),
            encoding="utf-8",
        )

    return {
        "vs": vs,
        "model": str(model_path),
        "human_color": "B" if human == BLACK else "W",
        "komi": komi,
        "board_size": size,
        "sims": sims if vs != "random" else None,
        "seed": seed,
        "winner": winner if not quit_requested else None,
        "result": result,
        "moves": len(move_list),
        "quit_requested": quit_requested,
        "forced_terminal": not board.is_terminal(),
        "sgf": str(sgf_path) if sgf_path is not None else None,
        "move_list": [None if m is None else list(m) for m, _ in move_list],
        "color_list": ["B" if c == BLACK else "W" for _, c in move_list],
    }


def _read_human_move(inp, out, board: Board, size: int, color: int):
    """Read one human move from ``inp``, re-prompting until legal.

    Returns the GTP move token, or ``None`` when the human asked to quit.
    EOF is treated as a pass so scripted sessions end cleanly.
    """
    while True:
        out.write("Your move (GTP coords, e.g. D4; 'pass'; 'quit'): ")
        out.flush()
        line = inp.readline()
        if line == "":
            return "pass"
        tok = line.strip()
        if not tok:
            continue
        if tok.lower() in ("quit", "exit", "q"):
            return None
        try:
            move = parse_vertex(tok, size)
        except GTPCommandError:
            out.write(f"  invalid coordinate: {tok!r} (expected e.g. D4 or pass)\n")
            out.flush()
            continue
        if not board.is_legal(move, color):
            out.write(f"  illegal move: {tok}\n")
            out.flush()
            continue
        return "pass" if move is None else tok.upper()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omigamax.cli.play",
        description="omigamax todo-20 human-vs-engine CLI play: terminal "
                    "board, GTP coordinate input, MCTS (or random) replies, "
                    "two-pass terminal with the score reported.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"omigamax checkpoint (default {DEFAULT_MODEL})")
    parser.add_argument("--vs", choices=("omigamax", "random"),
                        default="omigamax",
                        help="the engine the human plays against: omigamax "
                             "(MCTS; default) or random (seeded random-legal, "
                             "fast)")
    parser.add_argument("--sims", type=int, default=None,
                        help="omigamax MCTS simulations per move "
                             "(default: config simulations=200)")
    parser.add_argument("--komi", type=float, default=DEFAULT_KOMI,
                        help=f"komi on white (default {DEFAULT_KOMI:g})")
    parser.add_argument("--board-size", type=int, default=DEFAULT_SIZE,
                        help=f"board edge (default {DEFAULT_SIZE})")
    parser.add_argument("--human-color", choices=("B", "W"), default="B",
                        help="human colour (default B)")
    parser.add_argument("--seed", type=int, default=0, help="engine seed")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"move cap per session (default {DEFAULT_MAX_MOVES})")
    parser.add_argument("--sgf-dir", type=str, default=None,
                        help="write the finished game's SGF here "
                             "(default: none)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the session record JSON here")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    record = play_session(
        args.model,
        vs=args.vs,
        sims=args.sims,
        komi=args.komi,
        size=args.board_size,
        human_color=args.human_color,
        seed=args.seed,
        max_moves=args.max_moves,
        sgf_dir=args.sgf_dir,
    )
    if args.evidence:
        import json

        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"evidence written: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
