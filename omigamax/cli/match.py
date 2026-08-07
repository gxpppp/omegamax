"""Todo 20: match CLI -- engine-vs-engine auto-play with win-rate / ELO report.

Plan (todo 20, authoritative):

* ``match --engine2 katago --games N`` -- omigamax vs KataGo. omigamax runs
  **in-process** as the todo-18 :class:`GTPEngine` at ``--sims`` MCTS
  simulations per move under the todo-15 evaluation discipline (no Dirichlet
  root noise, ``tau = 0`` argmax). KataGo runs as a **GTP subprocess**
  (``katago gtp -model <weights> -config <cfg>`` -- the plan's "GTP 管道互叫":
  flushed pipe + CRLF-tolerant frame reads, same as todo 18), driven with the
  same rule components todo 5 validated (simple ko, AREA scoring, tax NONE,
  no suicide). ``match --engine2 random`` plays omigamax against a
  uniform-random-legal engine in-process (the plan's vs-random milestone
  command, Oracle F4).
* **colour alternation**: game index ``i`` even -> engine1 is black, odd ->
  engine1 is white (komi 7.5 favours white, so alternating keeps the match
  fair -- same protocol as todos 12/15). Seeds are ``master_seed + i``.
* a local :class:`Board` **arbitrates legality**: every ``genmove`` response
  is parsed and checked against the side to move before being accepted; a
  rejected play or an illegal returned move is an engine error -- recorded
  with the engine stderr tail in the per-game record, never silently accepted.
  Two consecutive passes end the game (Tromp-Taylor); ``--max-moves``
  force-terminates and scores (timeout protection, todo-13 style).
* the report prints per-game results (winner, colour, moves, result), the
  aggregate win rate of engine1, and the standard ELO difference
  ``ELO(p) = 400 * log10(p / (1 - p))`` -- reused from todo 15's
  :func:`omigamax.train.evaluate.elo_from_winrate` (the plan's *external*
  vs-KataGo/vs-random metric, kept separate from the internal gate ELO).
  Every game's SGF is written to ``--out-dir`` (default ``logs/matches/``).
* ``play`` subcommand: human-vs-engine terminal play -- see
  :mod:`omigamax.cli.play`.

Win-rate is computed over the games that completed without an engine error
(``completed = games - errors``; ``errors`` is reported separately). A 100%
or 0% run clamps to +-2400 ELO instead of ``+-inf``.

Usage::

    uv run python -m omigamax.cli.match --engine2 katago --games 9
    uv run python -m omigamax.cli.match --engine2 random --games 20
    uv run python -m omigamax.cli.match match --engine2 random --games 5 --sims 40
    uv run python -m omigamax.cli.match play --model models/best.pt --vs random
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from omigamax.config import load_config
from omigamax.gtp.gtp import GTPCommandError, GTPEngine, parse_vertex, to_gtp
from omigamax.rules import BLACK, WHITE, Board
from omigamax.rules.sgf import export_sgf
from omigamax.train.evaluate import elo_from_winrate

# Rule components KataGo must use to match omigamax's engine (todo 5 locked).
from omigamax.cli.rule_consistency import KATAGO_RULE_SETUP

DEFAULT_MODEL = "models/best.pt"
DEFAULT_GAMES = 20
DEFAULT_KOMI = 7.5
DEFAULT_SIZE = 19
DEFAULT_MAX_MOVES = 1000          # timeout protection (todo-13 style)
DEFAULT_TIMEOUT_S = 120.0         # plan QA: 引擎进程挂起 -> 超时 kill（120s/手）
DEFAULT_OUT_DIR = "logs/matches"
DEFAULT_KATAGO_DIR = "tools/katago"
DEFAULT_EVIDENCE = ".omo/evidence/omigamax-go/task-20-match.json"
DEFAULT_LOG = ".omo/evidence/omigamax-go/task-20-match.txt"

# Color tokens sent over GTP (lowercase letters are accepted by both omigamax's
# parse_color and KataGo's GTP parser).
_COLOR_TOK = {BLACK: "b", WHITE: "w"}


# ---------------------------------------------------------------------------
# GTP command interface
# ---------------------------------------------------------------------------

class GTPClient:
    """A thin GTP v2 client over a child process (e.g. ``katago gtp``).

    Every response frame is read to its blank-line terminator (multi-line
    bodies are joined). Reads happen on a helper thread so a hung child is
    killed at ``timeout`` seconds (``select`` is unavailable on Windows
    pipes). CRLF line endings from the peer are tolerated (rstrip ``\\r``).
    """

    def __init__(self, argv, *, name, timeout=DEFAULT_TIMEOUT_S, cwd=None) -> None:
        self.name = name
        self.timeout = float(timeout)
        self.proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd) if cwd is not None else None,
        )
        self._stderr_lines: "list[str]" = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            for line in self.proc.stderr:
                self._stderr_lines.append(line.rstrip("\r\n"))
        except Exception:  # pragma: no cover - process teardown
            pass

    def _readline(self, deadline: float) -> str:
        remaining = deadline - time.time()
        if remaining <= 0:
            self.proc.kill()
            raise TimeoutError(
                f"{self.name} timed out (>{self.timeout:g}s per command); "
                f"stderr tail: {self.stderr_tail(8)}"
            )
        box: "queue.Queue" = queue.Queue()

        def _read() -> None:
            try:
                box.put(("line", self.proc.stdout.readline()))
            except Exception as exc:  # pragma: no cover - teardown races
                box.put(("err", exc))

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=remaining)
        if t.is_alive():
            self.proc.kill()
            raise TimeoutError(
                f"{self.name} timed out (>{self.timeout:g}s per command); "
                f"stderr tail: {self.stderr_tail(8)}"
            )
        kind, value = box.get_nowait()
        if kind == "err":
            raise RuntimeError(f"{self.name} pipe read failed: {value}") from value
        line = value
        if line == "":
            raise RuntimeError(
                f"{self.name} closed stdout; stderr tail: {self.stderr_tail(8)}"
            )
        return line

    def command(self, cmd: str) -> "tuple[bool, str]":
        """Send one GTP command; return ``(ok, response_text)``."""
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"{self.name} exited (rc={self.proc.returncode}); "
                f"stderr tail: {self.stderr_tail(8)}"
            )
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + self.timeout
        first = self._readline(deadline).rstrip("\r\n")
        if not (first.startswith("= ") or first.startswith("?")):
            raise RuntimeError(f"{self.name} malformed GTP frame: {first!r}")
        ok = first.startswith("= ")
        body = [first[2:]]
        while True:
            line = self._readline(deadline).rstrip("\r\n")
            if line == "":
                break
            body.append(line)
        return ok, "\n".join(body)

    def stderr_tail(self, n: int = 8) -> "list[str]":
        return list(self._stderr_lines)[-n:]

    def close(self) -> None:
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except Exception:  # pragma: no cover - already dead
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # pragma: no cover - hung child
            self.proc.kill()


class GTPEngineInProcess:
    """Adapter: the todo-18 :class:`GTPEngine` behind the match ``command()``."""

    def __init__(self, engine: GTPEngine, name: str = "omigamax") -> None:
        self._engine = engine
        self.name = name

    def command(self, cmd: str) -> "tuple[bool, str]":
        frame = self._engine.handle_line(cmd)
        if frame is None:  # blank input line -> empty success
            return True, ""
        text = "\n".join(frame[:-1])
        ok = text.startswith("= ")
        return ok, text[2:]

    def stderr_tail(self, n: int = 8) -> "list[str]":
        return []

    def close(self) -> None:
        pass


class RandomEngine:
    """A uniform-random-legal engine behind the match ``command()`` interface.

    Picks uniformly among the legal points plus one pass slot (the same
    opponent todo 12's ``_RandomAgent`` and the plan's ``--engine2 random``
    use). Maintains its own board so the GTP commands it receives stay
    consistent.
    """

    def __init__(self, size: int = DEFAULT_SIZE, komi: float = DEFAULT_KOMI,
                 seed: int = 0) -> None:
        self.size = int(size)
        self.komi = float(komi)
        self.board = Board(self.size)
        self.rng = np.random.default_rng(int(seed))
        self.name = "random"

    def _genmove(self, color: int) -> "tuple[int, int] | None":
        points = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board.is_legal((r, c), color)
        ]
        idx = int(self.rng.integers(0, len(points) + 1))  # last slot = pass
        return None if idx == len(points) else points[idx]

    def command(self, cmd: str) -> "tuple[bool, str]":
        tokens = cmd.strip().split()
        if not tokens:
            return True, ""
        name = tokens[0].lower()
        try:
            if name == "boardsize":
                self.size = int(tokens[1])
                self.board = Board(self.size)
                return True, ""
            if name == "komi":
                self.komi = float(tokens[1])
                return True, ""
            if name == "clear_board":
                self.board = Board(self.size)
                return True, ""
            if name == "play":
                color = self._parse_color(tokens[1])
                move = parse_vertex(tokens[2], self.size)
                if not self.board.is_legal(move, color):
                    return False, "illegal move"
                self.board.play(move, color)
                return True, ""
            if name == "genmove":
                color = self._parse_color(tokens[1])
                move = self._genmove(color)
                self.board.play(move, color)
                return True, to_gtp(move, self.size)
            if name == "final_score":
                if not self.board.is_terminal():
                    return False, "game not finished"
                return True, self.board.result_string(self.komi)
            if name == "name":
                return True, "random"
            if name == "quit":
                return True, ""
            return True, ""
        except (GTPCommandError, ValueError, IndexError):
            return False, f"bad command: {cmd}"

    @staticmethod
    def _parse_color(token: str) -> int:
        c = token.strip().lower()
        if c in ("b", "black"):
            return BLACK
        if c in ("w", "white"):
            return WHITE
        raise GTPCommandError(f"invalid color: {token!r}")

    def stderr_tail(self, n: int = 8) -> "list[str]":
        return []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------

def make_omigamax_engine(model, sims, *, size, komi, seed, device, config):
    """An in-process todo-18 GTP engine over ``model`` (evaluation discipline)."""
    engine = GTPEngine(
        model_path=model,
        board_size=size,
        komi=komi,
        simulations=sims,
        device=device,
        seed=seed,
        config_path=config,
    )
    return GTPEngineInProcess(engine, name="omigamax")


def _locate_katago(katago_dir, binary=None, weights=None, config=None):
    """Locate ``katago.exe``, the weights and a gtp config under ``katago_dir``."""
    # Resolve to absolute paths: the subprocess runs with cwd=katago_dir, so a
    # relative -model/-config would resolve against the wrong directory.
    katago_dir = Path(katago_dir).resolve()
    if binary is None:
        for cand in (katago_dir / "eigen" / "katago.exe", katago_dir / "katago.exe"):
            if cand.exists():
                binary = cand
                break
    if binary is None or not Path(binary).exists():
        raise FileNotFoundError(
            f"KataGo binary not found under {katago_dir} "
            "(looked for eigen/katago.exe and katago.exe)"
        )
    binary = Path(binary).resolve()
    if weights is None:
        candidates = sorted(
            list(katago_dir.glob("*.bin.gz")) + list(katago_dir.glob("*.txt.gz")),
            key=lambda p: p.stat().st_size, reverse=True,
        )
        if candidates:
            weights = candidates[0]
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(f"KataGo weights not found under {katago_dir}")
    weights = Path(weights).resolve()
    if config is None:
        config = Path(binary).parent / "default_gtp.cfg"
    if not Path(config).exists():
        raise FileNotFoundError(f"KataGo gtp config not found: {config}")
    return Path(binary), Path(weights), Path(config).resolve()


def make_katago_engine(args, katago_visits=None):
    """Spawn a ``katago gtp`` subprocess and configure it for the match."""
    binary, weights, config = _locate_katago(
        args.katago_dir, args.binary, args.weights, args.katago_config
    )
    client = GTPClient(
        [str(binary), "gtp", "-model", str(weights), "-config", str(config)],
        name="katago",
        timeout=args.timeout,
        cwd=Path(args.katago_dir).resolve(),
    )
    setup = []
    for cmd, arg in KATAGO_RULE_SETUP:
        ok, resp = client.command(f"{cmd} {arg}")
        if not ok:
            client.close()
            raise RuntimeError(f"KataGo rejected setup '{cmd} {arg}': {resp}")
        setup.append(f"{cmd} {arg}")
    if katago_visits is not None and katago_visits > 0:
        # Cap KataGo's per-move search so a CPU/eigen build stays usable for
        # short evaluation runs (todo-5 style fallback notes); non-fatal.
        ok, resp = client.command(f"kata-set-param maxVisits {int(katago_visits)}")
        if not ok:
            setup.append(f"kata-set-param maxVisits {int(katago_visits)} "
                         f"(rejected: {resp})")
        else:
            setup.append(f"kata-set-param maxVisits {int(katago_visits)}")
    return client, {"binary": str(binary), "weights": str(weights),
                    "config": str(config), "setup": setup}


# ---------------------------------------------------------------------------
# one match game
# ---------------------------------------------------------------------------

def _stderr_tail(engine, n: int = 5) -> "list[str]":
    fn = getattr(engine, "stderr_tail", None)
    if callable(fn):
        try:
            return fn(n)
        except Exception:  # pragma: no cover - engine teardown
            return []
    return []


def play_match_game(engine_b, engine_w, *, size, komi, max_moves, seed,
                    engine1_name, engine1_is_black) -> dict:
    """Play one match game between ``engine_b`` (black) and ``engine_w``.

    Each engine exposes ``command(cmd) -> (ok, text)``. ``engine1_name`` /
    ``engine1_is_black`` are the reporting side (colour alternation is decided
    by the caller). A local :class:`Board` arbitrates legality; on any engine
    error the game is abandoned and an ``error`` record with the offending
    command and the engine stderr tail is returned.
    """
    size = int(size)
    komi = float(komi)
    max_moves = int(max_moves)
    board = Board(size)
    engine1_color = BLACK if engine1_is_black else WHITE
    # fresh board on both engines (GTP engines reset on clear_board)
    for eng in (engine_b, engine_w):
        eng.command(f"boardsize {size}")
        eng.command(f"komi {komi:g}")
        eng.command("clear_board")

    def fail(message, **extra) -> dict:
        return {
            "seed": int(seed),
            "engine1_color": "B" if engine1_is_black else "W",
            "winner": None,
            "engine1_win": False,
            "result": None,
            "moves": len(board.moves),
            "forced_terminal": False,
            "error": message,
            "stderr_b": _stderr_tail(engine_b),
            "stderr_w": _stderr_tail(engine_w),
            **extra,
        }

    move_list: "list[tuple]" = []
    moves = 0
    while not board.is_terminal() and moves < max_moves:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        engine = engine_b if color == BLACK else engine_w
        opp = engine_w if color == BLACK else engine_b
        tok = _COLOR_TOK[color]
        ok, resp = engine.command(f"genmove {tok}")
        if not ok:
            return fail(f"genmove rejected by {engine.name}: {resp!r}")
        move_gtp = resp.strip()
        try:
            move = parse_vertex(move_gtp, size)
        except GTPCommandError as exc:
            return fail(f"{engine.name} returned unparseable move "
                        f"{move_gtp!r}: {exc}")
        if not board.is_legal(move, color):
            return fail(f"{engine.name} returned illegal move {move_gtp!r} "
                        f"for {'black' if color == BLACK else 'white'}")
        board.play(move, color)
        move_list.append((move, color))
        ok2, resp2 = opp.command(f"play {tok} {move_gtp}")
        if not ok2:
            return fail(f"{opp.name} rejected play {move_gtp!r}: {resp2!r}")
        moves += 1
    winner = board.winner(komi)
    return {
        "seed": int(seed),
        "engine1_color": "B" if engine1_is_black else "W",
        "winner": winner,
        "engine1_win": bool(winner == ("B" if engine1_is_black else "W")),
        "result": board.result_string(komi),
        "moves": moves,
        "forced_terminal": not board.is_terminal(),
        "error": None,
        "stderr_b": [],
        "stderr_w": [],
        "move_list": [None if m is None else list(m) for m, _ in move_list],
        "color_list": ["B" if c == BLACK else "W" for _, c in move_list],
    }


def aggregate_results(records: "list[dict]", games: int) -> dict:
    """Win-rate / ELO aggregation over per-game records (pure, testable).

    ``winrate`` is engine1 wins over the games that completed without an
    error; ``errors`` is reported separately. ``elo_diff`` is the standard
    ELO difference ``400*log10(p/(1-p))`` (todo 15 helper), clamped to
    ``+-2400`` for degenerate 0%/100% runs.
    """
    games = int(games)
    completed = [r for r in records if not r.get("error")]
    engine1_wins = sum(1 for r in completed if r.get("engine1_win"))
    draws = sum(1 for r in completed if r.get("winner") is None)
    errors = len(records) - len(completed)
    total = len(completed)
    winrate = (engine1_wins / total) if total else 0.0
    return {
        "games": games,
        "completed": total,
        "errors": errors,
        "engine1_wins": engine1_wins,
        "draws": draws,
        "winrate": round(winrate, 6),
        "elo_diff": round(elo_from_winrate(winrate), 3) if total else 0.0,
    }


def write_game_sgf(rec: dict, sgf_dir, label: str, index: int, size: int,
                   komi: float) -> Path:
    """Replay a per-game record's move list into a fresh Board and export SGF."""
    sgf_dir = Path(sgf_dir)
    sgf_dir.mkdir(parents=True, exist_ok=True)
    board = Board(size)
    for mv, col in zip(rec["move_list"], rec["color_list"]):
        move = None if mv is None else tuple(mv)
        color = BLACK if col == "B" else WHITE
        board.play(move, color)
    e1_is_black = rec["engine1_color"] == "B"
    pb = label if e1_is_black else "katago"
    pw = "katago" if e1_is_black else label
    path = sgf_dir / f"match_{index:03d}.sgf"
    path.write_text(
        export_sgf(board, komi=komi, player_black=pb, player_white=pw),
        encoding="utf-8",
    )
    return path


def run_match(engine1, engine2, *, games, size, komi, max_moves, seed,
              sgf_dir=None, label="omigamax") -> dict:
    """Play ``games`` games between two ``command()`` engines.

    Colours alternate: game ``i`` even -> engine1 is black, odd -> engine1 is
    white. Seeds are ``seed + i``. Each completed game's SGF is written to
    ``sgf_dir`` (``logs/matches/`` by default). Returns the aggregation
    report with per-game detail.
    """
    games = int(games)
    size = int(size)
    komi = float(komi)
    max_moves = int(max_moves)
    t0 = time.perf_counter()
    records: "list[dict]" = []
    for i in range(games):
        seed_i = int(seed) + i
        e1_black = (i % 2 == 0)
        if e1_black:
            rec = play_match_game(
                engine1, engine2, size=size, komi=komi, max_moves=max_moves,
                seed=seed_i, engine1_name=label, engine1_is_black=True,
            )
        else:
            rec = play_match_game(
                engine2, engine1, size=size, komi=komi, max_moves=max_moves,
                seed=seed_i, engine1_name=label, engine1_is_black=False,
            )
        rec["index"] = i
        if sgf_dir is not None and not rec.get("error"):
            rec["sgf"] = str(write_game_sgf(
                rec, sgf_dir, label, i, size, komi,
            ))
        records.append(rec)
    summary = aggregate_results(records, games)
    summary["wall_time_s"] = round(time.perf_counter() - t0, 2)
    summary["games_detail"] = records
    return summary


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _print_match_report(report: dict, engine1_name: str, engine2_name: str) -> None:
    print(f"=== omigamax match (todo 20): {engine1_name} vs {engine2_name} ===",
          flush=True)
    print(f"games={report['games']} completed={report['completed']} "
          f"errors={report['errors']} komi=7.5 "
          f"wall_time={report['wall_time_s']}s", flush=True)
    for rec in report["games_detail"]:
        forced = " (max-moves forced)" if rec["forced_terminal"] else ""
        if rec["error"]:
            print(f"  game {rec['index']} seed={rec['seed']}: "
                  f"{engine1_name}={rec['engine1_color']} ERROR: "
                  f"{rec['error']}", flush=True)
            continue
        print(f"  game {rec['index']} seed={rec['seed']}: "
              f"{engine1_name}={rec['engine1_color']} winner={rec['winner']} "
              f"moves={rec['moves']} result={rec['result']}{forced} "
              f"sgf={rec.get('sgf', '-')}", flush=True)
    wr = report["winrate"]
    print(f"win rate ({engine1_name}): {report['engine1_wins']}/"
          f"{report['completed']} = {wr:.3f}", flush=True)
    print(f"ELO diff: {report['elo_diff']}", flush=True)
    print("RESULT: PASS (exit 0)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_match_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omigamax.cli.match",
        description="omigamax todo-20 match CLI: engine-vs-engine auto-play "
                    "(omigamax vs KataGo via GTP subprocess, or vs a "
                    "random-legal engine) with per-game results, aggregate "
                    "win rate and ELO. Colours alternate across games; SGFs "
                    "to logs/matches/. `play` subcommand: human-vs-engine "
                    "terminal play.",
    )
    parser.add_argument("--engine1", choices=("omigamax",), default="omigamax",
                        help="engine1 (default omigamax)")
    parser.add_argument("--engine2", choices=("katago", "random"),
                        default="random",
                        help="opponent: katago (GTP subprocess) or random "
                             "(in-process random-legal; default random)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"omigamax checkpoint (default {DEFAULT_MODEL})")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"number of games (default {DEFAULT_GAMES})")
    parser.add_argument("--sims", type=int, default=None,
                        help="omigamax MCTS simulations per move "
                             "(default: config simulations=200; pass a small "
                             "value for fast demos)")
    parser.add_argument("--board-size", type=int, default=None,
                        help="board edge (default: config board_size=19)")
    parser.add_argument("--komi", type=float, default=None,
                        help="komi on white (default: config komi=7.5)")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"move cap per game (default {DEFAULT_MAX_MOVES})")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"per-command timeout for the GTP subprocess "
                             f"(default {DEFAULT_TIMEOUT_S:g}s)")
    parser.add_argument("--katago-dir", type=str, default=DEFAULT_KATAGO_DIR,
                        help=f"directory holding the KataGo build + weights "
                             f"(default {DEFAULT_KATAGO_DIR})")
    parser.add_argument("--binary", type=str, default=None,
                        help="path to katago.exe (overrides auto-locate)")
    parser.add_argument("--weights", type=str, default=None,
                        help="path to KataGo weights (overrides auto-locate)")
    parser.add_argument("--config-file", dest="katago_config", type=str,
                        default=None,
                        help="path to a KataGo gtp config (overrides auto-locate)")
    parser.add_argument("--katago-visits", type=int, default=None,
                        help="cap KataGo's per-move search via "
                             "kata-set-param maxVisits (keeps CPU builds "
                             "usable for short runs)")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help=f"SGF output dir (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--no-sgf", action="store_true",
                        help="skip writing game SGFs")
    parser.add_argument("--evidence", type=str, default=DEFAULT_EVIDENCE,
                        help=f"result JSON path (default {DEFAULT_EVIDENCE})")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG,
                        help=f"text log path (default {DEFAULT_LOG})")
    return parser


def match_main(argv: "list[str] | None" = None) -> int:
    parser = build_match_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    size = int(args.board_size if args.board_size is not None
                else cfg.get("board_size", DEFAULT_SIZE))
    komi = float(args.komi if args.komi is not None
                 else cfg.get("komi", DEFAULT_KOMI))
    sims = int(args.sims if args.sims is not None
               else cfg.get("simulations", 200))

    if not Path(args.model).exists():
        print(f"error: model file not found: {args.model}",
              file=sys.stderr, flush=True)
        return 1

    engine1 = make_omigamax_engine(
        args.model, sims, size=size, komi=komi, seed=args.seed,
        device=args.device, config=args.config,
    )
    engine2 = None
    engine2_name = args.engine2
    katago_info = None
    if args.engine2 == "random":
        engine2 = RandomEngine(size=size, komi=komi, seed=args.seed)
    elif args.engine2 == "katago":
        engine2, katago_info = make_katago_engine(args, katago_visits=args.katago_visits)

    try:
        report = run_match(
            engine1, engine2, games=args.games, size=size, komi=komi,
            max_moves=args.max_moves, seed=args.seed,
            sgf_dir=None if args.no_sgf else args.out_dir,
            label="omigamax",
        )
    finally:
        engine1.close()
        if engine2 is not None:
            engine2.close()

    if katago_info is not None:
        report["katago"] = katago_info
    report["engine1"] = args.engine1
    report["engine2"] = args.engine2
    report["model"] = args.model
    report["sims"] = sims
    report["board_size"] = size
    report["komi"] = komi
    report["max_moves"] = args.max_moves
    report["seed"] = args.seed
    report["sgf_dir"] = None if args.no_sgf else args.out_dir

    _print_match_report(report, "omigamax", engine2_name)

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"evidence written: {path}", flush=True)
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "=== TODO 20: match CLI ===",
            f"command: python -m omigamax.cli.match --engine1 {args.engine1} "
            f"--engine2 {args.engine2} --model {args.model} --games {args.games} "
            f"--sims {sims} --board-size {size} --komi {komi} --seed {args.seed}",
        ]
        if katago_info is not None:
            lines.append(f"katago binary: {katago_info['binary']}")
            lines.append(f"katago weights: {katago_info['weights']}")
            lines.append(f"katago config: {katago_info['config']}")
            lines += [f"  setup: {s}" for s in katago_info["setup"]]
        for rec in report["games_detail"]:
            if rec["error"]:
                lines.append(
                    f"game {rec['index']}: ERROR {rec['error']} "
                    f"stderr_b={rec['stderr_b']} stderr_w={rec['stderr_w']}"
                )
            else:
                lines.append(
                    f"game {rec['index']}: omigamax={rec['engine1_color']} "
                    f"winner={rec['winner']} moves={rec['moves']} "
                    f"result={rec['result']}"
                )
        lines.append(
            f"win rate (omigamax): {report['engine1_wins']}/"
            f"{report['completed']} = {report['winrate']:.3f} "
            f"ELO diff {report['elo_diff']}"
        )
        lines.append(f"wall_time_s: {report['wall_time_s']}")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"log written: {log_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# module entry: `match` subcommand is the default; `play` dispatches to
# omigamax.cli.play so `python -m omigamax.cli.match play ...` works too.
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "play":
        from omigamax.cli.play import main as play_main

        return play_main(argv[1:])
    return match_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
