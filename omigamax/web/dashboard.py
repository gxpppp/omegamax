"""KataGui-style web dashboard for the omigamax RL loop (read-only).

Serves a single-page monitor (``static/index.html``) plus three JSON APIs over
the training artifacts:

  * ``GET /api/train``  -- aggregated metrics from ``logs/train.jsonl``
    (train_step loss/lr series, eval_gate elo/winrate/replaced series, cycle
    boundaries, latest values, liveness from file mtimes).
  * ``GET /api/games``  -- newest-first list of ``data/selfplay/*.npz`` games
    (name, mtime, size, move_count, winner, komi, board_size), capped at 50.
  * ``GET /api/games/<id>`` -- full replay for one game: the absolute 0/1/2
    board grid at every position plus the derived move list (see
    ``reconstruct_game`` for the derivation method, verified against the rules
    engine and the stored result string).

All endpoints are cheap, poll-friendly and never raise on missing data
(missing files yield empty lists / ``None`` fields, never a 500).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_TRAIN_LOG = "logs/train.jsonl"
DEFAULT_SELFPLAY_DIR = "data/selfplay"
GAME_CAP = 50
STALE_AFTER_S = 600.0  # no artifact written for this long -> "training not alive"

# AGZ plane layout (omigamax.network.features.encode): plane 2t = current
# player's stones t moves ago, plane 2t+1 = opponent's stones t moves ago,
# plane 16 = colour to play (1 -> black). Stone codes: 1 = black, 2 = white.
PLANE_COLOR = 16
BLACK, WHITE = 1, 2


# ---------------------------------------------------------------------------
# train.jsonl parsing
# ---------------------------------------------------------------------------

def parse_train_log(path: "str | Path") -> dict:
    """Read ``logs/train.jsonl`` into a dashboard-ready dict.

    Returns ``{"latest", "steps", "evals", "cycles", "events", "file_mtime"}``
    where:

      * ``latest``  -- most recent state (last train_step merged with the last
        eval_gate, plus current cycle), or ``None`` if the log is empty.
      * ``steps``   -- list of train_step points ``{step, loss, lr, games,
        cycle, timestamp}`` (chronological, capped for the chart).
      * ``evals``   -- list of eval_gate points ``{step, cycle, winrate,
        replaced, elo, timestamp}`` (chronological).
      * ``cycles``  -- cycle_start boundaries ``{cycle, step, games, timestamp}``.
      * ``events``  -- total number of parsed JSON lines.

    Never raises: a missing or unreadable file yields empty structures.
    """
    path = Path(path)
    steps: list[dict] = []
    evals: list[dict] = []
    cycles: list[dict] = []
    events = 0
    mtime = None
    try:
        stat = path.stat()
        mtime = _iso(stat.st_mtime)
    except OSError:
        mtime = None
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events += 1
                    kind = ev.get("event")
                    if kind == "train_step":
                        steps.append(
                            {
                                "step": ev.get("step"),
                                "loss": ev.get("loss"),
                                "lr": ev.get("lr"),
                                "games": ev.get("games"),
                                "cycle": ev.get("cycle"),
                                "timestamp": ev.get("timestamp"),
                            }
                        )
                    elif kind == "eval_gate":
                        evals.append(
                            {
                                "step": ev.get("step"),
                                "cycle": ev.get("cycle"),
                                "winrate": ev.get("winrate"),
                                "replaced": ev.get("replaced"),
                                "elo": ev.get("elo"),
                                "timestamp": ev.get("timestamp"),
                            }
                        )
                    elif kind == "cycle_start":
                        cycles.append(
                            {
                                "cycle": ev.get("cycle"),
                                "step": ev.get("step"),
                                "games": ev.get("games"),
                                "games_generated": ev.get("games_generated"),
                                "timestamp": ev.get("timestamp"),
                            }
                        )
        except OSError:
            pass

    latest = None
    if steps or evals:
        last_step = steps[-1] if steps else {}
        last_eval = evals[-1] if evals else {}
        latest = {
            "step": last_step.get("step"),
            "loss": last_step.get("loss"),
            "lr": last_step.get("lr"),
            "games": last_step.get("games"),
            "cycle": last_step.get("cycle") or last_eval.get("cycle"),
            "elo": last_step.get("elo") if last_step.get("elo") is not None else last_eval.get("elo"),
            "winrate": last_eval.get("winrate"),
            "replaced": last_eval.get("replaced"),
            "timestamp": last_step.get("timestamp") or last_eval.get("timestamp"),
        }
    return {
        "latest": latest,
        "steps": steps,
        "evals": evals,
        "cycles": cycles,
        "events": events,
        "file_mtime": mtime,
    }


def _iso(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# npz game reading
# ---------------------------------------------------------------------------

def list_games(selfplay_dir: "str | Path", cap: int = GAME_CAP) -> list[dict]:
    """Newest-first metadata list of ``*.npz`` self-play games.

    Each entry: ``{id, name, mtime, size, move_count, winner, result,
    board_size, komi, forced_terminal}``. Files that fail to parse (e.g. a
    half-written file) are skipped. ``cap`` bounds the returned list.
    """
    d = Path(selfplay_dir)
    games = []
    if d.is_dir():
        for p in d.glob("*.npz"):
            meta = _read_game_meta(p)
            if meta is not None:
                games.append(meta)
    games.sort(key=lambda g: (g.get("mtime_raw") or 0, g.get("id") or ""), reverse=True)
    return games[: max(1, int(cap))]


def _read_game_meta(path: Path) -> "dict | None":
    """Scalar metadata for one npz; ``None`` if unreadable."""
    try:
        stat = path.stat()
        with np.load(path) as data:
            board_size = int(data["board_size"])
            komi = float(data["komi"])
            winner = _np_str(data["winner"])
            result = _np_str(data["result"])
            move_count = int(data["move_count"])
            forced = bool(data["forced_terminal"])
            z = np.asarray(data["z"])
            if winner in ("B", "W"):
                pass
            elif z.size and z[-1] > 0:
                winner = "B"
            elif z.size:
                winner = "W"
    except Exception:  # noqa: BLE001 - skip corrupt / half-written files
        return None
    return {
        "id": path.stem,
        "name": path.name,
        "mtime": _iso(stat.st_mtime),
        "mtime_raw": stat.st_mtime,
        "size": int(stat.st_size),
        "move_count": move_count,
        "winner": winner,
        "result": result,
        "board_size": board_size,
        "komi": komi,
        "forced_terminal": forced,
    }


def _np_str(value) -> str:
    s = str(np.asarray(value).item()) if np.asarray(value).ndim == 0 else str(value)
    return s


# ---------------------------------------------------------------------------
# game replay reconstruction (the core derivation, verified on real games)
# ---------------------------------------------------------------------------

def reconstruct_game(path: "str | Path") -> "dict | None":
    """Full replay data for one npz game, or ``None`` if unreadable.

    Derivation method (verified against ``omigamax.rules.Board`` on live
    self-play games):

      1. Absolute board grid at recorded position ``t`` (the position BEFORE
         move ``t``) is decoded from the perspective-dependent planes:
         plane 0/1 hold the *mover's* / *opponent's* stones, plane 16 says who
         moves, so the 0/1/2 grid maps mover=1/2, opponent=2/1 accordingly.
      2. Move ``t`` (for ``t < T-1``) is the unique point that gained a stone
         between position ``t`` and ``t+1`` (an empty point that becomes
         occupied); no new stone = pass. This reproduces every recorded
         position exactly (full-grid equality with the rules engine).
      3. The final move (``t == T-1``): the last recorded position is the
         terminal board for games that ended by two passes (the stored result
         scores it exactly). For max-moves force-terminated games the last
         stone is recovered from ``pi[T-1].argmax()`` (temperature is 0 after
         ``temperature_threshold``, so the sampled move IS the argmax) and the
         result string is reproduced exactly.

    Returns ``{"board_size", "komi", "winner", "result", "move_count",
    "forced_terminal", "positions", "moves", "final"}`` where ``positions`` is
    the list of ``T+1`` absolute grids (index ``i`` = board after ``i`` moves,
    last one = terminal), ``moves`` is the list of ``T`` derived moves
    (``{color, r, c}`` or ``{color, pass}``, with ``captured`` count) and
    ``final`` is the last move's dict (included in ``moves`` too).
    """
    path = Path(path)
    try:
        with np.load(path) as data:
            s = np.asarray(data["s"], dtype=np.float32)
            pi = np.asarray(data["pi"], dtype=np.float32)
            board_size = int(data["board_size"])
            komi = float(data["komi"])
            winner = _np_str(data["winner"])
            result = _np_str(data["result"])
            move_count = int(data["move_count"])
            forced = bool(data["forced_terminal"])
    except Exception:  # noqa: BLE001
        return None
    n = board_size
    T = int(s.shape[0])

    # absolute 0/1/2 grids of the T recorded positions (board BEFORE move t).
    grids = []
    for t in range(T):
        mover_black = bool(s[t, PLANE_COLOR].mean() > 0.5)
        cur, opp = (BLACK, WHITE) if mover_black else (WHITE, BLACK)
        grid = np.zeros((n, n), dtype=np.int8)
        grid[s[t, 0] > 0.5] = cur
        grid[s[t, 1] > 0.5] = opp
        grids.append(grid)

    # move t (t < T-1) = the unique point that gained a stone between the
    # recorded positions t and t+1 (verified: full-grid equality with the
    # rules engine on real self-play games).
    moves: list[dict] = []
    for t in range(T - 1):
        mover_black = bool(s[t, PLANE_COLOR].mean() > 0.5)
        color = "B" if mover_black else "W"
        prev, nxt = grids[t], grids[t + 1]
        new_pt = None
        captured = 0
        for r in range(n):
            for c in range(n):
                if prev[r, c] == 0 and nxt[r, c] != 0:
                    new_pt = (r, c)
                elif prev[r, c] != 0 and nxt[r, c] == 0:
                    captured += 1
        if new_pt is None:
            moves.append({"color": color, "pass": True, "captured": 0})
        else:
            r, c = new_pt
            moves.append({"color": color, "r": r, "c": c, "captured": captured})

    # final move: two-pass end -> pass (the last recorded position IS the
    # terminal board, verified by scoring it against the stored result);
    # forced (max-moves) end -> the sampled move, which is pi[T-1].argmax()
    # because temperature is 0 after temperature_threshold (verified).
    final = None
    if T > 0:
        mover_black = bool(s[T - 1, PLANE_COLOR].mean() > 0.5)
        color = "B" if mover_black else "W"
        if "Resign" in result:
            final = None  # game ended before the mover's last search
        elif forced:
            a = int(np.argmax(pi[T - 1]))
            if a >= n * n:
                final = {"color": color, "pass": True, "captured": 0}
            else:
                final = {"color": color, "r": a // n, "c": a % n, "captured": 0}
        else:
            final = {"color": color, "pass": True, "captured": 0}

    # positions: index i = board after i moves; grids[0] is the empty opening
    # board, grids[t] (t>0) is after move t-1, then the terminal board.
    positions = [g.tolist() for g in grids]
    if final is not None:
        moves.append(final)
        if "r" in final:
            last = [row[:] for row in positions[-1]]
            last[final["r"]][final["c"]] = BLACK if final["color"] == "B" else WHITE
            positions.append(last)
        else:
            positions.append([row[:] for row in positions[-1]])

    return {
        "board_size": n,
        "komi": komi,
        "winner": winner,
        "result": result,
        "move_count": move_count,
        "forced_terminal": forced,
        "positions": positions,
        "moves": moves,
        "final": final,
    }


def _empty_grid(n: int) -> list:
    return [[0] * n for _ in range(n)]


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app(
    train_log: "str | Path" = DEFAULT_TRAIN_LOG,
    selfplay_dir: "str | Path" = DEFAULT_SELFPLAY_DIR,
) -> Flask:
    """Build the dashboard app (testable with the Flask test client)."""
    train_log = Path(train_log)
    selfplay_dir = Path(selfplay_dir)
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.route("/")
    def index():
        return send_from_directory(str(STATIC_DIR), "index.html")

    @app.route("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.route("/api/train")
    def api_train():
        data = parse_train_log(train_log)
        payload = {
            "latest": data["latest"],
            "steps": data["steps"],
            "evals": data["evals"],
            "cycles": data["cycles"],
            "events": data["events"],
            "file_mtime": data["file_mtime"],
            "alive": _is_alive(data, selfplay_dir),
            "now": _iso(time.time()),
        }
        return jsonify(payload)

    @app.route("/api/games")
    def api_games():
        return jsonify({"games": list_games(selfplay_dir)})

    @app.route("/api/games/<gid>")
    def api_game(gid: str):
        path = _resolve_game(selfplay_dir, gid)
        if path is None:
            return jsonify({"error": "game not found"}), 404
        data = reconstruct_game(path)
        if data is None:
            return jsonify({"error": "game unreadable"}), 404
        return jsonify({"id": gid, **data})

    return app


def _resolve_game(selfplay_dir: Path, gid: str) -> "Path | None":
    """Map an id (file stem) to an npz path, guarding against traversal."""
    if not gid or any(ch in gid for ch in "/\\"):
        return None
    p = selfplay_dir / f"{gid}.npz"
    try:
        if p.is_file() and p.parent.resolve() == selfplay_dir.resolve():
            return p
    except OSError:
        return None
    return None


def _is_alive(train_data: dict, selfplay_dir: Path) -> bool:
    """Training liveness: any artifact written recently?

    ``train.jsonl`` is only touched while training steps run (self-play phases
    between cycles write npz files instead), so liveness is the most recent
    mtime among the train log and the newest self-play game.
    """
    newest = 0.0
    if train_data.get("file_mtime"):
        try:
            newest = max(newest, datetime.fromisoformat(train_data["file_mtime"]).timestamp())
        except (ValueError, TypeError):
            pass
    try:
        for p in selfplay_dir.glob("*.npz"):
            newest = max(newest, p.stat().st_mtime)
    except OSError:
        pass
    return (time.time() - newest) < STALE_AFTER_S if newest else False


# ---------------------------------------------------------------------------
# CLI entry point: python -m omigamax.web.dashboard [--port 8123]
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="omigamax training monitor (KataGui-style web dashboard)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--train-log", default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--selfplay-dir", default=DEFAULT_SELFPLAY_DIR)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    app = create_app(train_log=args.train_log, selfplay_dir=args.selfplay_dir)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
