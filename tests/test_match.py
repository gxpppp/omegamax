"""Match / play CLI tests (todo 20).

Covers the plan's todo-20 acceptance surface on ``omigamax.cli.match`` and
``omigamax.cli.play``:

* **win-rate math** -- ``aggregate_results`` is a pure function: known
  per-game outcomes (with an engine error) produce the exact win rate, draw
  and error counts and the standard ELO difference (``400*log10(p/(1-p))``).
* **colour alternation** -- with a deterministic always-pass opponent the
  driver's game loop is fully predictable: both sides pass immediately, so
  white always wins (komi 7.5). Over 2 games engine1 is black then white and
  must win exactly 1 game -> win rate 0.5. This verifies the alternation AND
  the aggregate math end-to-end from a known opponent behaviour.
* **komi applied** -- the same always-pass game on an empty board with komi
  7.5 is ``W+7.5``; with komi 0 it is jigo (``winner is None``).
* **vs random-legal completes legally** -- a tiny 9x9 network (sims=4) vs the
  in-process :class:`RandomEngine`: 2 full games, all moves legal by
  construction (``Board.play`` raises otherwise), per-game winners reported.
* **KataGo path (mocked subprocess)** -- ``GTPClient`` is monkeypatched with a
  fake recording the GTP command sequence; ``make_katago_engine`` is asserted
  to locate ``tools/katago``, launch with ``-model``/``-config``, send the
  todo-5 rule setup and the ``kata-set-param maxVisits`` cap, and a full
  ``run_match`` game against the fake drives the boardsize/komi/clear_board +
  play/genmove exchange.
* **engine error handling** -- an opponent whose genmove returns an illegal
  move produces an ``error`` record and is counted, not a crash.
* **play** -- a scripted human session (stdin) completes and reports a score;
  an illegal human move is rejected and re-prompted; the text board renders
  GTP-oriented coordinates with the last move marked.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import pytest
import torch

from omigamax.cli import match as match_mod
from omigamax.cli.match import (
    GTPEngineInProcess,
    RandomEngine,
    aggregate_results,
    make_katago_engine,
    play_match_game,
    run_match,
)
from omigamax.cli.play import play_session, render_board
from omigamax.gtp.gtp import GTPEngine
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------

class AlwaysPassEngine:
    """A ``command()`` engine that always passes; records every command."""

    name = "alwayspass"

    def __init__(self) -> None:
        self.commands: "list[str]" = []

    def command(self, cmd: str):
        self.commands.append(cmd)
        name = cmd.split()[0]
        if name == "genmove":
            return True, "pass"
        if name == "name":
            return True, "alwayspass"
        return True, ""

    def stderr_tail(self, n: int = 8):
        return []

    def close(self) -> None:
        pass


class BadEngine(AlwaysPassEngine):
    """Always returns the occupied ``A1`` vertex (legal only once)."""

    name = "bad"

    def command(self, cmd: str):
        self.commands.append(cmd)
        name = cmd.split()[0]
        if name == "genmove":
            return True, "A1"
        return True, ""


class FakeGTPClient:
    """Replacement for ``match_mod.GTPClient``: records the GTP exchange."""

    def __init__(self, argv, *, name, timeout=120.0, cwd=None) -> None:
        self.argv = list(argv)
        self.name = name
        self.commands: "list[str]" = []
        self.closed = False
        self._genmove = iter(["D4", "pass"])

    def command(self, cmd: str):
        self.commands.append(cmd)
        name = cmd.split()[0]
        if name == "genmove":
            try:
                return True, next(self._genmove)
            except StopIteration:
                return True, "pass"
        if name == "name":
            return True, "katago"
        return True, ""

    def stderr_tail(self, n: int = 8):
        return []

    def close(self) -> None:
        self.closed = True


def make_omigamax_9x9(sims: int = 4, seed: int = 0) -> GTPEngineInProcess:
    """An in-process omigamax over a tiny 9x9 random net (fast CPU tests)."""
    net = create_model(1, 8, 9)
    engine = GTPEngine(network=net, board_size=9, komi=7.5,
                       simulations=sims, device="cpu", seed=seed)
    return GTPEngineInProcess(engine, name="omigamax")


# ---------------------------------------------------------------------------
# win-rate / ELO aggregation (pure math)
# ---------------------------------------------------------------------------

def test_aggregate_winrate_math():
    records = [
        {"error": None, "winner": "B", "engine1_win": True},
        {"error": None, "winner": "W", "engine1_win": False},
        {"error": None, "winner": "B", "engine1_win": True},
        {"error": "genmove rejected by opp: 'x'", "winner": None,
         "engine1_win": False},
    ]
    s = aggregate_results(records, games=4)
    assert s["games"] == 4
    assert s["completed"] == 3
    assert s["errors"] == 1
    assert s["engine1_wins"] == 2
    assert s["draws"] == 0
    assert s["winrate"] == pytest.approx(2 / 3)
    # ELO(p) = 400*log10(p/(1-p)) = 400*log10(2) for p = 2/3
    assert s["elo_diff"] == pytest.approx(400 * math.log10(2), rel=1e-6)


def test_aggregate_degenerate_all_errors():
    s = aggregate_results(
        [{"error": "x", "winner": None, "engine1_win": False}], games=1
    )
    assert s["completed"] == 0
    assert s["errors"] == 1
    assert s["winrate"] == 0.0
    assert s["elo_diff"] == 0.0


def test_aggregate_uses_completed_only():
    records = [{"error": None, "winner": "B", "engine1_win": True}] * 2
    s = aggregate_results(records, games=2)
    assert s["winrate"] == 1.0
    assert s["elo_diff"] == pytest.approx(2400.0)  # clamped


# ---------------------------------------------------------------------------
# driver: colour alternation, komi, error handling
# ---------------------------------------------------------------------------

def test_colors_alternate_across_games():
    # always-pass on both sides: white always wins (komi 7.5), so engine1
    # wins exactly the odd game where it is white -> win rate 0.5.
    report = run_match(AlwaysPassEngine(), AlwaysPassEngine(), games=2,
                       size=9, komi=7.5, max_moves=10, seed=0, sgf_dir=None)
    assert [r["engine1_color"] for r in report["games_detail"]] == ["B", "W"]
    assert report["completed"] == 2
    assert report["errors"] == 0
    assert report["engine1_wins"] == 1
    assert report["winrate"] == 0.5
    assert report["elo_diff"] == 0.0


def test_komi_applied():
    e1, e2 = AlwaysPassEngine(), AlwaysPassEngine()
    rec = play_match_game(e1, e2, size=9, komi=7.5, max_moves=10, seed=0,
                          engine1_name="omigamax", engine1_is_black=True)
    assert rec["winner"] == "W"          # empty board, komi 7.5 on white
    assert rec["result"] == "W+7.5"
    assert rec["moves"] == 2             # B pass, W pass
    assert rec["error"] is None

    rec0 = play_match_game(e1, e2, size=9, komi=0.0, max_moves=10, seed=1,
                           engine1_name="omigamax", engine1_is_black=True)
    assert rec0["winner"] is None        # jigo on an empty board with komi 0


def test_illegal_engine_move_is_recorded_not_crash():
    # BadEngine returns the now-occupied A1 on its second genmove -> the
    # driver abandons the game with an error record and the match still
    # aggregates it.
    report = run_match(AlwaysPassEngine(), BadEngine(), games=1, size=9,
                       komi=7.5, max_moves=10, seed=0, sgf_dir=None)
    assert report["errors"] == 1
    assert report["completed"] == 0
    assert report["games_detail"][0]["error"]
    assert "illegal move" in report["games_detail"][0]["error"]


class ExplodingEngine(AlwaysPassEngine):
    """A ``command()`` engine that raises ``fail_times`` times on one command."""

    name = "exploder"

    def __init__(self, *, fail_on: str = "genmove", fail_times: int = 1) -> None:
        super().__init__()
        self.fail_on = fail_on
        self.failures_left = int(fail_times)

    def command(self, cmd: str):
        self.commands.append(cmd)
        name = cmd.split()[0]
        if name == self.fail_on and self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError(f"{self.name} exploded on {cmd!r}")
        if name == "genmove":
            return True, "pass"
        return True, ""


def test_engine_command_exception_becomes_per_game_error_not_abort():
    """F2 advisory: a raising ``command()`` yields a fail() record for THAT
    game; the match keeps playing the remaining games."""
    # game 0: engine1 (black) explodes on its first genmove -> per-game error;
    # game 1: engine1 (white) no longer explodes -> completes normally.
    e1 = ExplodingEngine(fail_on="genmove", fail_times=1)
    e2 = AlwaysPassEngine()
    report = run_match(e1, e2, games=2, size=9, komi=7.5, max_moves=10,
                       seed=0, sgf_dir=None)
    assert report["errors"] == 1
    assert report["completed"] == 1
    assert "command failed" in report["games_detail"][0]["error"]

    # a setup-phase exception (boardsize) is also a per-game fail() record
    rec = play_match_game(ExplodingEngine(fail_on="boardsize"),
                          AlwaysPassEngine(), size=9, komi=7.5, max_moves=10,
                          seed=3, engine1_name="omigamax",
                          engine1_is_black=True)
    assert rec["error"] and "setup command failed" in rec["error"]


# ---------------------------------------------------------------------------
# real vs-random integration (tiny net, cpu)
# ---------------------------------------------------------------------------

def test_match_vs_random_completes_legal():
    engine1 = make_omigamax_9x9(sims=4)
    engine2 = RandomEngine(size=9, komi=7.5, seed=1)
    report = run_match(engine1, engine2, games=2, size=9, komi=7.5,
                       max_moves=200, seed=0, sgf_dir=None)
    assert report["completed"] == 2
    assert report["errors"] == 0
    for rec in report["games_detail"]:
        assert rec["winner"] in ("B", "W")
        assert rec["error"] is None
        assert rec["engine1_color"] in ("B", "W")
        assert rec["moves"] > 0
    # the aggregate wins equal the per-game engine1 wins
    assert report["engine1_wins"] == sum(
        1 for r in report["games_detail"] if r["engine1_win"]
    )
    assert report["winrate"] == pytest.approx(
        report["engine1_wins"] / 2
    )


def test_random_engine_accepts_play_and_genmove():
    rng = RandomEngine(size=9, komi=7.5, seed=0)
    ok, resp = rng.command("boardsize 9")
    assert ok
    ok, resp = rng.command("komi 7.5")
    assert ok
    ok, resp = rng.command("clear_board")
    assert ok
    ok, resp = rng.command("play b D4")
    assert ok
    ok, resp = rng.command("genmove w")
    assert ok
    move = resp.strip()
    assert move == "pass" or move in [c + str(r) for r in range(1, 10)
                                      for c in "ABCDEFGHJ"]


# ---------------------------------------------------------------------------
# KataGo path (mocked subprocess)
# ---------------------------------------------------------------------------

def _katago_args(**overrides) -> argparse.Namespace:
    base = dict(katago_dir=str(PROJECT_ROOT / "tools" / "katago"),
                binary=None, weights=None, katago_config=None, timeout=10.0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_katago_binary_weights_located():
    binary, weights, config = match_mod._locate_katago(
        PROJECT_ROOT / "tools" / "katago"
    )
    assert binary.name == "katago.exe"
    assert weights.suffix in (".gz",)
    assert config.name == "default_gtp.cfg"


def test_katago_engine_setup_mocked(monkeypatch):
    monkeypatch.setattr(match_mod, "GTPClient", FakeGTPClient)
    client, info = make_katago_engine(_katago_args(), katago_visits=20)
    assert any("katago.exe" in a for a in client.argv)
    assert "-model" in client.argv
    assert "-config" in client.argv
    assert info["binary"].endswith("katago.exe")
    # todo-5 rule components + the per-move search cap
    assert "kata-set-rule ko SIMPLE" in client.commands
    assert "kata-set-rule scoring AREA" in client.commands
    assert "kata-set-rule tax NONE" in client.commands
    assert "kata-set-param maxVisits 20" in client.commands
    client.close()
    assert client.closed


def test_match_vs_katago_short_game_mocked(monkeypatch):
    # A full run_match game against a fake "katago" client: the driver must
    # exchange boardsize/komi/clear_board + play/genmove over the GTP
    # interface exactly as it would against the real subprocess.
    fake = FakeGTPClient([], name="katago")
    report = run_match(AlwaysPassEngine(), fake, games=1, size=9, komi=7.5,
                       max_moves=10, seed=0, sgf_dir=None)
    assert report["completed"] == 1
    assert report["errors"] == 0
    # the fake (white) receives the setup + the play/genmove exchange
    for expected in ("boardsize 9", "komi 7.5", "clear_board",
                     "play b pass", "genmove w"):
        assert any(c == expected or c.startswith(expected + " ")
                   for c in fake.commands), fake.commands


# ---------------------------------------------------------------------------
# play: scripted human sessions
# ---------------------------------------------------------------------------

def _play_session(script: str, **kwargs):
    out = io.StringIO()
    inp = io.StringIO(script)
    record = play_session(
        "models/best.pt", vs="omigamax", size=9, human_color="B",
        engine=AlwaysPassEngine(), stdin=inp, stdout=out, **kwargs,
    )
    return record, out.getvalue()


def test_play_scripted_session_completes():
    record, text = _play_session("D4\npass\npass\n")
    # board rendered with GTP columns; result reported
    assert "A B C D E F G H J" in text
    assert "Game over" in text
    assert record["winner"] in ("B", "W", None)
    assert record["result"]
    assert record["moves"] == 3  # B:D4, W:pass, B:pass -> terminal
    assert record["quit_requested"] is False


def test_play_illegal_human_move_rejected():
    record, text = _play_session("D4\nD4\npass\npass\n")
    assert "illegal move: D4" in text   # second D4 rejected + re-prompted
    assert record["moves"] == 3         # then pass -> terminal
    assert "Game over" in text


def test_play_quit_requested():
    record, text = _play_session("D4\nquit\n")
    assert record["quit_requested"] is True
    assert "Game over" not in text


def test_play_invalid_coordinate_reprompts():
    record, text = _play_session("Z9\npass\npass\n")
    assert "invalid coordinate" in text
    assert record["moves"] == 2


def test_render_board_marks_last_move():
    board = Board(9)
    board.play((5, 3), BLACK)   # D4 in GTP coords on 9x9
    board.play((2, 5), WHITE)
    text = render_board(board, last_move=(2, 5))
    assert "B" in text
    assert "w" in text                       # last move is lower-case
    assert "A B C D E F G H J" in text       # GTP columns skip I
