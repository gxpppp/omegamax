"""GTP platform-reserved command set & robustness tests (todo 19).

Plan (todo 19): 命令健壮性表驱动测试 -- table-driven robustness coverage of the
platform-reserved command set that the online-platform integration (todo 20+)
will rely on:

* mixed case / extra whitespace tolerance (case-insensitive dispatch),
* unknown commands, boardsize re-set mid-game, handicap sequences,
* loadsgf followed by continued play, quit at any point mid-game,
* kgs-time_settings variants (main-time / byo-yomi / clock-type prefix),
* the fuzz/robustness layer: every malformed line (garbage, wrong arity, bad
  coords, absurd ids, empty/whitespace lines, binary junk, 10k-char lines)
  MUST produce a well-formed ``?`` response frame and the engine MUST stay
  alive -- every case is followed by a ``name`` liveness probe.

The acceptance gate (plan): ``pytest tests/test_gtp_robustness.py -v`` all green
with >= 20 cases (10+ malformed). This module ships 50+ cases including the
full malformed battery and a subprocess fuzz session over the real CLI.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from omigamax.gtp.gtp import GTPEngine
from omigamax.network.model import create_model
from omigamax.rules import BLACK, EMPTY, WHITE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every platform-reserved command that must answer ``known_command`` -> true.
PLATFORM_COMMANDS = [
    "protocol_version", "name", "version", "known_command", "list_commands",
    "boardsize", "clear_board", "komi", "play", "genmove",
    "fixed_handicap", "place_free_handicap", "set_free_handicap",
    "loadsgf", "kgs-time_settings", "time_left", "final_score",
    "printsgf", "undo", "kgs-chat", "quit",
]


def make_engine(**kwargs) -> GTPEngine:
    """A fresh engine over a tiny random 9x9 network (fast CPU tests)."""
    net = create_model(1, 8, 9)
    defaults = dict(
        network=net, board_size=9, komi=7.5, simulations=8,
        device="cpu", seed=0,
    )
    defaults.update(kwargs)
    return GTPEngine(**defaults)


def response(eng: GTPEngine, line: str) -> str:
    """The text of the first response line (the status frame header)."""
    lines = eng.handle_line(line)
    assert lines is not None and lines[-1] == ""
    return lines[0]


def assert_frame_ok(lines, line: str) -> None:
    """A valid GTP frame: status header + blank terminator, never a crash."""
    assert lines is not None, f"no frame for {line!r}"
    assert lines[-1] == "", f"missing blank terminator for {line!r}"
    assert lines[0].startswith(("=", "?")), f"bad status header: {lines[0]!r}"


def assert_alive(eng: GTPEngine) -> None:
    """Plan acceptance: the engine still answers ``name`` after every case."""
    assert response(eng, "name") == "= omigamax"


# ---------------------------------------------------------------------------
# 1. The malformed-input battery (plan: 10+ 畸形输入).
#    Every entry must yield a well-formed frame and leave the engine alive.
#    ``is_error`` marks entries expected to produce a ``?`` frame.
# ---------------------------------------------------------------------------

MALFORMED_BATTERY = [
    # garbage / unknown commands
    ("asdf;;;", True),
    ("#### garbage ####", True),
    ("!@#$%^&*()", True),
    ("; drop table", True),
    ("bogus command with args", True),
    ("123", True),                                   # bare id, no command
    # absurd command ids
    ("12345678901234567890 boardsize", True),        # huge id + arity error
    ("12345678901234567890 name", False),            # huge id + valid command
    # play: bad color / bad coords / wrong arity / trailing junk
    ("play x Q1", True),
    ("play b ZZ", True),
    ("play b z9", True),                             # lowercase bad column
    ("play b A0", True),                             # row 0
    ("play b T10", True),                            # out of bounds on 9x9
    ("play b I5", True),                             # I is skipped -> invalid
    ("play b 5", True),                              # missing column
    ("play", True),
    ("play B", True),
    ("play b D4 extra args", True),                  # trailing junk
    ("play b \x00d4", True),                         # null byte in vertex
    # boardsize
    ("boardsize", True),
    ("boardsize 0", True),
    ("boardsize 53", True),
    ("boardsize 1", True),
    ("boardsize -1", True),
    ("boardsize abc", True),
    ("boardsize 9 9", True),
    # komi
    ("komi", True),
    ("komi abc", True),
    ("komi nan", True),
    ("komi inf", True),
    # genmove
    ("genmove q", True),
    ("genmove", True),
    # handicap
    ("fixed_handicap 99", True),
    ("fixed_handicap 1", True),
    ("fixed_handicap abc", True),
    ("place_free_handicap 0", True),
    ("place_free_handicap 12", True),
    ("place_free_handicap xyz", True),
    ("set_free_handicap 2 C3", True),
    ("set_free_handicap 1 C3", True),
    ("set_free_handicap abc C3", True),
    # loadsgf / time control
    ("loadsgf", True),
    ("loadsgf /nonexistent/file.sgf", True),
    ("time_left x 5 0", True),
    ("time_left B 5", True),
    ("time_left B 5 0 extra", True),
    ("kgs-time_settings 600 60", True),
    ("kgs-time_settings abc 60 5", True),
    ("kgs-time_settings 600 60 5 extra", True),
    # known_command
    ("known_command", True),
    # binary-ish junk
    ("\x00\x01\x02binary", True),
    ("name\x00", True),
]


@pytest.mark.parametrize("line,is_error", MALFORMED_BATTERY)
def test_malformed_battery_keeps_engine_alive(line: str, is_error: bool):
    eng = make_engine()
    lines = eng.handle_line(line)
    assert_frame_ok(lines, line)
    if is_error:
        assert lines[0].startswith("?"), (line, lines[0])
    else:
        assert lines[0].startswith("="), (line, lines[0])
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 2. empty / whitespace / over-long lines
# ---------------------------------------------------------------------------

def test_empty_and_whitespace_lines_ignored():
    eng = make_engine()
    assert eng.handle_line("") is None
    assert eng.handle_line("\n") is None
    assert eng.handle_line("   \t  \r\n") is None
    assert_alive(eng)


def test_very_long_garbage_line_no_crash():
    eng = make_engine()
    lines = eng.handle_line("a" * 10_000)
    assert_frame_ok(lines, "a" * 10_000)
    assert lines[0].startswith("? ")
    assert len(lines[0]) <= 2005  # response frame is bounded
    assert_alive(eng)


def test_very_long_numeric_id_bounded():
    eng = make_engine()
    lines = eng.handle_line("9" * 10_000 + " name")
    assert_frame_ok(lines, "9" * 10_000)
    assert lines[0].startswith("=")
    assert len(lines[0]) <= 2005
    assert lines[0].endswith("omigamax")
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 3. case / whitespace tolerance (table-driven)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "PLAY b d4",
    "play b D4",
    "Name",
    "CLEAR_BOARD",
    "KOMI 5.5",
    "   play    b   d4   ",
    "\tboardsize\t13\t",
])
def test_case_and_whitespace_tolerance(line: str):
    eng = make_engine()
    assert response(eng, line).startswith("="), line
    assert_alive(eng)


def test_case_variant_changes_state():
    eng = make_engine()
    assert response(eng, "PLAY b d4") == "= "
    assert eng.board.get(5, 3) == BLACK
    assert response(eng, "KOMI 5.5") == "= "
    assert eng.komi == 5.5
    assert response(eng, "CLEAR_BOARD") == "= "
    assert eng.board.is_empty()


# ---------------------------------------------------------------------------
# 4. boardsize re-set mid-game
# ---------------------------------------------------------------------------

def test_boardsize_midgame_reset_then_continue():
    eng = make_engine()
    eng.handle_line("play B D4")
    eng.handle_line("play W E5")
    assert len(eng.board.moves) == 2
    assert response(eng, "boardsize 13") == "= "
    assert eng.size == 13
    assert eng.board.is_empty()
    assert response(eng, "play B K10") == "= "
    assert response(eng, "boardsize 9") == "= "
    assert response(eng, "play B D4") == "= "
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 5. handicap sequences
# ---------------------------------------------------------------------------

def test_handicap_sequence_replacement_mid_game():
    eng = make_engine()
    assert response(eng, "fixed_handicap 3") == "= C3,G7,G3"
    assert eng._color_to_move() == WHITE
    assert response(eng, "play W pass") == "= "
    # re-placing handicap mid-game resets the position and mover
    assert response(eng, "fixed_handicap 2") == "= C3,G7"
    assert eng.board.state.count(BLACK) == 2
    assert eng._color_to_move() == WHITE
    assert response(eng, "play W D4") == "= "
    assert_alive(eng)


def test_handicap_sequence_free_then_fixed():
    eng = make_engine()
    assert response(eng, "place_free_handicap 4") == "= C3,G7,G3,C7"
    assert response(eng, "play W pass") == "= "
    assert response(eng, "fixed_handicap 9") == "= C3,G7,G3,C7,E5,E7,E3,C5,G5"
    assert eng.board.state.count(BLACK) == 9
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 6. loadsgf then continue play; malformed SGF files
# ---------------------------------------------------------------------------

def test_loadsgf_then_continue_play(tmp_path):
    eng = make_engine()
    sgf = tmp_path / "g.sgf"
    sgf.write_text(
        "(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[5.5]RE[W+5.5];B[dd];W[ee])",
        encoding="utf-8",
    )
    assert response(eng, f"loadsgf {sgf}") == "= "
    assert eng.size == 9
    assert len(eng.board.moves) == 2
    # the game continues with black to move (GTP coords: D4 = row 5, col 3)
    assert response(eng, "play B D4") == "= "
    assert len(eng.board.moves) == 3
    assert_alive(eng)


def test_loadsgf_malformed_content_errors_cleanly(tmp_path):
    eng = make_engine()
    not_sgf = tmp_path / "not.sgf"
    not_sgf.write_text("this is not an sgf at all", encoding="utf-8")
    assert response(eng, f"loadsgf {not_sgf}").startswith("? ")
    assert_alive(eng)

    bad_koord = tmp_path / "badkoord.sgf"
    bad_koord.write_text(
        "(;GM[1]FF[4]SZ[9]KM[5.5]RE[W+5.5];B[zz];W[ee])", encoding="utf-8",
    )
    assert response(eng, f"loadsgf {bad_koord}").startswith("? ")
    assert_alive(eng)

    bad_komi = tmp_path / "badkomi.sgf"
    bad_komi.write_text(
        "(;GM[1]FF[4]SZ[9]KM[abc]RE[W+5.5];B[dd])", encoding="utf-8",
    )
    assert response(eng, f"loadsgf {bad_komi}").startswith("? ")
    assert_alive(eng)

    missing_sz = tmp_path / "nosz.sgf"
    missing_sz.write_text("(;GM[1]FF[4]KM[5.5]RE[W+5.5];B[dd])", encoding="utf-8")
    assert response(eng, f"loadsgf {missing_sz}").startswith("? ")
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 7. kgs-time_settings variants (plan: 分钟/秒/读秒形式)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,main_s,byo_s,byo_stones", [
    ("kgs-time_settings 600 60 5", 600, 60, 5),            # main + byo-yomi
    ("kgs-time_settings 300 0 0", 300, 0, 0),              # main time only
    ("kgs-time_settings 0 30 5", 0, 30, 5),                # byo-yomi only
    ("kgs-time_settings byoyomi 600 60 5", 600, 60, 5),    # clock-type prefix
    ("kgs-time_settings none 0 30 5", 0, 30, 5),
    ("kgs-time_settings absolute 300 0 0", 300, 0, 0),
    ("kgs-time_settings canadian 60 10 5", 60, 10, 5),
])
def test_kgs_time_settings_variants(line, main_s, byo_s, byo_stones):
    eng = make_engine()
    assert response(eng, line) == "= ", line
    assert eng._time_settings["main_time_s"] == main_s, line
    assert eng._time_settings["byo_time_s"] == byo_s, line
    assert eng._time_settings["byo_stones"] == byo_stones, line
    assert_alive(eng)


def test_kgs_time_settings_zero_keeps_budget():
    eng = make_engine()
    before = eng.simulations
    assert response(eng, "kgs-time_settings 0 0 0") == "= "
    assert eng.simulations == before  # no time -> current budget unchanged
    assert_alive(eng)


def test_kgs_time_settings_garbage_errors():
    eng = make_engine()
    for bad in (
        "kgs-time_settings 600 60",
        "kgs-time_settings abc 60 5",
        "kgs-time_settings 600 60 5 extra",
        "kgs-time_settings",
    ):
        assert response(eng, bad).startswith("? "), bad
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 8. quit at any point mid-game
# ---------------------------------------------------------------------------

def test_quit_anytime_mid_game():
    eng = make_engine()
    eng.handle_line("play B D4")
    eng.handle_line("play W E5")
    assert eng.handle_line("quit") == ["= ", ""]
    assert eng.should_quit is True
    # the engine object still answers after quit (the CLI loop exits)
    assert_alive(eng)


def test_quit_ignores_extra_args():
    eng = make_engine()
    assert eng.handle_line("quit now please") == ["= ", ""]
    assert eng.should_quit is True


def test_quit_between_commands_in_subprocess():
    # quit terminates the session; commands sent before it still answered.
    eng = make_engine()
    eng.handle_line("boardsize 9")
    eng.handle_line("quit")
    assert eng.should_quit


# ---------------------------------------------------------------------------
# 9. known_command for the full platform command set
# ---------------------------------------------------------------------------

def test_known_command_true_for_all_platform_commands():
    eng = make_engine()
    for cmd in PLATFORM_COMMANDS:
        assert response(eng, f"known_command {cmd}") == "= true", cmd
    assert response(eng, "known_command nope") == "= false"


def test_known_command_case_insensitive():
    eng = make_engine()
    assert response(eng, "known_command GENMOVE") == "= true"
    assert response(eng, "known_command Undo") == "= true"


def test_list_commands_includes_undo():
    eng = make_engine()
    lines = eng.handle_line("list_commands")
    body = "\n".join(lines[:-1])
    assert "undo" in body
    assert "kgs-chat" not in body  # KGS-specific, not standard GTP


# ---------------------------------------------------------------------------
# 10. final_score conventions
# ---------------------------------------------------------------------------

def test_final_score_format_and_komi_sensitivity():
    # One black stone with no white stones claims the whole 9x9 region under
    # Tromp-Taylor scoring (black area 81) -- the result string flips with komi.
    eng = make_engine(komi=0.0)
    eng.handle_line("play B D4")
    eng.handle_line("play B pass")
    eng.handle_line("play W pass")
    r = response(eng, "final_score")
    assert re.fullmatch(r"= (B|W)[+-]\d+(\.\d+)?", r), r
    assert r == "= B+81"

    eng2 = make_engine(komi=100.0)
    eng2.handle_line("play B D4")
    eng2.handle_line("play B pass")
    eng2.handle_line("play W pass")
    assert response(eng2, "final_score") == "= W+19"


def test_final_score_before_finish_errors_cleanly():
    eng = make_engine()
    eng.handle_line("play B D4")
    assert response(eng, "final_score").startswith("? ")
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 11. undo (replays the board to a previous state)
# ---------------------------------------------------------------------------

def test_undo_reverts_last_move():
    eng = make_engine()
    eng.handle_line("play B D4")
    eng.handle_line("play W E5")
    assert response(eng, "undo") == "= "
    assert len(eng.board.moves) == 1
    assert eng.board.get(5, 3) == BLACK    # D4 remains
    assert eng.board.get(4, 4) == EMPTY    # E5 removed
    assert response(eng, "undo") == "= "
    assert eng.board.is_empty()
    assert response(eng, "undo").startswith("? ")  # nothing left to undo
    assert_alive(eng)


def test_undo_n_moves():
    eng = make_engine()
    for color, mv in (("B", "D4"), ("W", "E5"), ("B", "F6")):
        eng.handle_line(f"play {color} {mv}")
    assert response(eng, "undo 2") == "= "
    assert len(eng.board.moves) == 1
    assert response(eng, "undo 5").startswith("? ")  # not enough moves
    assert_alive(eng)


def test_undo_invalid_count():
    eng = make_engine()
    eng.handle_line("play B D4")
    assert response(eng, "undo abc").startswith("? ")
    assert response(eng, "undo 0").startswith("? ")
    assert response(eng, "undo -1").startswith("? ")
    assert response(eng, "undo 1 2").startswith("? ")  # too many args
    assert_alive(eng)


def test_undo_after_handicap_is_handicap_aware():
    eng = make_engine()
    eng.handle_line("fixed_handicap 2")
    eng.handle_line("play W pass")
    assert response(eng, "undo") == "= "
    assert len(eng.board.moves) == 2        # both handicap stones remain
    assert eng._handicap == 2
    assert response(eng, "undo 2") == "= "  # re-undo past them -> cleared
    assert eng.board.is_empty()
    assert eng._handicap == 0
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 12. kgs-chat: silent response + chat-log hook
# ---------------------------------------------------------------------------

def test_kgs_chat_silent_and_logged():
    eng = make_engine()
    assert response(eng, "kgs-chat 12 hello world") == "= "
    assert response(eng, "kgs-chat 0 你好") == "= "
    assert response(eng, "kgs-chat") == "= "  # bare call tolerated
    assert eng.chat_log == [("12", "hello world"), ("0", "你好"), ("", "")]
    assert_alive(eng)


# ---------------------------------------------------------------------------
# 13. subprocess fuzz battery: the full malformed session through the real CLI
# ---------------------------------------------------------------------------

FUZZ_SESSION = "\n".join([
    "name",
    "protocol_version",
    "boardsize 9",
    "komi 7.5",
    "play b d4",
    "clear_board",
    "genmove B",
    "asdf;;;",
    "12345678901234567890 boardsize",
    "play x Q1",
    "play b ZZ",
    "play b D4 extra args",
    "boardsize 0",
    "boardsize 53",
    "boardsize abc",
    "komi abc",
    "komi nan",
    "genmove q",
    "fixed_handicap 99",
    "fixed_handicap 1",
    "fixed_handicap abc",
    "place_free_handicap xyz",
    "set_free_handicap 2 C3",
    "loadsgf /nonexistent/file.sgf",
    "time_left x 5 0",
    "",
    "   \t  ",
    "a" * 10_000,
    "9" * 10_000 + " name",
    "\x00\x01\x02binary",
    "PLAY b e5",
    "kgs-chat 0 hello world",
    "kgs-time_settings byoyomi 600 60 5",
    "kgs-time_settings 0 0 0",
    "kgs-time_settings 600 60 5 extra",
    "fixed_handicap 3",
    "undo",
    "undo 5",
    "name",
    "quit",
])


def test_fuzz_battery_subprocess_survives(tmp_path):
    """The real CLI loop survives every malformed input in one session."""
    net = create_model(1, 8, 9)
    model_file = tmp_path / "tiny.pt"
    torch.save(
        {
            "format": "omigamax-train-checkpoint",
            "version": 1,
            "global_step": 0,
            "arch": {"blocks": 1, "channels": 8, "board_size": 9},
            "model_state_dict": net.state_dict(),
        },
        model_file,
    )
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "omigamax.cli.gtp_main",
         "--model", str(model_file), "--board-size", "9",
         "--simulations", "4", "--device", "cpu"],
        input=FUZZ_SESSION, text=True, capture_output=True, timeout=180,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    lines = proc.stdout.splitlines()
    # every non-blank line is a well-formed GTP frame header
    for ln in lines:
        assert ln == "" or ln.startswith(("=", "?")), repr(ln)
    # the engine answered the handshake and survived to the final liveness probe
    assert "= omigamax" in lines
    assert "? unknown command: asdf;;;" in lines
    assert "? invalid color: 'x'" in lines
    assert "? invalid coordinate: 'ZZ'" in lines
    assert "? unacceptable board size: 0" in lines
    # Windows normalizes Path() to backslashes -- only match the message head
    assert any("? file not found" in ln and "file.sgf" in ln for ln in lines)
    assert "? invalid handicap: 99" in lines
    assert "? cannot undo: not enough moves" in lines
    assert lines.count("= omigamax") >= 2  # handshake + final liveness probe
