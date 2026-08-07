"""GTP engine tests (todo 18).

Covers the GTP v2 protocol surface on :class:`omigamax.gtp.GTPEngine`: the
handshake commands, response framing with ids (``=id text`` / ``?id text``),
board/komi state, coordinate conversion (A-T skipping I, pass), legal/illegal
``play``, ``genmove`` legality, ``final_score``, the kgs-time_settings /
time_left stubs, handicap, loadsgf / printsgf, unknown-command errors and a
full subprocess pipe session (CRLF input tolerated, LF-only output).

The engine-under-test uses a tiny 9x9 network (blocks=1, channels=8) so the
suite runs on CPU in well under a second.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from omigamax.gtp.gtp import GTPCommandError, GTPEngine, parse_vertex, to_gtp
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Commands the plan locks for todo 18 (minus the handshake trio asserted via
# list_commands below).
REQUIRED_COMMANDS = [
    "protocol_version", "name", "version", "known_command", "list_commands",
    "boardsize", "clear_board", "komi", "play", "genmove",
    "fixed_handicap", "place_free_handicap", "set_free_handicap",
    "loadsgf", "kgs-time_settings", "time_left", "final_score", "quit",
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


# ---------------------------------------------------------------------------
# protocol handshake
# ---------------------------------------------------------------------------

def test_protocol_version():
    eng = make_engine()
    assert response(eng, "protocol_version") == "= 2"


def test_name_and_version():
    eng = make_engine()
    assert response(eng, "name") == "= omigamax"
    assert response(eng, "version") == "= 0.1.0"


def test_known_command():
    eng = make_engine()
    assert response(eng, "known_command genmove") == "= true"
    assert response(eng, "known_command nope") == "= false"
    assert response(eng, "known_command") == "? known_command requires one argument"


def test_list_commands():
    eng = make_engine()
    lines = eng.handle_line("list_commands")
    assert lines[0].startswith("= ")  # status frame header
    assert lines[-1] == ""  # blank-frame terminator
    body = "\n".join(lines[:-1])
    for cmd in REQUIRED_COMMANDS:
        assert cmd in body


def test_id_echo():
    eng = make_engine()
    assert response(eng, "7 protocol_version") == "=7 2"
    assert response(eng, "42 name") == "=42 omigamax"
    # an error keeps its id too
    assert response(eng, "9 play B D4 D4") == "?9 play requires a color and a vertex"


# ---------------------------------------------------------------------------
# state: boardsize / clear_board / komi
# ---------------------------------------------------------------------------

def test_boardsize_clears_board_and_resizes():
    eng = make_engine()
    eng.handle_line("play B D4")
    assert len(eng.board.moves) == 1
    assert response(eng, "boardsize 9") == "= "
    assert eng.size == 9
    assert eng.board.is_empty()
    # genmove still works after resize
    assert response(eng, "genmove B").startswith("= ")


def test_boardsize_rejects_invalid():
    eng = make_engine()
    for bad in ("1", "25", "abc"):
        assert response(eng, f"boardsize {bad}").startswith("? "), bad


def test_clear_board_resets_position():
    eng = make_engine()
    eng.handle_line("play B D4")
    eng.handle_line("play W E5")
    assert response(eng, "clear_board") == "= "
    assert eng.board.is_empty()
    assert len(eng.board.moves) == 0
    assert eng._handicap == 0


def test_komi():
    eng = make_engine()
    assert response(eng, "komi 5.5") == "= "
    assert eng.komi == 5.5
    assert response(eng, "komi xyz").startswith("? ")


# ---------------------------------------------------------------------------
# play: legal moves + pass, illegal moves rejected
# ---------------------------------------------------------------------------

def test_play_legal_and_pass():
    eng = make_engine()
    assert response(eng, "play B D4") == "= "
    assert eng.board.get(5, 3) == BLACK  # D4 on 9x9 -> (row 5, col 3)
    assert response(eng, "play W pass") == "= "
    assert eng.board.pass_count == 1


def test_play_illegal_occupied():
    eng = make_engine()
    eng.handle_line("play B D4")
    assert response(eng, "play B D4") == "? illegal move: B D4"
    # engine stays alive after the error
    assert response(eng, "name") == "= omigamax"


def test_play_suicide_rejected():
    eng = make_engine()
    # Surround A1 (8,0) with white stones, then black self-captures at A1.
    eng.handle_line("play W B1")  # (8,1)
    eng.handle_line("play W A2")  # (7,0)
    assert response(eng, "play B A1") == "? illegal move: B A1"


def test_play_out_of_turn_is_lenient():
    # The engine accepts legal moves regardless of turn order (KGS-friendly).
    eng = make_engine()
    assert response(eng, "play W D4") == "= "


def test_play_malformed():
    eng = make_engine()
    assert response(eng, "play B") == "? play requires a color and a vertex"
    assert response(eng, "play X D4").startswith("? ")
    assert response(eng, "play B Q9").startswith("? ")  # out of bounds on 9x9


# ---------------------------------------------------------------------------
# coordinate conversion
# ---------------------------------------------------------------------------

def test_coordinate_roundtrip():
    assert parse_vertex("A1", 19) == (18, 0)
    assert parse_vertex("T19", 19) == (0, 18)
    assert to_gtp((18, 0), 19) == "A1"
    assert to_gtp((0, 18), 19) == "T19"
    assert parse_vertex("pass", 19) is None
    assert to_gtp(None, 19) == "pass"
    # lower-case input accepted (GTP case tolerance)
    assert parse_vertex("h3", 9) == (6, 7)


def test_coordinate_skips_i():
    with pytest.raises(GTPCommandError):
        parse_vertex("I5", 19)
    # I is the only skipped letter: H and J both work
    assert parse_vertex("H19", 19) == (0, 7)
    assert parse_vertex("J19", 19) == (0, 8)


def test_coordinate_out_of_bounds():
    with pytest.raises(GTPCommandError):
        parse_vertex("T20", 19)
    with pytest.raises(GTPCommandError):
        parse_vertex("A0", 19)
    with pytest.raises(GTPCommandError):
        parse_vertex("Z4", 19)


# ---------------------------------------------------------------------------
# genmove
# ---------------------------------------------------------------------------

def test_genmove_black_replays_legal():
    eng = make_engine()
    assert response(eng, "clear_board") == "= "
    text = response(eng, "genmove B")
    assert text.startswith("= ")
    coord = text[2:]
    move = parse_vertex(coord, 9)
    if move is None:
        assert coord == "pass"
        assert eng.board.pass_count == 1
    else:
        r, c = move
        assert eng.board.get(r, c) == BLACK
    assert len(eng.board.moves) == 1


def test_genmove_white_replays_legal():
    eng = make_engine()
    eng.handle_line("play B D4")
    text = response(eng, "genmove W")
    assert text.startswith("= ")
    move = parse_vertex(text[2:], 9)
    if move is not None:
        assert eng.board.get(*move) == WHITE
    assert len(eng.board.moves) == 2


def test_genmove_pass_when_terminal():
    eng = make_engine()
    eng.handle_line("play B pass")
    eng.handle_line("play W pass")
    assert response(eng, "genmove B") == "= pass"


# ---------------------------------------------------------------------------
# final_score
# ---------------------------------------------------------------------------

def test_final_score_after_two_passes():
    eng = make_engine()
    eng.handle_line("play B pass")
    eng.handle_line("play W pass")
    assert response(eng, "final_score") == "= W+7.5"  # empty 9x9 + komi 7.5


def test_final_score_not_finished():
    eng = make_engine()
    eng.handle_line("play B D4")
    assert response(eng, "final_score") == "? game not finished"


# ---------------------------------------------------------------------------
# time control stubs
# ---------------------------------------------------------------------------

def test_kgs_time_settings_maps_to_budget():
    eng = make_engine()
    assert response(eng, "kgs-time_settings 600 60 5") == "= "
    assert eng._time_settings == {"main_time_s": 600, "byo_time_s": 60, "byo_stones": 5}
    # 600 s / 250 moves * 100 sims/s = 240 sims
    assert eng.simulations == 240


def test_kgs_time_settings_byo_branch():
    eng = make_engine()
    assert response(eng, "kgs-time_settings 0 30 5") == "= "
    # 30 s / 5 stones * 100 sims/s = 600 sims
    assert eng.simulations == 600


def test_kgs_time_settings_katago_clock_type_variant():
    # KataGo prefixes a clock type; the engine tolerates it (interop).
    eng = make_engine()
    assert response(eng, "kgs-time_settings byoyomi 600 60 5") == "= "
    assert eng.simulations == 240
    assert eng._time_settings["main_time_s"] == 600
    eng2 = make_engine()
    assert response(eng2, "kgs-time_settings absolute 300 0 0") == "= "
    assert eng2._time_settings["byo_time_s"] == 0


def test_time_left_stub():
    eng = make_engine()
    assert response(eng, "time_left B 120 0") == "= "
    assert eng._time_left[BLACK] == (120, 0)
    assert response(eng, "time_left W 60 5") == "= "
    assert eng._time_left[WHITE] == (60, 5)


# ---------------------------------------------------------------------------
# handicap
# ---------------------------------------------------------------------------

def test_fixed_handicap_and_white_moves_first():
    eng = make_engine()
    assert response(eng, "fixed_handicap 3") == "= C3,G7,G3"
    assert eng.board.state.count(BLACK) == 3
    assert eng._color_to_move() == WHITE  # white moves first after handicap
    assert response(eng, "play W pass") == "= "


def test_fixed_handicap_invalid():
    eng = make_engine()
    assert response(eng, "fixed_handicap 1").startswith("? ")
    assert response(eng, "fixed_handicap 10").startswith("? ")
    assert response(eng, "fixed_handicap abc").startswith("? ")


def test_place_free_handicap():
    eng = make_engine()
    assert response(eng, "place_free_handicap 2") == "= C3,G7"
    assert eng.board.state.count(BLACK) == 2
    assert eng._color_to_move() == WHITE


def test_set_free_handicap():
    eng = make_engine()
    assert response(eng, "set_free_handicap 2 C3 G7") == "= "
    assert eng.board.state.count(BLACK) == 2
    assert eng._color_to_move() == WHITE
    # wrong count / duplicate stone -> error, board unchanged by the failed call
    assert response(eng, "clear_board") == "= "
    assert response(eng, "set_free_handicap 2 C3").startswith("? ")
    assert response(eng, "set_free_handicap 2 C3 C3").startswith("? ")


# ---------------------------------------------------------------------------
# loadsgf / printsgf
# ---------------------------------------------------------------------------

def test_loadsgf(tmp_path):
    eng = make_engine()
    sgf_file = tmp_path / "game.sgf"
    sgf_file.write_text(
        "(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[5.5]PB[a]PW[b]RE[W+5.5];B[dd];W[ee])",
        encoding="utf-8",
    )
    assert response(eng, f"loadsgf {sgf_file}") == "= "
    assert eng.size == 9
    assert eng.komi == 5.5
    assert len(eng.board.moves) == 2
    assert response(eng, f"loadsgf {tmp_path / 'missing.sgf'}").startswith("? ")


def test_printsgf(tmp_path):
    eng = make_engine()
    eng.handle_line("play B D4")
    out = tmp_path / "out.sgf"
    assert response(eng, f"printsgf {out}").startswith("= ")
    from omigamax.rules.sgf import parse_sgf

    parsed = parse_sgf(out.read_text(encoding="utf-8"))
    assert parsed["size"] == 9
    assert len(parsed["moves"]) == 1


# ---------------------------------------------------------------------------
# quit / unknown commands / robustness
# ---------------------------------------------------------------------------

def test_quit():
    eng = make_engine()
    assert eng.handle_line("quit") == ["= ", ""]
    assert eng.should_quit is True


def test_unknown_command():
    eng = make_engine()
    assert response(eng, "bogus arg") == "? unknown command: bogus"
    assert response(eng, "name") == "= omigamax"  # still alive


def test_kgs_chat_silent():
    eng = make_engine()
    assert response(eng, "kgs-chat 0 hello") == "= "


def test_malformed_input_does_not_crash():
    eng = make_engine()
    assert eng.handle_line("") is None          # blank lines ignored
    assert eng.handle_line("   ") is None
    assert response(eng, "boardsize").startswith("? ")
    assert response(eng, "123").startswith("? ")  # bare id, no command
    assert response(eng, "name") == "= omigamax"


# ---------------------------------------------------------------------------
# full pipe session (subprocess): CRLF input tolerated, LF-only output
# ---------------------------------------------------------------------------

def test_subprocess_pipe_session(tmp_path):
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
    session = (
        "7 protocol_version\r\n"
        "name\r\n"
        "known_command genmove\r\n"
        "boardsize 9\r\n"
        "komi 5.5\r\n"
        "clear_board\r\n"
        "play B D4\r\n"
        "play B D4\r\n"
        "genmove W\r\n"
        "final_score\r\n"
        "quit\r\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "omigamax.cli.gtp_main",
         "--model", str(model_file), "--board-size", "9",
         "--simulations", "4", "--device", "cpu"],
        input=session, text=True, capture_output=True, timeout=180,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # LF-only output (no CRLF translation) -- Windows pipe safety
    assert "\r" not in out, f"CRLF leaked into stdout: {out!r}"
    lines = out.splitlines()
    assert "=7 2" in lines
    assert "= omigamax" in lines
    assert "= true" in lines
    assert "? illegal move: B D4" in lines
    assert "? game not finished" in lines
    # every non-blank line is a valid GTP frame header
    for ln in lines:
        assert ln == "" or ln.startswith(("=", "?")), repr(ln)
    # exactly one genmove frame with a legal coordinate or pass
    coord = None
    for ln in lines:
        m = re.fullmatch(r"= (pass|[A-HJ-T][1-9])", ln)
        if m:
            coord = m.group(1)
    assert coord is not None, f"no genmove response in output: {out!r}"
    board = Board(9)
    assert board.is_legal(parse_vertex(coord, 9), WHITE)
