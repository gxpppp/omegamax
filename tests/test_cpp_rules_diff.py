"""Differential test: C++ rules core (omigamax_core.CppBoard) vs Python reference.

Plays 1000 fixed-seed random games (seeds 0-999; 9x9 and 19x19) stepping
:class:`omigamax.rules.board.Board` and :class:`omigamax_core.CppBoard` in
lockstep, asserting state / legal_actions / capture count / score / winner /
result_string are EXACTLY equal at every step (float scoring compares with
``==`` because Tromp-Taylor uses integer arithmetic — 0.5 increments — so the
results must be bit-exact). Plus scripted boundary cases: empty board, full
board, single-eye/false-eye, ko loop repetition, suicide prohibition, two-pass
endgame, komi half-point edge, and capture races.

Python is the reference: any divergence is a bug in the C++ side.
"""

import random

import pytest

import omigamax_core
from omigamax.mcts.mcts import legal_actions as py_legal_actions
from omigamax.rules import BLACK, WHITE
from omigamax.rules.board import Board

SIZE2 = 9 * 9  # pass index on a 9x9 board
PASS9 = 81
PASS19 = 19 * 19

# Game lengths are capped well above any double-pass outcome observed with the
# pass probabilities below.
MAX_MOVES = 400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _play_both(pb, cb, mv, color):
    """Play mv on both boards; assert equal captures and equal state."""
    if mv is None:
        removed_py = pb.pass_move(color)
        removed_cpp = cb.pass_move(color)
    else:
        r, c = mv
        removed_py = pb.play((r, c), color)
        removed_cpp = cb.play(r, c, color)
    assert removed_py == removed_cpp, "capture count divergence"
    assert list(pb.state) == cb.state(), "state divergence"
    return removed_py


def _assert_eq(pb, cb, label):
    assert list(pb.state) == cb.state(), f"[{label}] state mismatch"
    assert pb.pass_count == cb.pass_count, f"[{label}] pass_count mismatch"
    assert pb.is_terminal() == cb.is_terminal(), f"[{label}] terminal mismatch"
    assert pb.score() == cb.score(7.5), f"[{label}] score mismatch"
    assert pb.winner() == cb.winner(7.5), f"[{label}] winner mismatch"
    assert pb.result_string() == cb.result_string(7.5), (
        f"[{label}] result_string mismatch"
    )
    assert pb.territory() == cb.territory(), f"[{label}] territory mismatch"


def _assert_legal_parity(pb, cb, color, label):
    assert list(py_legal_actions(pb, color)) == cb.legal_actions(color), (
        f"[{label}] legal_actions mismatch"
    )


def _board_from_state(size, stones):
    """Both boards built from a color -> [(r, c)] map (move history reset)."""
    pb = Board(size)
    for color, points in stones.items():
        for r, c in points:
            pb._state[r * size + c] = color
    pb.pass_count = 0
    pb.last_captured_point = None
    cb = omigamax_core.CppBoard(size)
    for color, points in stones.items():
        for r, c in points:
            cb.set_stone(r, c, color)
    return pb, cb


def _play_game(size, seed, pass_prob, max_moves=MAX_MOVES):
    """One lockstep random game; returns (moves_played, step_count)."""
    rng = random.Random(seed)
    pb = Board(size)
    cb = omigamax_core.CppBoard(size)
    color = BLACK
    n = 0
    for step in range(max_moves):
        # --- parity checks at every step ---
        assert list(pb.state) == cb.state(), f"seed={seed} step={step} state"
        py_actions = py_legal_actions(pb, color)
        assert list(py_actions) == cb.legal_actions(color), (
            f"seed={seed} step={step} legal"
        )
        assert pb.score() == cb.score(7.5), f"seed={seed} step={step} score"
        assert pb.winner() == cb.winner(7.5), f"seed={seed} step={step} winner"
        assert pb.result_string() == cb.result_string(7.5), (
            f"seed={seed} step={step} result"
        )
        # spot-check group/liberty queries on a couple of random points
        for _ in range(2):
            pt = rng.randrange(size * size)
            r, c = divmod(pt, size)
            assert set(pb.group(r, c)) == set(cb.group(r, c)), (
                f"seed={seed} step={step} group"
            )
            assert set(pb.liberties(r, c)) == set(cb.liberties(r, c)), (
                f"seed={seed} step={step} liberties"
            )
            assert pb.liberty_count(r, c) == cb.liberty_count(r, c)
            assert pb.has_liberty(r, c) == cb.has_liberty(r, c)
            assert omigamax_core.has_liberty(
                list(pb.state), size, r, c
            ) == pb.has_liberty(r, c), f"seed={seed} step={step} free has_liberty"

        # --- choose a move (random legal point, or pass) ---
        point_actions = [a for a in py_actions if a < size * size]
        if rng.random() < pass_prob or not point_actions:
            mv = None
        else:
            mv = divmod(point_actions[rng.randrange(len(point_actions))], size)

        _play_both(pb, cb, mv, color)
        n += 1
        if pb.is_terminal():
            break
        color = WHITE if color == BLACK else BLACK
    # Games do not always double-pass within the move cap; lockstep parity is
    # still asserted at every step either way (termination is a separate,
    # scripted boundary case).
    _assert_eq(pb, cb, f"seed={seed}")
    return n, step


# ---------------------------------------------------------------------------
# random lockstep games: seeds 0-999 (9x9 bulk + 19x19 parameterization)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(800))
def test_random_game_9x9(seed):
    n, _ = _play_game(9, seed, pass_prob=0.08)
    assert 0 < n <= MAX_MOVES


@pytest.mark.parametrize("seed", range(800, 1000))
def test_random_game_19x19(seed):
    n, _ = _play_game(19, seed, pass_prob=0.05)
    assert 0 < n <= MAX_MOVES


def test_legal_actions_default_color_parity():
    """legal_actions() with no color derives from move-count parity."""
    pb = Board(9)
    cb = omigamax_core.CppBoard(9)
    moves = [((3, 3), BLACK), (None, WHITE), ((5, 5), BLACK), ((4, 4), WHITE)]
    for mv, color in moves:
        assert list(py_legal_actions(pb)) == cb.legal_actions()
        _play_both(pb, cb, mv, color)
    assert list(py_legal_actions(pb)) == cb.legal_actions()


# ---------------------------------------------------------------------------
# boundary cases
# ---------------------------------------------------------------------------

def test_empty_board():
    pb, cb = _mk = _board_from_state(9, {})
    assert pb.is_empty() and cb.is_empty()
    assert cb.state() == [0] * SIZE2
    _assert_legal_parity(pb, cb, BLACK, "empty")
    assert cb.legal_actions(BLACK) == list(range(0, PASS9 + 1))
    assert cb.legal_actions(WHITE) == cb.legal_actions(BLACK)
    _assert_eq(pb, cb, "empty")
    assert pb.score() == (40.5, 48.0)
    assert pb.winner() == cb.winner(7.5) == "W"
    assert pb.result_string() == cb.result_string(7.5) == "W+7.5"
    # territory maps every empty point; with no bordering stones all are
    # neutral (None) -- both implementations must agree exactly.
    assert len(pb.territory()) == SIZE2
    assert all(v is None for v in pb.territory().values())
    assert pb.territory() == cb.territory()
    assert not omigamax_core.has_liberty(list(pb.state), 9, 0, 0)


def test_full_board():
    pb, cb = _board_from_state(9, {BLACK: [(r, c) for r in range(9) for c in range(9)]})
    _assert_legal_parity(pb, cb, BLACK, "full")
    assert cb.legal_actions(BLACK) == [PASS9]  # no legal point; pass only
    _assert_eq(pb, cb, "full")
    assert pb.score() == (81.0, 7.5)
    assert pb.winner() == cb.winner(7.5) == "B"
    assert pb.result_string() == cb.result_string(7.5) == "B+73.5"
    assert pb.territory() == cb.territory() == {}


def test_single_eye_vs_false_eye():
    # A genuine single eye: (4,4) surrounded on all 4 sides by black. Every
    # empty point (including the eye and the outer ring) borders only black,
    # so the whole board is black territory under Tromp-Taylor.
    ring = [(3, 4), (4, 3), (4, 5), (5, 4)]
    pb, cb = _board_from_state(9, {BLACK: ring})
    _assert_eq(pb, cb, "real-eye")
    _assert_legal_parity(pb, cb, WHITE, "real-eye")
    assert pb.score() == (81.0, 7.5)
    assert pb.territory()[(4, 4)] == BLACK == cb.territory()[(4, 4)]

    # A false eye: one side of the ring is white, so the enclosed region
    # borders both colors -> neutral (not territory for either side).
    pb2, cb2 = _board_from_state(
        9, {BLACK: [(3, 4), (4, 3), (5, 4)], WHITE: [(4, 5)]}
    )
    _assert_eq(pb2, cb2, "false-eye")
    _assert_legal_parity(pb2, cb2, BLACK, "false-eye")
    assert pb2.territory()[(4, 4)] is None
    assert cb2.territory()[(4, 4)] is None
    assert pb2.score() == (41.5, 47.0)
    assert pb2.winner() == cb2.winner(7.5) == "W"
    assert pb2.result_string() == cb2.result_string(7.5) == "W+5.5"


def test_ko_loop_repetition():
    pb = Board(9)
    cb = omigamax_core.CppBoard(9)
    # classic single-stone ko shape (mirrors tests/test_rules_todo4.py)
    for mv, color in [
        ((0, 1), BLACK),  # 1
        ((2, 0), WHITE),  # 2
        ((1, 0), BLACK),  # 3
        ((3, 1), WHITE),  # 4
        ((1, 2), BLACK),  # 5
        ((2, 2), WHITE),  # 6
        ((8, 8), BLACK),  # 7
        ((1, 1), WHITE),  # 8  the ko stone
    ]:
        _play_both(pb, cb, mv, color)
    removed = _play_both(pb, cb, (2, 1), BLACK)  # 9 captures the ko stone
    assert removed == 1
    assert pb.last_captured_point == (1, 1)
    assert cb.last_captured_point == (1, 1)
    # the immediate retake is simple-ko illegal on BOTH boards
    assert not pb.is_legal((1, 1), WHITE)
    assert not cb.is_legal(1, 1, WHITE)
    _assert_legal_parity(pb, cb, WHITE, "ko-retake")
    # an intervening move elsewhere resolves the ko
    _play_both(pb, cb, (7, 7), WHITE)
    _play_both(pb, cb, (6, 6), BLACK)
    assert pb.last_captured_point is None and cb.last_captured_point is None
    assert pb.is_legal((1, 1), WHITE) and cb.is_legal(1, 1, WHITE)
    removed = _play_both(pb, cb, (1, 1), WHITE)
    assert removed == 1
    assert cb.last_captured_point == (2, 1) == pb.last_captured_point
    # and the reciprocal ko now applies to black
    assert not pb.is_legal((2, 1), BLACK)
    assert not cb.is_legal(2, 1, BLACK)
    _assert_eq(pb, cb, "ko-loop")


def test_suicide_forbidden():
    # (0,1) and (1,0) are two white groups that each keep a liberty, so black
    # (0,0) would be suicide -> illegal on both boards, and play raises.
    pb, cb = _board_from_state(9, {WHITE: [(0, 1), (1, 0)]})
    # positive control: an adjacent empty point is a legal white move
    assert pb.is_legal((0, 2), WHITE) and cb.is_legal(0, 2, WHITE)
    assert not pb.is_legal((0, 0), BLACK)
    assert not cb.is_legal(0, 0, BLACK)
    _assert_legal_parity(pb, cb, BLACK, "suicide")
    with pytest.raises(ValueError):
        pb.play((0, 0), BLACK)
    with pytest.raises(ValueError):
        cb.play(0, 0, BLACK)
    # the board is left unchanged by the rejected move
    assert cb.get(0, 0) == 0 and pb.get(0, 0) == 0
    _assert_eq(pb, cb, "suicide")


def test_double_pass_endgame():
    pb = Board(9)
    cb = omigamax_core.CppBoard(9)
    _play_both(pb, cb, (3, 3), BLACK)
    _play_both(pb, cb, None, WHITE)
    assert not pb.is_terminal() and not cb.is_terminal()
    _play_both(pb, cb, None, BLACK)
    assert pb.is_terminal() and cb.is_terminal()
    assert cb.pass_count == 2 == pb.pass_count
    _assert_eq(pb, cb, "double-pass")
    # playing a stone after two passes clears terminal (pass_count resets)
    _play_both(pb, cb, (4, 4), WHITE)
    assert not pb.is_terminal() and not cb.is_terminal()
    _assert_eq(pb, cb, "post-terminal")


def test_komi_edge_half_point():
    # 8 black stones on the top edge vs 1 white stone: black raw count is
    # exactly 7.0 above white's, so with komi 7.5 white wins by 0.5.
    pb, cb = _board_from_state(
        9,
        {BLACK: [(0, c) for c in range(8)], WHITE: [(8, 8)]},
    )
    _assert_eq(pb, cb, "komi-edge")
    assert pb.score() == (44.0, 44.5)
    assert pb.winner() == cb.winner(7.5) == "W"
    assert pb.result_string() == cb.result_string(7.5) == "W+0.5"
    # komi 7.0 would flip the margin to black by 0.5
    assert pb.score(7.0) == cb.score(7.0) == (44.0, 44.0)
    assert pb.winner(7.0) == cb.winner(7.0) is None
    assert pb.result_string(7.0) == cb.result_string(7.0) == "Jigo"


def test_capture_races():
    # (a) one move captures TWO separate white groups (5 stones total).
    pb, cb = _board_from_state(
        9,
        {
            WHITE: [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0)],
            BLACK: [(0, 4), (1, 1), (1, 2), (1, 3), (2, 1), (3, 0)],
        },
    )
    removed = _play_both(pb, cb, (0, 0), BLACK)
    assert removed == 5
    assert pb.last_captured_point is None and cb.last_captured_point is None
    for r, c in [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0)]:
        assert cb.get(r, c) == 0, f"captured stone still present at {(r, c)}"
    _assert_eq(pb, cb, "two-group-capture")

    # (b) a two-stone capture does NOT set a ko point, so the vacated point
    #     may be replayed immediately (simple ko, not superko).
    pb2, cb2 = Board(9), omigamax_core.CppBoard(9)
    for mv, color in [
        ((3, 3), BLACK), ((2, 3), WHITE), ((3, 4), BLACK), ((4, 3), WHITE),
        ((8, 8), BLACK), ((3, 2), WHITE), ((8, 7), BLACK), ((2, 4), WHITE),
        ((8, 6), BLACK), ((4, 4), WHITE), ((8, 5), BLACK),
    ]:
        _play_both(pb2, cb2, mv, color)
    removed = _play_both(pb2, cb2, (3, 5), WHITE)
    assert removed == 2
    assert pb2.last_captured_point is None and cb2.last_captured_point is None
    assert pb2.is_legal((3, 3), BLACK) and cb2.is_legal(3, 3, BLACK)
    assert _play_both(pb2, cb2, (3, 3), BLACK) == 0
    _assert_eq(pb2, cb2, "two-stone-capture")

    # (c) snapback: single-stone capture sets the ko point; the retake is
    #     blocked on both, and a pass clears it.
    pb3, cb3 = Board(9), omigamax_core.CppBoard(9)
    # white ko stone at (1,1); black captures it at (2,1) (ko shape above).
    for mv, color in [
        ((0, 1), BLACK), ((2, 0), WHITE), ((1, 0), BLACK), ((3, 1), WHITE),
        ((1, 2), BLACK), ((2, 2), WHITE), ((8, 8), BLACK), ((1, 1), WHITE),
    ]:
        _play_both(pb3, cb3, mv, color)
    removed = _play_both(pb3, cb3, (2, 1), BLACK)
    assert removed == 1 and cb3.last_captured_point == (1, 1)
    _play_both(pb3, cb3, None, WHITE)  # pass clears the ko record
    assert pb3.last_captured_point is None and cb3.last_captured_point is None
    _play_both(pb3, cb3, (6, 6), BLACK)
    assert _play_both(pb3, cb3, (1, 1), WHITE) == 1
    _assert_eq(pb3, cb3, "snapback")


def test_has_liberty_free_function():
    pb, cb = _board_from_state(9, {BLACK: [(0, 0), (0, 1)], WHITE: [(8, 8)]})
    s = list(pb.state)
    assert omigamax_core.has_liberty(s, 9, 0, 0) is True    # (0,2) empty
    assert omigamax_core.has_liberty(s, 9, 8, 8) is True    # open area
    assert omigamax_core.has_liberty(s, 9, 5, 5) is False   # empty point
    # a group fully surrounded by the opponent has no liberty (single stone,
    # multi-stone group, and empty point all agree); the white ring keeps a
    # liberty at (0,4), so its stones report True.
    pb2, cb2 = _board_from_state(
        9,
        {
            BLACK: [(1, 1), (1, 2)],
            WHITE: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (2, 1),
                    (2, 2), (2, 3), (1, 3)],
        },
    )
    s2 = list(pb2.state)
    assert omigamax_core.has_liberty(s2, 9, 1, 1) is False
    assert omigamax_core.has_liberty(s2, 9, 1, 2) is False
    assert omigamax_core.has_liberty(s2, 9, 0, 0) is True  # white has space
    assert omigamax_core.has_liberty(s2, 9, 4, 4) is False  # empty point
