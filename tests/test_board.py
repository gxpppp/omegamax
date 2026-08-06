"""TDD tests for the rules engine (todo 3): board, liberties, legality, captures.

Written BEFORE the implementation per the plan's TDD discipline (todo 3).
Covers:
  * board representation (parameterized size, 0-based (row, col) coords)
  * move legality (out-of-bounds, occupancy, pass)
  * liberties (single stone center/edge/corner, groups, eye-like shapes)
  * captures (single stone, multi-stone group, multiple groups at once,
    edge, corner, no-capture when a liberty remains)
  * suicide prohibition (fill own eye / corner / atari) and the
    self-atari vs suicide distinction, incl. capture-overrides-suicide
"""

import pytest

from omigamax.rules import Board, IllegalMoveError, EMPTY, BLACK, WHITE
from omigamax.rules.liberties import (
    group,
    has_liberty,
    liberties,
    liberty_count,
    neighbors,
    opponent,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_state(size, stones):
    """Build a raw flat board state (list of size*size color codes).

    ``stones`` maps color -> list of (row, col) coordinates.
    """
    state = [EMPTY] * (size * size)
    for color, points in stones.items():
        for r, c in points:
            state[r * size + c] = color
    return state


# ---------------------------------------------------------------------------
# board representation
# ---------------------------------------------------------------------------

def test_default_board_is_19x19_and_empty():
    b = Board()
    assert b.size == 19
    assert b.is_empty()
    for r, c in [(0, 0), (9, 9), (18, 18)]:
        assert b.get(r, c) == EMPTY


def test_parameterized_board_size():
    for size in (2, 3, 5, 9, 13, 19):
        b = Board(size)
        assert b.size == size
        assert b.is_on_board(size - 1, size - 1)
        assert not b.is_on_board(size, 0)


def test_placing_stone_updates_state():
    b = Board(9)
    b.play((3, 3), BLACK)
    assert b.get(3, 3) == BLACK
    assert b.get(3, 4) == EMPTY
    assert not b.is_empty()


def test_coordinates_are_0_based():
    b = Board(9)
    b.play((2, 3), BLACK)
    assert b.get(2, 3) == BLACK
    assert b.get(3, 2) == EMPTY


# ---------------------------------------------------------------------------
# move legality: out-of-bounds, occupancy, pass
# ---------------------------------------------------------------------------

def test_out_of_bounds_moves_illegal():
    b = Board(9)
    for move in [(-1, 0), (0, -1), (9, 0), (0, 9), (9, 9), (-2, 3), (3, -2)]:
        assert not b.is_legal(move, BLACK)


def test_occupied_point_illegal_for_both_colors():
    b = Board(9)
    b.play((5, 5), BLACK)
    assert not b.is_legal((5, 5), WHITE)
    assert not b.is_legal((5, 5), BLACK)


def test_corner_and_edge_moves_legal():
    b = Board(9)
    for move in [(0, 0), (0, 8), (8, 0), (8, 8), (0, 4), (4, 0)]:
        assert b.is_legal(move, BLACK)


def test_pass_is_always_legal_and_recorded():
    b = Board(9)
    assert b.is_legal(None, BLACK)
    b.play(None, BLACK)
    assert b.pass_count == 1
    b.play(None, WHITE)
    assert b.pass_count == 2
    assert b.moves[0] == (None, BLACK)
    assert b.moves[1] == (None, WHITE)


def test_playing_a_stone_resets_pass_count():
    b = Board(9)
    b.play(None, BLACK)
    b.play(None, WHITE)
    b.play((0, 0), BLACK)
    assert b.pass_count == 0


def test_play_raises_illegal_move_error_and_leaves_board_unchanged():
    b = Board(9)
    b.play((3, 3), BLACK)
    with pytest.raises(IllegalMoveError):
        b.play((3, 3), WHITE)
    with pytest.raises(IllegalMoveError):
        b.play((99, 99), BLACK)
    assert b.get(3, 3) == BLACK
    assert len(b.moves) == 1
    assert b.pass_count == 0


# ---------------------------------------------------------------------------
# liberties: single stones, groups, eye-like shapes
# ---------------------------------------------------------------------------

def test_single_stone_center_four_liberties():
    state = make_state(9, {BLACK: [(4, 4)]})
    assert liberties(state, 9, 4, 4) == {(3, 4), (5, 4), (4, 3), (4, 5)}
    assert liberty_count(state, 9, 4, 4) == 4


def test_single_stone_edge_three_liberties():
    state = make_state(9, {BLACK: [(0, 4)]})
    assert liberty_count(state, 9, 0, 4) == 3


def test_single_stone_corner_two_liberties():
    state = make_state(9, {BLACK: [(0, 0)]})
    assert liberty_count(state, 9, 0, 0) == 2


def test_adjacent_same_color_form_one_group():
    state = make_state(9, {BLACK: [(4, 4), (4, 5)]})
    assert group(state, 9, 4, 4) == {(4, 4), (4, 5)}
    assert liberty_count(state, 9, 4, 4) == 6


def test_line_of_three_stones_eight_liberties():
    state = make_state(9, {BLACK: [(4, 3), (4, 4), (4, 5)]})
    assert group(state, 9, 4, 4) == {(4, 3), (4, 4), (4, 5)}
    assert liberty_count(state, 9, 4, 4) == 8


def test_adjacent_opposite_colors_not_connected():
    state = make_state(9, {BLACK: [(4, 4)], WHITE: [(4, 5)]})
    assert group(state, 9, 4, 4) == {(4, 4)}
    assert group(state, 9, 4, 5) == {(4, 5)}


def test_eye_like_ring_has_single_liberty():
    # Black fills the entire border of a 3x3 board; the connected ring's
    # only liberty is the empty center point -- a true single-point eye.
    ring = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    state = make_state(3, {BLACK: ring})
    assert group(state, 3, 0, 0) == set(ring)
    assert liberties(state, 3, 0, 0) == {(1, 1)}
    assert liberty_count(state, 3, 0, 0) == 1


def test_surrounded_stone_has_no_liberties():
    state = make_state(5, {BLACK: [(2, 2)], WHITE: [(1, 2), (3, 2), (2, 1), (2, 3)]})
    assert liberties(state, 5, 2, 2) == set()
    assert not has_liberty(state, 5, 2, 2)


def test_neighbors_helper_bounds():
    assert set(neighbors(2, 2, 5)) == {(1, 2), (3, 2), (2, 1), (2, 3)}
    assert set(neighbors(0, 0, 5)) == {(0, 1), (1, 0)}
    assert set(neighbors(0, 4, 5)) == {(0, 3), (1, 4)}


def test_opponent_helper():
    assert opponent(BLACK) == WHITE
    assert opponent(WHITE) == BLACK


def test_board_liberties_public_api():
    b = Board(9)
    b.play((4, 4), BLACK)
    assert b.liberty_count(4, 4) == 4
    b.play((4, 5), BLACK)
    assert b.liberty_count(4, 4) == 6
    b.play((4, 3), WHITE)
    assert b.liberty_count(4, 4) == 5
    assert b.has_liberty(4, 4)
    assert b.group(4, 4) == {(4, 4), (4, 5)}


# ---------------------------------------------------------------------------
# captures
# ---------------------------------------------------------------------------

def test_capture_single_stone():
    b = Board(9)
    b.play((3, 3), WHITE)
    b.play((2, 3), BLACK)
    b.play((4, 3), BLACK)
    b.play((3, 2), BLACK)
    assert b.get(3, 3) == WHITE  # one liberty left
    removed = b.play((3, 4), BLACK)  # last liberty -> capture
    assert removed == 1
    assert b.get(3, 3) == EMPTY
    for p in [(2, 3), (4, 3), (3, 2), (3, 4)]:
        assert b.get(*p) == BLACK


def test_capture_multi_stone_group():
    b = Board(9)
    b.play((3, 3), WHITE)
    b.play((3, 4), WHITE)
    b.play((2, 3), BLACK)
    b.play((2, 4), BLACK)
    b.play((4, 3), BLACK)
    b.play((4, 4), BLACK)
    b.play((3, 2), BLACK)
    removed = b.play((3, 5), BLACK)  # captures the two-stone group
    assert removed == 2
    assert b.get(3, 3) == EMPTY
    assert b.get(3, 4) == EMPTY
    for p in [(2, 3), (2, 4), (4, 3), (4, 4), (3, 2), (3, 5)]:
        assert b.get(*p) == BLACK


def test_capture_multiple_groups_simultaneously():
    b = Board(9)
    b.play((3, 3), WHITE)
    b.play((4, 4), WHITE)
    b.play((2, 3), BLACK)
    b.play((3, 2), BLACK)
    b.play((4, 3), BLACK)
    b.play((5, 4), BLACK)
    b.play((4, 5), BLACK)
    removed = b.play((3, 4), BLACK)  # last liberty of both white stones
    assert removed == 2
    assert b.get(3, 3) == EMPTY
    assert b.get(4, 4) == EMPTY


def test_capture_at_edge():
    b = Board(9)
    b.play((0, 1), WHITE)
    b.play((0, 0), BLACK)
    b.play((0, 2), BLACK)
    removed = b.play((1, 1), BLACK)
    assert removed == 1
    assert b.get(0, 1) == EMPTY


def test_capture_at_corner():
    b = Board(9)
    b.play((0, 0), WHITE)
    b.play((0, 1), BLACK)
    removed = b.play((1, 0), BLACK)
    assert removed == 1
    assert b.get(0, 0) == EMPTY


def test_no_capture_when_liberty_remains():
    b = Board(9)
    b.play((3, 3), WHITE)
    b.play((2, 3), BLACK)
    b.play((4, 3), BLACK)
    b.play((3, 2), BLACK)
    assert b.get(3, 3) == WHITE  # still one liberty at (3, 4)
    assert b.is_legal((3, 4), WHITE)
    b.play((3, 4), WHITE)  # white escapes along the edge
    assert b.liberty_count(3, 4) >= 2


# ---------------------------------------------------------------------------
# suicide prohibition and self-atari vs suicide
# ---------------------------------------------------------------------------

def test_self_atari_is_legal_not_suicide():
    b = Board(9)
    b.play((3, 3), BLACK)
    b.play((2, 3), WHITE)
    b.play((4, 3), WHITE)
    b.play((3, 2), WHITE)
    assert b.get(3, 3) == BLACK  # black is in atari (one liberty)
    assert b.is_legal((3, 4), BLACK)  # self-atari, but NOT suicide
    b.play((3, 4), BLACK)
    assert b.get(3, 4) == BLACK
    assert b.liberty_count(3, 3) == 3


def test_suicide_in_atari_illegal():
    # black two-stone group squeezed to exactly one liberty (3, 1)
    b = Board(5)
    b.play((1, 1), BLACK)
    b.play((2, 1), BLACK)
    b.play((0, 1), WHITE)
    b.play((1, 0), WHITE)
    b.play((1, 2), WHITE)
    b.play((2, 0), WHITE)
    b.play((2, 2), WHITE)
    b.play((4, 1), WHITE)
    b.play((3, 0), WHITE)
    b.play((3, 2), WHITE)
    assert b.liberty_count(1, 1) == 1
    assert not b.is_legal((3, 1), BLACK)  # filling last liberty = suicide


def test_fill_own_eye_is_suicide_illegal():
    # Black border ring on 3x3 has exactly one liberty: the eye at (1, 1).
    # Filling it joins a group with no liberties and captures nothing, so it
    # is a pure suicide and must be rejected.
    b = Board(3)
    for p in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]:
        b.play(p, BLACK)
    assert b.liberty_count(0, 0) == 1  # the eye at (1, 1)
    assert not b.is_legal((1, 1), BLACK)  # filling own eye = suicide
    # the opponent may play the eye and capture the whole black group
    assert b.is_legal((1, 1), WHITE)
    removed = b.play((1, 1), WHITE)
    assert removed == 8
    assert b.get(0, 0) == EMPTY
    assert b.get(2, 2) == EMPTY
    assert b.get(1, 1) == WHITE


def test_suicide_at_corner_illegal():
    b = Board(5)
    b.play((0, 0), BLACK)
    b.play((0, 1), WHITE)
    b.play((1, 1), WHITE)
    b.play((2, 0), WHITE)
    assert b.liberty_count(0, 0) == 1  # liberty at (1, 0)
    assert not b.is_legal((1, 0), BLACK)


def test_capture_overrides_suicide_is_legal():
    # a white ring in atari around the center; black captures it by playing
    # into what would otherwise be an own-eye (the ring vanishes, so black
    # gains liberties from the capture and the move is legal).
    b = Board(7)
    for r in range(2, 5):
        for c in range(2, 5):
            if (r, c) != (3, 3):
                b.play((r, c), WHITE)
    for r in range(1, 6):
        for c in range(1, 6):
            if 2 <= r <= 4 and 2 <= c <= 4:
                continue
            b.play((r, c), BLACK)
    assert b.liberty_count(2, 2) == 1  # white ring has one liberty (center)
    assert b.is_legal((3, 3), BLACK)
    removed = b.play((3, 3), BLACK)
    assert removed == 8
    assert b.get(3, 3) == BLACK
    for r in range(2, 5):
        for c in range(2, 5):
            if (r, c) != (3, 3):
                assert b.get(r, c) == EMPTY
