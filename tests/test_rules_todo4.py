"""TDD tests for the rules engine (todo 4): ko, pass/terminal, scoring, SGF.

Written BEFORE the implementation per the plan's TDD discipline (todo 4).
The plan's locked decisions are followed exactly:
  * ko: SIMPLE ko only (NOT superko) -- a move is illegal iff it retakes the
    single stone captured on the opponent's immediately preceding move
    (recorded on the Board as ``last_captured_point``).
  * terminal: two consecutive passes end the game.
  * scoring: Tromp-Taylor area scoring (stones + territory, neutral points
    split), komi 7.5 configurable, black wins iff score_black > score_white.
  * SGF: FF[4] export, coordinates A-T skipping I, komi + result annotated,
    plus a parser for the export/round-trip acceptance.
"""

import pytest

from omigamax.rules import Board, IllegalMoveError, BLACK, EMPTY, WHITE
from omigamax.rules.ko import is_ko_prohibited
from omigamax.rules.scoring import (
    is_terminal,
    result_string,
    score,
    territory,
    winner,
)
from omigamax.rules.sgf import (
    export_sgf,
    move_to_sgf,
    parse_sgf,
    point_to_sgf,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def board_from_state(size, stones):
    """Board with stones placed directly from a color -> [(r, c)] map."""
    b = Board(size)
    for color, points in stones.items():
        for r, c in points:
            b._state[r * size + c] = color
    return b


def play_ko_setup(b):
    """Play the classic single-stone ko shape on ``b`` (9x9).

    After this, black (2, 1) has captured the white ko stone at (1, 1) and
    ``b.last_captured_point == (1, 1)``; the immediate white retake at (1, 1)
    is the ko move under test.
    """
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
        b.play(mv, color)
    removed = b.play((2, 1), BLACK)  # 9  captures the ko stone
    assert removed == 1
    assert b.last_captured_point == (1, 1)


# ---------------------------------------------------------------------------
# ko (simple ko per the plan's locked decision; NOT superko)
# ---------------------------------------------------------------------------

def test_is_ko_prohibited_pure_function():
    # the ko predicate is a pure function of the recorded captured point
    assert is_ko_prohibited((1, 1), (1, 1)) is True
    assert is_ko_prohibited((1, 1), (2, 2)) is False
    assert is_ko_prohibited(None, (1, 1)) is False
    assert is_ko_prohibited((1, 1), None) is False


def test_simple_ko_recapture_immediately_illegal():
    b = Board(9)
    play_ko_setup(b)
    # classic ko shape: the immediate retake at (1, 1) would recreate the
    # position before the capturing move -- it must be rejected.
    assert not b.is_legal((1, 1), WHITE)
    # a move anywhere else remains legal
    assert b.is_legal((7, 7), WHITE)


def test_simple_ko_recapture_raises_illegal_move_error():
    b = Board(9)
    play_ko_setup(b)
    with pytest.raises(IllegalMoveError):
        b.play((1, 1), WHITE)
    # the board is left unchanged by the rejected move
    assert b.get(1, 1) == EMPTY
    assert b.get(2, 1) == BLACK
    assert len(b.moves) == 9


def test_ko_alternation_allowed_after_intervening_move():
    b = Board(9)
    play_ko_setup(b)
    # white plays a ko threat elsewhere, black answers elsewhere: the ko is
    # resolved, so white's retake at (1, 1) becomes legal (captures black
    # (2, 1)) -- and the reciprocal ko then applies to black at (2, 1).
    b.play((7, 7), WHITE)
    b.play((6, 6), BLACK)
    assert b.last_captured_point is None
    assert b.is_legal((1, 1), WHITE)
    removed = b.play((1, 1), WHITE)
    assert removed == 1
    assert b.get(2, 1) == EMPTY
    assert not b.is_legal((2, 1), BLACK)


def test_pass_clears_ko_state():
    b = Board(9)
    play_ko_setup(b)
    # white passes instead of retaking; the ko record is cleared
    b.play(None, WHITE)
    assert b.last_captured_point is None
    assert not b.is_terminal()  # only one pass so far
    # black plays a stone; white may now freely retake at (1, 1)
    b.play((6, 6), BLACK)
    removed = b.play((1, 1), WHITE)
    assert removed == 1
    assert b.get(2, 1) == EMPTY


def test_pass_never_sets_ko_state():
    b = Board(9)
    b.play((3, 3), BLACK)
    b.play(None, WHITE)
    assert b.last_captured_point is None
    b.play((4, 4), WHITE)
    assert b.last_captured_point is None  # no capture -> no ko point


def test_non_ko_capture_recapture_allowed():
    # a move that captures TWO stones does not record a ko point, so playing
    # into the vacated point afterwards is legal (simple ko, not superko).
    b = Board(9)
    for mv, color in [
        ((3, 3), BLACK),  # 1
        ((2, 3), WHITE),  # 2
        ((3, 4), BLACK),  # 3
        ((4, 3), WHITE),  # 4
        ((8, 8), BLACK),  # 5
        ((3, 2), WHITE),  # 6
        ((8, 7), BLACK),  # 7
        ((2, 4), WHITE),  # 8
        ((8, 6), BLACK),  # 9
        ((4, 4), WHITE),  # 10
        ((8, 5), BLACK),  # 11
    ]:
        b.play(mv, color)
    removed = b.play((3, 5), WHITE)  # 12  captures the two-stone black group
    assert removed == 2
    assert b.last_captured_point is None
    assert b.is_legal((3, 3), BLACK)
    assert b.play((3, 3), BLACK) == 0
    assert b.get(3, 3) == BLACK


# ---------------------------------------------------------------------------
# terminal: two consecutive passes
# ---------------------------------------------------------------------------

def test_two_consecutive_passes_end_game():
    b = Board(9)
    b.play((3, 3), BLACK)
    b.play(None, WHITE)
    b.play(None, BLACK)
    assert is_terminal(b)
    assert b.is_terminal()


def test_one_pass_does_not_end_game():
    b = Board(9)
    b.play((3, 3), BLACK)
    b.play(None, WHITE)
    assert not is_terminal(b)


def test_stone_between_passes_prevents_terminal():
    b = Board(9)
    b.play(None, BLACK)
    b.play((3, 3), WHITE)
    b.play(None, BLACK)
    assert b.pass_count == 1
    assert not is_terminal(b)


def test_terminal_cleared_by_stone_after_two_passes():
    b = Board(9)
    b.play(None, BLACK)
    b.play(None, WHITE)
    assert is_terminal(b)
    b.play((3, 3), BLACK)
    assert not is_terminal(b)


# ---------------------------------------------------------------------------
# Tromp-Taylor area scoring
# ---------------------------------------------------------------------------

def test_empty_board_neutral_split():
    # 25 empty points, no bordering stones -> all neutral, split 0.5/0.5
    b = Board(5)
    assert score(b._state, b.size, komi=0.0) == (12.5, 12.5)
    assert winner(b._state, b.size, komi=0.0) is None


def test_known_position_stone_plus_territory():
    # 5x5: black wall at column 2, white wall at column 3.
    #   black = 5 stones + 10 territory (cols 0-1) = 15
    #   white = 5 stones + 5 territory  (col 4)    = 10
    b = board_from_state(
        5,
        {BLACK: [(r, 2) for r in range(5)], WHITE: [(r, 3) for r in range(5)]},
    )
    black, white = score(b._state, b.size, komi=0.0)
    assert black == 15
    assert white == 10
    assert winner(b._state, b.size, komi=0.0) == "B"
    assert result_string(b._state, b.size, komi=0.0) == "B+5"
    assert b.score(komi=0.0) == (15, 10)


def test_territory_ownership_known_position():
    b = board_from_state(
        5,
        {BLACK: [(r, 2) for r in range(5)], WHITE: [(r, 3) for r in range(5)]},
    )
    own = territory(b._state, b.size)
    assert own[(0, 0)] == BLACK      # left region is black territory
    assert own[(4, 1)] == BLACK
    assert own[(0, 4)] == WHITE      # right region is white territory
    assert own[(4, 4)] == WHITE
    assert (2, 2) not in own         # occupied points are not territory
    assert (3, 3) not in own


def test_komi_7_5_applied_to_white_flips_winner():
    b = board_from_state(
        5,
        {BLACK: [(r, 2) for r in range(5)], WHITE: [(r, 3) for r in range(5)]},
    )
    # komi is added to white's score; black wins iff score_black > score_white
    assert b.score(komi=7.5) == (15.0, 17.5)
    assert winner(b._state, b.size, komi=7.5) == "W"
    assert result_string(b._state, b.size, komi=7.5) == "W+2.5"


def test_exact_komi_tie_is_jigo():
    b = board_from_state(
        5,
        {BLACK: [(r, 2) for r in range(5)], WHITE: [(r, 3) for r in range(5)]},
    )
    assert winner(b._state, b.size, komi=5.0) is None
    assert result_string(b._state, b.size, komi=5.0) == "Jigo"


def test_dead_stone_handling_per_tt():
    # A "dead" white stone sitting inside black's enclosed area. Under
    # Tromp-Taylor there is no dead-stone removal: the stone still counts for
    # white and the empty points around it become neutral (split), so black
    # may not claim them as territory.
    b = board_from_state(
        5,
        {
            BLACK: [
                (r, c) for r in range(5) for c in range(5)
                if r in (0, 4) or c in (0, 4)
            ],
            WHITE: [(2, 2)],
        },
    )
    own = territory(b._state, b.size)
    interior = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
    assert all(own[p] is None for p in interior)   # neutral, not black
    assert (0, 1) not in own                       # border stones occupied
    assert (2, 2) not in own                       # dead stone is occupied
    black, white = score(b._state, b.size, komi=0.0)
    assert black == 20  # 16 border stones + 8 neutral points split /2
    assert white == 5   # 1 dead stone + 8 neutral points split /2
    assert result_string(b._state, b.size, komi=7.5) == "B+7.5"


def test_scoring_handicap_free_state_only():
    # scoring is a pure function of the board state -- the same final position
    # reached through a different move order scores identically.
    via_state = board_from_state(
        5,
        {BLACK: [(r, 2) for r in range(5)], WHITE: [(r, 3) for r in range(5)]},
    )
    via_play = Board(5)
    for n in range(5):
        via_play.play((n, 2), BLACK)
        via_play.play((n, 3), WHITE)
    assert via_play._state == via_state._state
    assert score(via_play._state, 5, komi=0.0) == score(via_state._state, 5, komi=0.0) == (15, 10)


# ---------------------------------------------------------------------------
# SGF FF[4] export and parse round-trip
# ---------------------------------------------------------------------------

def test_sgf_export_header():
    b = Board(5)
    b.play((2, 2), BLACK)
    b.play((2, 3), WHITE)
    b.play(None, BLACK)
    b.play(None, WHITE)
    text = export_sgf(b)
    assert text.startswith("(")
    assert text.endswith(")")
    assert "GM[1]" in text
    assert "FF[4]" in text
    assert "CA[UTF-8]" in text
    assert "SZ[5]" in text
    assert "KM[7.5]" in text


def test_sgf_point_coordinates_skip_i():
    # SGF columns/rows are A-T skipping I: 0->a ... 7->h, 8->j, 9->k ... 18->s
    # SGF writes column first, then row.
    assert point_to_sgf(0, 0) == "aa"
    assert point_to_sgf(0, 8) == "ja"
    assert point_to_sgf(0, 9) == "ka"
    assert point_to_sgf(8, 8) == "jj"
    assert point_to_sgf(18, 18) == "tt"
    assert move_to_sgf((0, 0)) == "aa"
    assert move_to_sgf(None) == ""


def test_sgf_pass_moves_exported():
    b = Board(5)
    b.play((2, 2), BLACK)
    b.play(None, WHITE)
    b.play(None, BLACK)
    text = export_sgf(b)
    assert ";B[cc]" in text
    assert ";W[]" in text
    assert ";B[]" in text
    assert text.index(";W[]") < text.index(";B[]")


def test_sgf_result_annotation():
    b = Board(5)
    b.play((2, 2), BLACK)
    b.play((2, 3), WHITE)
    b.play(None, BLACK)
    b.play(None, WHITE)
    # 12.5 vs 12.5 + komi 7.5 -> white wins by 7.5
    text = export_sgf(b)
    assert "RE[W+7.5]" in text
    assert "RE[B+3.5]" not in text


def test_sgf_round_trip_parse():
    b = Board(5)
    for mv, color in [
        ((2, 2), BLACK),
        ((2, 3), WHITE),
        (None, BLACK),
        (None, WHITE),
    ]:
        b.play(mv, color)
    text = export_sgf(b)
    parsed = parse_sgf(text)
    assert parsed["size"] == 5
    assert parsed["komi"] == 7.5
    assert parsed["result"] == "W+7.5"
    assert parsed["moves"] == [
        (BLACK, (2, 2)),
        (WHITE, (2, 3)),
        (BLACK, None),
        (WHITE, None),
    ]


def test_parse_sgf_handwritten():
    sgf = (
        "(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RE[B+0.5]"
        ";B[dd];W[cc];B[];W[ee])"
    )
    parsed = parse_sgf(sgf)
    assert parsed["size"] == 9
    assert parsed["komi"] == 7.5
    assert parsed["result"] == "B+0.5"
    assert parsed["moves"] == [
        (BLACK, (3, 3)),
        (WHITE, (2, 2)),
        (BLACK, None),
        (WHITE, (4, 4)),
    ]
