"""Board state and the public Go playing API (todo 3).

Coordinates are ``(row, col)`` tuples, both 0-based, with ``(0, 0)`` the
top-left corner; a pass move is ``None``; colors are ``EMPTY`` (0),
``BLACK`` (1), ``WHITE`` (2).

Todo 4 (ko / two-pass terminal / scoring) and todo 5 (SGF) build on this
module -- the recorded history (``moves``, ``pass_count``,
``last_captured_point``) is the seam they slot into, so no breaking API
change is expected.
"""

from .captures import capture
from .ko import is_ko_prohibited
from .legality import IllegalMoveError, is_legal_move
from .liberties import (
    BLACK,
    EMPTY,
    WHITE,
    group,
    has_liberty,
    liberties,
    liberty_count,
)
from .scoring import (
    is_terminal,
    result_string,
    score,
    territory,
    winner,
)


class Board:
    """A Go board with parameterized size (default 19)."""

    def __init__(self, size=19):
        self.size = size
        self._state = [EMPTY] * (size * size)
        # -- history recorded for later todos (ko / terminal / SGF) --
        self.moves = []  # list of (move, color); move=None for a pass
        self.pass_count = 0  # consecutive passes since the last stone
        # Point (row, col) of a single-stone capture on the last move,
        # or None -- the data simple-ko detection (todo 4) needs.
        self.last_captured_point = None

    # -- accessors -----------------------------------------------------

    @property
    def state(self):
        """Copy of the flat board state (index ``r*size + c``)."""
        return list(self._state)

    def _idx(self, r, c):
        return r * self.size + c

    def is_on_board(self, r, c):
        """True if (r, c) lies within the board."""
        return 0 <= r < self.size and 0 <= c < self.size

    def get(self, r, c):
        """Color of the stone at (r, c) (``EMPTY`` if none)."""
        return self._state[self._idx(r, c)]

    def is_empty(self):
        """True if no stone has been placed yet."""
        return not any(self._state)

    # -- group / liberty queries (public API) --------------------------

    def group(self, r, c):
        """Set of coordinates of the connected group containing (r, c)."""
        return group(self._state, self.size, r, c)

    def liberties(self, r, c):
        """Set of empty points adjacent to the group containing (r, c)."""
        return liberties(self._state, self.size, r, c)

    def liberty_count(self, r, c):
        """Number of liberties of the group containing (r, c)."""
        return liberty_count(self._state, self.size, r, c)

    def has_liberty(self, r, c):
        """True if the group containing (r, c) has at least one liberty."""
        return has_liberty(self._state, self.size, r, c)

    # -- terminal / scoring (todo 4) ------------------------------------

    def is_terminal(self):
        """True if the game is over: two consecutive passes."""
        return is_terminal(self)

    def territory(self):
        """Map each empty point to its territory owner (see ``scoring``)."""
        return territory(self._state, self.size)

    def score(self, komi=7.5):
        """Tromp-Taylor area score ``(black, white)`` with komi on white."""
        return score(self._state, self.size, komi)

    def winner(self, komi=7.5):
        """``'B'``, ``'W'`` or ``None`` (jigo)."""
        return winner(self._state, self.size, komi)

    def result_string(self, komi=7.5):
        """SGF-style result, e.g. ``'B+3.5'`` or ``'W+2'``."""
        return result_string(self._state, self.size, komi)

    # -- play ----------------------------------------------------------

    def is_legal(self, move, color):
        """True if ``move`` is legal for ``color``.

        Pass (``None``) is always legal. Enforces bounds, occupancy, suicide
        prohibition (todo 3) and simple ko (todo 4): a move that would
        immediately retake the single stone captured on the opponent's last
        move is illegal. Superko is deliberately not enforced (plan's locked
        decision for todo 4).
        """
        if not is_legal_move(self._state, self.size, move, color):
            return False
        return not is_ko_prohibited(self.last_captured_point, move)

    def play(self, move, color, check_legal: bool = True):
        """Place a stone (or pass) and remove captured opponent stones.

        Returns the number of stones captured (0 for a pass or a
        non-capturing move). Raises :class:`IllegalMoveError` if ``move`` is
        not legal; the board is left unchanged in that case.

        ``check_legal=True`` (default) verifies legality first. Pass
        ``check_legal=False`` to skip the re-check when the caller already
        knows the move is legal (MCTS expansion builds children from
        ``node.legal_moves``, which were vetted by ``legal_actions`` -- the
        re-check doubles the per-child legality work on the hot path, P11
        self-play speedup). The caller guarantees legality in that case.
        """
        if move is None:
            self.moves.append((None, color))
            self.pass_count += 1
            self.last_captured_point = None
            return 0
        if check_legal and not self.is_legal(move, color):
            raise IllegalMoveError(
                f"illegal move {move} for color {color}"
            )
        r, c = move
        self.pass_count = 0
        self._state[self._idx(r, c)] = color
        removed, groups = capture(self._state, self.size, r, c, color)
        if removed == 1 and groups:
            # single-stone capture -> record its point for simple-ko (todo 4)
            self.last_captured_point = next(iter(groups[0]))
        else:
            self.last_captured_point = None
        self.moves.append((move, color))
        return removed

    def pass_move(self, color):
        """Play a pass move for ``color``."""
        return self.play(None, color)
