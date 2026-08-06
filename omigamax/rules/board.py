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

    # -- play ----------------------------------------------------------

    def is_legal(self, move, color):
        """True if ``move`` is legal for ``color``.

        Pass (``None``) is always legal. Ko/superko are not enforced yet
        (todo 4); bounds, occupancy and suicide prohibition are.
        """
        return is_legal_move(self._state, self.size, move, color)

    def play(self, move, color):
        """Place a stone (or pass) and remove captured opponent stones.

        Returns the number of stones captured (0 for a pass or a
        non-capturing move). Raises :class:`IllegalMoveError` if ``move`` is
        not legal; the board is left unchanged in that case.
        """
        if move is None:
            self.moves.append((None, color))
            self.pass_count += 1
            self.last_captured_point = None
            return 0
        if not self.is_legal(move, color):
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
