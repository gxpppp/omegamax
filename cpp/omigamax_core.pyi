"""omigamax C++ core module (pybind11 extension type stub).

Bit-exact mirror of :mod:`omigamax.rules` — a C++ Board with the same
play/legality/ko/scoring semantics as the Python reference.
"""

def has_liberty(state: list[int], size: int, r: int, c: int) -> bool:
    """True if the group at (r, c) in a flat state has at least one liberty."""


class CppBoard:
    def __init__(self, size: int) -> None:
        """Board with the given parameterized size (e.g. 9, 19)."""

    @property
    def size(self) -> int: ...

    @property
    def num_moves(self) -> int: ...

    @property
    def pass_count(self) -> int: ...

    @property
    def last_captured_point(self) -> tuple[int, int] | None:
        """Point of a single-stone capture on the last move, or None."""

    def state(self) -> list[int]:
        """Copy of the flat board state (index ``r*size + c``)."""

    def get(self, r: int, c: int) -> int:
        """Color of the stone at (r, c) (0/1/2)."""

    def is_on_board(self, r: int, c: int) -> bool: ...

    def is_empty(self) -> bool: ...

    def group(self, r: int, c: int) -> list[tuple[int, int]]:
        """Coordinates of the connected group containing (r, c)."""

    def liberties(self, r: int, c: int) -> list[tuple[int, int]]:
        """Empty points adjacent to the group containing (r, c)."""

    def liberty_count(self, r: int, c: int) -> int: ...

    def has_liberty(self, r: int, c: int) -> bool: ...

    def is_terminal(self) -> bool:
        """True if the game is over: two consecutive passes."""

    def territory(self) -> dict[tuple[int, int], int | None]:
        """Map each empty point to its territory owner (None neutral)."""

    def score(self, komi: float = 7.5) -> tuple[float, float]:
        """Tromp-Taylor area score ``(black, white)`` with komi on white."""

    def winner(self, komi: float = 7.5) -> str | None:
        """``'B'``, ``'W'`` or ``None`` (jigo)."""

    def result_string(self, komi: float = 7.5) -> str:
        """SGF-style result, e.g. ``'B+3.5'``, ``'W+2'``, ``'Jigo'``."""

    def is_legal(self, r: int, c: int, color: int) -> bool:
        """True if (r, c) is a legal point move for ``color`` (pass is legal)."""

    def play(
        self, r: int, c: int, color: int, check_legal: bool = True
    ) -> int:
        """Place a stone (raises ValueError if illegal); returns captures."""

    def pass_move(self, color: int) -> int:
        """Play a pass; returns 0."""

    def legal_actions(self, color: int | None = None) -> list[int]:
        """Ascending legal action indices (points, then the pass index)."""

    def set_stone(self, r: int, c: int, color: int) -> None:
        """Directly place a stone, resetting move history (test helper)."""
