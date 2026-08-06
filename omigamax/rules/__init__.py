"""Go rules engine: board, legality, captures, ko, scoring, SGF (todos 3-5)."""

from .board import Board
from .legality import IllegalMoveError
from .liberties import BLACK, EMPTY, WHITE

__all__ = ["Board", "IllegalMoveError", "EMPTY", "BLACK", "WHITE"]
