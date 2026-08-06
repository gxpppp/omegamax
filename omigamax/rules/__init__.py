"""Go rules engine: board, legality, captures, ko, scoring, SGF (todos 3-5)."""

from .board import Board
from .ko import is_ko_prohibited
from .legality import IllegalMoveError
from .liberties import BLACK, EMPTY, WHITE
from .scoring import (
    is_terminal,
    result_string,
    score,
    territory,
    winner,
)
from .sgf import export_sgf, move_to_sgf, parse_sgf, point_to_sgf

__all__ = [
    "Board",
    "IllegalMoveError",
    "EMPTY",
    "BLACK",
    "WHITE",
    "is_ko_prohibited",
    "is_terminal",
    "score",
    "winner",
    "result_string",
    "territory",
    "export_sgf",
    "parse_sgf",
    "point_to_sgf",
    "move_to_sgf",
]
