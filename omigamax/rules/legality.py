"""Move legality for Go (todo 3): bounds, occupancy, suicide prohibition.

Superko / simple-ko detection is deliberately NOT implemented here (todo 4);
:class:`~omigamax.rules.board.Board` records the history (``moves``,
``pass_count``, ``last_captured_point``) that todo 4 needs, so it can slot in
without an API change.
"""

from .captures import captured_groups
from .liberties import EMPTY, has_liberty


class IllegalMoveError(ValueError):
    """Raised by ``Board.play`` when a move is not legal under the rules."""


def is_on_board(move, size):
    """True if ``move = (row, col)`` lies within an ``size`` x ``size`` board."""
    r, c = move
    return 0 <= r < size and 0 <= c < size


def is_legal_move(state, size, move, color):
    """Return True if ``move`` may be played as ``color`` on ``state``.

    ``move`` is ``(row, col)``, or ``None`` for a pass (always legal).
    Checks, in order: pass, out-of-bounds, occupancy, and suicide prohibition
    -- a move that leaves its own resulting group with no liberties is illegal
    unless it captures at least one opponent stone. The placement used for the
    suicide check is temporary and always restored.
    """
    if move is None:
        return True
    r, c = move
    if not is_on_board(move, size):
        return False
    idx = r * size + c
    if state[idx] != EMPTY:
        return False
    # Simulate the placement (restored in ``finally``) to evaluate suicide.
    state[idx] = color
    try:
        if captured_groups(state, size, r, c, color):
            return True
        return has_liberty(state, size, r, c)
    finally:
        state[idx] = EMPTY
