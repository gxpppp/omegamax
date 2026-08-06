"""Capture detection and execution for Go (todo 3).

Works on the flat board state described in :mod:`omigamax.rules.liberties`.
A stone placed at (r, c) captures every adjacent opponent group that is left
with no liberties. Captured stones are removed from the state in place.
"""

from .liberties import EMPTY, group, has_liberty, neighbors, opponent


def captured_groups(state, size, r, c, color):
    """Return the list of opponent groups adjacent to (r, c) with no liberties.

    ``color`` is the color of the stone already present at (r, c). Each element
    of the returned list is a ``set`` of coordinates (one group each).
    """
    opp = opponent(color)
    captured = []
    seen = set()
    for nr, nc in neighbors(r, c, size):
        if (nr, nc) in seen:
            continue
        if state[nr * size + nc] != opp:
            continue
        if has_liberty(state, size, nr, nc):
            continue
        stones = group(state, size, nr, nc)
        captured.append(stones)
        seen.update(stones)
    return captured


def remove_group(state, size, stones):
    """Set every coordinate in ``stones`` to EMPTY; return the number removed."""
    removed = 0
    for r, c in stones:
        idx = r * size + c
        if state[idx] != EMPTY:
            state[idx] = EMPTY
            removed += 1
    return removed


def capture(state, size, r, c, color):
    """Capture adjacent liberty-less opponent groups after placing at (r, c).

    Returns ``(total_removed, groups)`` where ``groups`` is the list of
    captured group coordinate sets.
    """
    groups = captured_groups(state, size, r, c, color)
    total = 0
    for stones in groups:
        total += remove_group(state, size, stones)
    return total, groups
