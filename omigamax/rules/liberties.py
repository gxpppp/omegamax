"""Liberty and group computation for Go (todo 3).

A *group* is a maximal set of same-colored stones connected orthogonally.
The *liberties* of a group are the empty points adjacent to it.

All functions operate on a flat 1-D board state of length ``size*size`` with
index ``r*size + c`` -- the shared low-level representation used across the
rules package (board, captures, legality). This keeps the module pure and
free of any numpy/torch dependency.
"""

EMPTY = 0
BLACK = 1
WHITE = 2


def opponent(color):
    """Return the other stone color."""
    if color == BLACK:
        return WHITE
    if color == WHITE:
        return BLACK
    raise ValueError(f"not a stone color: {color!r}")


def neighbors(r, c, size):
    """Yield (row, col) for the on-board orthogonal neighbors of (r, c)."""
    if r > 0:
        yield (r - 1, c)
    if r < size - 1:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c < size - 1:
        yield (r, c + 1)


def group(state, size, r, c):
    """Return the set of coordinates of the connected group at (r, c).

    An empty point has an empty group (``set()``).
    """
    color = state[r * size + c]
    if color == EMPTY:
        return set()
    visited = set()
    stack = [(r, c)]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        cr, cc = cur
        for nr, nc in neighbors(cr, cc, size):
            if state[nr * size + nc] == color and (nr, nc) not in visited:
                stack.append((nr, nc))
    return visited


def liberties(state, size, r, c):
    """Return the set of empty points adjacent to the group at (r, c)."""
    result = set()
    for gr, gc in group(state, size, r, c):
        for nr, nc in neighbors(gr, gc, size):
            if state[nr * size + nc] == EMPTY:
                result.add((nr, nc))
    return result


def liberty_count(state, size, r, c):
    """Number of liberties of the group at (r, c)."""
    return len(liberties(state, size, r, c))


def has_liberty(state, size, r, c):
    """True if the group at (r, c) has at least one liberty.

    Early-exits at the first liberty found instead of materializing the full
    liberty set (P11 self-play speedup); returns exactly
    ``liberty_count(state, size, r, c) > 0`` for every position.
    """
    color = state[r * size + c]
    if color == EMPTY:
        return False
    visited = set()
    stack = [(r, c)]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        cr, cc = cur
        for nr, nc in neighbors(cr, cc, size):
            ni = nr * size + nc
            if state[ni] == EMPTY:
                return True
            if state[ni] == color and (nr, nc) not in visited:
                stack.append((nr, nc))
    return False
