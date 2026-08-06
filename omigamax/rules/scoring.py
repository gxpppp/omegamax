"""Terminal detection and Tromp-Taylor area scoring (todo 4).

Terminal rule: the game ends when two consecutive passes are played
(``Board.pass_count >= 2``).

Scoring is the Tromp-Taylor "area" rule: every point on the board belongs to
the player whose stones occupy it plus the empty territory that player
surrounds. An empty point is *territory* for a color only when every stone
adjacent to its connected empty region is that color; regions bordered by
both colors (or by none, e.g. an empty board) are *neutral* and split 0.5 /
0.5 between the players.

There is no dead-stone removal under TT: stones standing on the board at the
end count for their owner, and a group surrounded but not captured merely
forces the empty points around it to neutral. Komi (default 7.5) is added to
white's total; black wins iff ``score_black > score_white``.
"""

from .liberties import BLACK, EMPTY, WHITE, neighbors


def is_terminal(board):
    """True if the game is over: two consecutive passes."""
    return board.pass_count >= 2


def territory(state, size):
    """Map each empty point to its territory owner.

    Returns ``{point: color}`` where ``color`` is ``BLACK``, ``WHITE`` or
    ``None`` for neutral points. Occupied points are absent from the map.
    """
    own = {}
    visited = set()
    for start in range(size * size):
        if start in visited or state[start] != EMPTY:
            continue
        region = []
        bordering = set()
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            region.append(idx)
            r, c = divmod(idx, size)
            for nr, nc in neighbors(r, c, size):
                nidx = nr * size + nc
                if state[nidx] == EMPTY:
                    stack.append(nidx)
                else:
                    bordering.add(state[nidx])
        if len(bordering) == 1:
            owner = bordering.pop()
        else:
            owner = None
        for idx in region:
            r, c = divmod(idx, size)
            own[(r, c)] = owner
    return own


def score(state, size, komi=7.5):
    """Tromp-Taylor area score ``(black_total, white_total)``.

    Komi is added to white's total. Neutral points are split equally.
    """
    black_stones = state.count(BLACK)
    white_stones = state.count(WHITE)
    neutral = 0
    black_territory = 0
    white_territory = 0
    for owner in territory(state, size).values():
        if owner is BLACK:
            black_territory += 1
        elif owner is WHITE:
            white_territory += 1
        else:
            neutral += 1
    half_neutral = neutral / 2.0
    black_total = black_stones + black_territory + half_neutral
    white_total = white_stones + white_territory + half_neutral + komi
    return black_total, white_total


def winner(state, size, komi=7.5):
    """Return ``'B'``, ``'W'`` or ``None`` (jigo)."""
    black, white = score(state, size, komi)
    if black > white:
        return "B"
    if white > black:
        return "W"
    return None


def result_string(state, size, komi=7.5):
    """SGF-style result, e.g. ``'B+3.5'``, ``'W+2'`` or ``'Jigo'``."""
    black, white = score(state, size, komi)
    if black > white:
        return f"B+{black - white:g}"
    if white > black:
        return f"W+{white - black:g}"
    return "Jigo"
