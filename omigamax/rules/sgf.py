"""SGF FF[4] export and a parser for round-trip checks (todo 4).

Coordinates follow the SGF convention: columns and rows are the letters A-T
skipping I (a..h, j..s), column first. A pass is an empty property value
``[]``. Exported games are FF[4], include the board size (``SZ``), komi
(``KM``) and the result (``RE``), and carry an explicit UTF-8 charset.

The parser is intentionally small: it extracts the header properties and the
``;B[...]`` / ``;W[...]`` move sequence -- everything the round-trip
acceptance (export -> parse -> same moves/size/komi/result) requires.
"""

import re

from .liberties import BLACK, WHITE

# SGF column/row letters for indices 0..18 (A-T, skipping I).
COORDINATES = "abcdefghjklmnopqrst"

_COLOR_CHARS = {BLACK: "B", WHITE: "W"}


def point_to_sgf(r, c):
    """Encode a 0-based ``(row, col)`` as SGF coordinates (column first)."""
    return COORDINATES[c] + COORDINATES[r]


def move_to_sgf(move):
    """Encode a move; a pass (``None``) becomes an empty coordinate."""
    if move is None:
        return ""
    return point_to_sgf(*move)


def sgf_to_point(coord):
    """Decode SGF coordinates back to ``(row, col)``; empty -> ``None``.

    Raises :class:`ValueError` for malformed coordinates (wrong length or a
    letter outside ``a..t`` skipping ``i``) -- the GTP loadsgf path needs a
    clean error rather than a silently dropped move.
    """
    if not coord:
        return None
    if len(coord) != 2:
        raise ValueError(f"bad SGF coordinate: {coord!r}")
    try:
        return (COORDINATES.index(coord[1]), COORDINATES.index(coord[0]))
    except ValueError:
        raise ValueError(f"bad SGF coordinate: {coord!r}")


def export_sgf(board, komi=7.5, result=None, player_black="Black",
               player_white="White"):
    """Serialize ``board`` to an FF[4] SGF game record string.

    The result is computed from the current position with :mod:`.scoring`
    unless ``result`` (an SGF result string) is supplied.
    """
    from .scoring import result_string

    if result is None:
        result = result_string(board._state, board.size, komi)
    moves = "".join(
        f";{_COLOR_CHARS[color]}[{move_to_sgf(mv)}]"
        for mv, color in board.moves
    )
    return (
        f"(;GM[1]FF[4]CA[UTF-8]SZ[{board.size}]KM[{komi:g}]"
        f"PB[{player_black}]PW[{player_white}]RE[{result}]"
        f"{moves})"
    )


def parse_sgf(text):
    """Parse an SGF game record into ``{size, komi, result, moves}``.

    ``moves`` is a list of ``(color, move)`` tuples where ``move`` is
    ``(row, col)`` or ``None`` for a pass. Raises :class:`ValueError` with a
    clear message when the SZ/KM/RE headers are missing or a move coordinate is
    malformed, so callers can reject garbage input instead of crashing.
    """
    size_m = re.search(r"SZ\[(\d+)\]", text)
    if size_m is None:
        raise ValueError("missing SZ property")
    size = int(size_m.group(1))
    km_m = re.search(r"KM\[([^\]]*)\]", text)
    if km_m is None:
        raise ValueError("missing KM property")
    try:
        komi = float(km_m.group(1))
    except ValueError:
        raise ValueError(f"bad KM value: {km_m.group(1)!r}")
    re_m = re.search(r"RE\[([^\]]*)\]", text)
    if re_m is None:
        raise ValueError("missing RE property")
    result = re_m.group(1)
    moves = []
    for match in re.finditer(r";([BW])\[([^\]]*)\]", text):
        color = BLACK if match.group(1) == "B" else WHITE
        moves.append((color, sgf_to_point(match.group(2))))
    return {"size": size, "komi": komi, "result": result, "moves": moves}
