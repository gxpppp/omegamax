"""Simple-ko detection (todo 4).

Per the plan's locked decision this is **simple ko only** -- NOT positional
superko. A move is illegal if it immediately retakes the single stone that
the opponent's *just preceding* move captured: that retake recreates the
whole-board position from before the capturing move. The captured stone's
point is recorded on the Board as ``last_captured_point`` (the seam designed
in todo 3); ko detection therefore needs no global position-history table.

An extension point for full superko is intentionally left open (e.g. a
``rule`` parameter on :func:`Board.is_legal`) -- not implemented here.
"""

def is_ko_prohibited(last_captured_point, move):
    """Return True if ``move`` is a simple-ko retake.

    ``last_captured_point`` is the point ``(row, col)`` of the single stone
    captured on the opponent's immediately preceding move (``None`` if the
    last move was not such a capture). A pass (``move is None``) is never a
    ko retake.
    """
    if move is None or last_captured_point is None:
        return False
    # Compare the two coordinates directly -- ``tuple()`` copies would cost
    # 1.6M+ allocations on the MCTS expansion path (P11 self-play speedup).
    return (
        move[0] == last_captured_point[0]
        and move[1] == last_captured_point[1]
    )
