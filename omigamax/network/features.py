"""17-plane feature encoding and policy index mapping for the network (todo 7).

Plane layout strictly follows the AGZ paper (Nature 550, 2017, Methods):

    st = [Xt, Yt, Xt-1, Yt-1, ..., Xt-7, Yt-7, C]

-- exactly 17 planes, with *no* constant-1 plane:

  * plane ``2t``   : current player's stones at the position ``t`` moves ago
                     (``t = 0`` is the most recent / current position)
  * plane ``2t+1`` : opponent's stones at the position ``t`` moves ago
  * plane 16       : colour to play -- all ``1.0`` if black to play, all
                     ``0.0`` if white to play

History shorter than 8 positions (start of game) is zero-filled.

Policy index mapping (a single ``r * board_size + c`` convention, consistent
with todo 6's model where the pass logit sits at index ``board_size**2``):

  * point ``(r, c)`` <-> index ``r * board_size + c``
  * pass            <-> index ``board_size ** 2``

All functions are pure numpy: no torch dependency in this module, so the
encoding can be used anywhere (tests, self-play data generation, MCTS).
"""

from __future__ import annotations

import math

import numpy as np

from omigamax.rules import BLACK, WHITE, Board

HISTORY_STEPS = 8
TOTAL_PLANES = 17
DEFAULT_BOARD_SIZE = 19


# ---------------------------------------------------------------------------
# action-index mapping helpers
# ---------------------------------------------------------------------------

def point_to_index(r: int, c: int, board_size: int) -> int:
    """Map a point ``(r, c)`` to its flat policy index ``r*size + c``."""
    if not (0 <= r < board_size and 0 <= c < board_size):
        raise ValueError(
            f"point ({r}, {c}) is off-board for board_size={board_size}"
        )
    return r * board_size + c


def index_to_point(index: int, board_size: int) -> tuple[int, int]:
    """Map a flat point index back to ``(row, col)``.

    The pass index (``board_size**2``) is *not* a point and raises.
    """
    n = board_size * board_size
    if not (0 <= index < n):
        raise ValueError(
            f"index {index} is not a point (valid range 0..{n - 1})"
        )
    return (index // board_size, index % board_size)


def pass_index(board_size: int) -> int:
    """Index of the pass move: ``board_size**2`` (after all points)."""
    return board_size * board_size


def is_pass(index: int, board_size: int) -> bool:
    """True if ``index`` is the pass move."""
    return index == pass_index(board_size)


# ---------------------------------------------------------------------------
# position snapshot normalisation
# ---------------------------------------------------------------------------

def _infer_board_size(snapshot) -> int:
    """Derive the board size from a single position snapshot."""
    if isinstance(snapshot, Board):
        return snapshot.size
    arr = np.asarray(snapshot)
    if arr.ndim == 2:
        r, c = arr.shape
        if r != c:
            raise ValueError(f"2-D snapshot must be square, got {arr.shape}")
        return int(r)
    if arr.ndim == 1:
        s = int(round(math.sqrt(arr.size)))
        if s * s != arr.size:
            raise ValueError(
                f"flat snapshot length {arr.size} is not a perfect square"
            )
        return s
    raise ValueError(
        f"snapshot must be a Board, a flat state list or a 2-D array, "
        f"got {type(snapshot).__name__}"
    )


def _snapshot_to_state(snapshot, board_size: int) -> np.ndarray:
    """Normalize one position snapshot to a 2-D ``(N, N)`` colour-code array."""
    if isinstance(snapshot, Board):
        arr = np.asarray(snapshot.state)
        size = snapshot.size
    else:
        arr = np.asarray(snapshot)
        if arr.ndim == 2:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(
                f"snapshot must be 1-D or 2-D, got {arr.ndim}-D"
            )
        size = int(round(math.sqrt(arr.size)))
    if size != board_size:
        raise ValueError(
            f"snapshot size {size} does not match board_size {board_size}"
        )
    return arr.astype(np.float32).reshape(board_size, board_size)


# ---------------------------------------------------------------------------
# 17-plane encoding
# ---------------------------------------------------------------------------

def encode(positions, current_color: int, board_size: int | None = None) -> np.ndarray:
    """Encode a position plus up to 8 moves of history into 17 planes.

    Args:
        positions: ordered most-recent first (``positions[0]`` is the current
            position, ``positions[1]`` one move ago, ...). Each snapshot is a
            :class:`Board`, a flat colour-code list (length ``N*N``, values
            0/1/2) or a 2-D ``(N, N)`` colour-code array.
        current_color: side to move, :data:`BLACK` (1) or :data:`WHITE` (2).
        board_size: board edge length; inferred from ``positions`` when
            ``None`` (defaults to 19 when ``positions`` is empty).

    Returns:
        ``(17, board_size, board_size)`` float32 array per the AGZ layout
        documented at the top of this module.
    """
    positions = list(positions)
    if board_size is None:
        board_size = (
            _infer_board_size(positions[0]) if positions else DEFAULT_BOARD_SIZE
        )
    if current_color not in (BLACK, WHITE):
        raise ValueError(
            f"current_color must be BLACK({BLACK}) or WHITE({WHITE}), "
            f"got {current_color!r}"
        )
    opponent = WHITE if current_color == BLACK else BLACK

    planes = np.zeros(
        (TOTAL_PLANES, board_size, board_size), dtype=np.float32
    )
    for t in range(HISTORY_STEPS):
        if t < len(positions):
            state = _snapshot_to_state(positions[t], board_size)
            planes[2 * t] = (state == current_color).astype(np.float32)
            planes[2 * t + 1] = (state == opponent).astype(np.float32)
    # plane 16: colour to play (all 1.0 for black, all 0.0 for white)
    planes[TOTAL_PLANES - 1] = 1.0 if current_color == BLACK else 0.0
    return planes


def encode_batch(
    positions_list,
    colors,
    board_size: int | None = None,
) -> np.ndarray:
    """Batch version of :func:`encode`.

    Args:
        positions_list: iterable of position-history lists (each as accepted
            by :func:`encode`).
        colors: iterable of side-to-move colours, one per position list.
        board_size: as in :func:`encode`.

    Returns:
        ``(B, 17, board_size, board_size)`` float32 array.
    """
    positions_list = list(positions_list)
    colors = list(colors)
    if len(positions_list) != len(colors):
        raise ValueError(
            f"{len(positions_list)} position lists but {len(colors)} colours"
        )
    if not positions_list:
        size = board_size if board_size is not None else DEFAULT_BOARD_SIZE
        return np.empty((0, TOTAL_PLANES, size, size), dtype=np.float32)
    return np.stack(
        [encode(p, c, board_size) for p, c in zip(positions_list, colors)]
    )


# ---------------------------------------------------------------------------
# decode_policy: logits -> legal-move probability distribution
# ---------------------------------------------------------------------------

def decode_policy(logits, board: Board, color: int | None = None) -> np.ndarray:
    """Turn raw policy logits into a distribution over *legal* moves.

    Args:
        logits: length ``board_size**2 + 1`` array (numpy or torch). Index
            ``r*size + c`` is the point ``(r, c)``; index ``size**2`` is pass.
        board: the position whose legal moves mask the distribution.
        color: side to move; when ``None`` it is derived from the move count
            (black opens, so even move count -> black to move).

    Returns:
        ``(board_size**2 + 1,)`` float32 probabilities over legal moves,
        summing to 1; illegal moves (and any non-finite logits) get exactly 0.
    """
    if hasattr(logits, "detach"):  # torch tensor
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    n = board.size
    n_points = n * n
    if logits.size != n_points + 1:
        raise ValueError(
            f"expected {n_points + 1} logits for board_size={n}, "
            f"got {logits.size}"
        )
    if color is None:
        # black opens the game -> even move count means black to play
        color = BLACK if len(board.moves) % 2 == 0 else WHITE

    scores = np.full(n_points + 1, -np.inf, dtype=np.float64)
    for r in range(n):
        for c in range(n):
            if board.is_legal((r, c), color):
                scores[point_to_index(r, c, n)] = logits[point_to_index(r, c, n)]
    # pass is always legal
    scores[pass_index(n)] = logits[pass_index(n)]

    if not np.isfinite(scores).any():
        # no legal move at all (defensive; in Go pass is always legal)
        probs = np.zeros(n_points + 1, dtype=np.float32)
        probs[pass_index(n)] = 1.0
        return probs

    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs[np.isinf(logits) | np.isnan(logits)] = 0.0
    probs /= probs.sum()
    return probs.astype(np.float32)
