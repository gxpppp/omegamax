"""8-fold dihedral symmetry augmentation for AGZ training (todo 14).

AlphaGo Zero trains each position under all 8 symmetries of the square board
(identity, 3 rotations, 4 reflections; config ``symmetry_aug=true``),
multiplying the effective dataset by 8 at no search cost -- the policy-value
network is equivariant under the symmetry group.

The *subtle* part is the policy-index mapping. The policy index of a point
``(r, c)`` is ``r * board_size + c`` (todo 7 convention); the pass logit
sits at index ``board_size**2``. Under a spatial symmetry ``T`` a stone at
``p`` moves to ``T(p)``, so the augmented policy must satisfy::

    pi_aug[idx(T(p))] = pi[idx(p)]          for every point p
    pi_aug[pass]       = pi[pass]

Equivalently ``pi_aug = pi[perm]`` with ``perm[T(p)] = p`` (flat indices).
The feature tensor ``s (17, N, N)`` is transformed in the same frame: a stone
at ``p`` appears at ``T(p)``, i.e. ``s_aug[:, T(p)] = s[:, p]``.

Both transformations share a single primitive -- the flat destination-index
map ``dest[src] = flat(T(src))`` (length ``N*N``) -- so they are consistent
*by construction*. ``apply_to_features`` scatters ``s`` through ``dest`` and
``apply_to_pi`` gathers ``pi`` through ``perm`` (the inverse relation).

The 8 transforms (on ``(row, col)``, 0-based, board edge ``N``, ``m = N-1``):

    0 identity          (r,   c)
    1 rotate 90°        (c,   m-r)
    2 rotate 180°       (m-r, m-c)
    3 rotate 270°       (m-c, r)
    4 flip columns      (r,   m-c)
    5 flip rows         (m-r, c)
    6 transpose         (c,   r)
    7 anti-diagonal     (m-c, m-r)

All functions are pure numpy (no torch dependency), matching
:mod:`omigamax.network.features`.
"""

from __future__ import annotations

import numpy as np

SYMMETRY_COUNT = 8

SYMMETRY_NAMES = [
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_columns",
    "flip_rows",
    "transpose",
    "anti_diagonal",
]


def transform_point(k: int, r: int, c: int, board_size: int) -> tuple[int, int]:
    """Image of point ``(r, c)`` under symmetry ``k`` (0..7)."""
    m = int(board_size) - 1
    if k == 0:
        return (r, c)
    if k == 1:
        return (c, m - r)
    if k == 2:
        return (m - r, m - c)
    if k == 3:
        return (m - c, r)
    if k == 4:
        return (r, m - c)
    if k == 5:
        return (m - r, c)
    if k == 6:
        return (c, r)
    if k == 7:
        return (m - c, m - r)
    raise ValueError(f"symmetry index must be in 0..{SYMMETRY_COUNT - 1}, got {k}")


def _flat_dest_map(k: int, board_size: int) -> np.ndarray:
    """``dest[src] = flat(T(src))`` for every point index ``src``.

    ``dest`` is a length ``N*N`` int64 array; because each ``T`` is a bijection
    of the board, ``dest`` is a permutation of ``0..N*N-1``.
    """
    n = int(board_size)
    r = np.arange(n, dtype=np.int64)
    c = np.arange(n, dtype=np.int64)
    rr, cc = np.meshgrid(r, c, indexing="ij")  # rr[i,j]=i, cc[i,j]=j
    m = n - 1
    if k == 0:
        dr, dc = rr, cc
    elif k == 1:
        dr, dc = cc, m - rr
    elif k == 2:
        dr, dc = m - rr, m - cc
    elif k == 3:
        dr, dc = m - cc, rr
    elif k == 4:
        dr, dc = rr, m - cc
    elif k == 5:
        dr, dc = m - rr, cc
    elif k == 6:
        dr, dc = cc, rr
    elif k == 7:
        dr, dc = m - cc, m - rr
    else:
        raise ValueError(f"symmetry index must be in 0..{SYMMETRY_COUNT - 1}, got {k}")
    return (dr * n + dc).reshape(-1)


def policy_permutation(k: int, board_size: int) -> np.ndarray:
    """Permutation ``perm`` of ``N*N+1`` with ``pi_aug = pi[perm]``.

    ``perm[dest] = src`` where ``dest`` is the flat image of ``src`` under
    symmetry ``k``; the pass index maps to itself.
    """
    n = int(board_size)
    flat = _flat_dest_map(k, n)  # dest[src]
    perm = np.arange(n * n + 1, dtype=np.int64)
    perm[flat] = np.arange(n * n, dtype=np.int64)
    return perm


def inverse_permutation(k: int, board_size: int) -> np.ndarray:
    """Inverse of :func:`policy_permutation` (``pi = pi_aug[inv]``)."""
    perm = policy_permutation(k, board_size)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=np.int64)
    return inv


def apply_to_features(s: np.ndarray, k: int) -> np.ndarray:
    """Transform a feature tensor ``(..., N, N)`` by symmetry ``k``.

    ``s_aug[:, T(p)] = s[:, p]``: a stone at ``p`` appears at ``T(p)`` in the
    augmented tensor (all planes -- stone planes transform; the constant
    colour plane is invariant under every symmetry by construction).
    """
    s = np.asarray(s, dtype=np.float32)
    n = s.shape[-1]
    if s.shape[-2] != n:
        raise ValueError(f"last two dims must be square, got {s.shape}")
    flat = _flat_dest_map(k, n)
    out = np.empty_like(s)
    s_flat = s.reshape(-1, n * n)
    out_flat = out.reshape(-1, n * n)
    out_flat[:, flat] = s_flat  # out[:, T(p)] = s[:, p]
    return out


def apply_to_pi(pi: np.ndarray, k: int, board_size: "int | None" = None) -> np.ndarray:
    """Permute a policy distribution by symmetry ``k``: ``pi_aug = pi[perm]``.

    Accepts a flat ``(N*N+1,)`` array or a batch ``(B, N*N+1)``. ``board_size``
    is derived from the last axis length when omitted.
    """
    pi = np.asarray(pi, dtype=np.float32)
    n_axis = pi.shape[-1]
    if board_size is None:
        n = int(round((n_axis - 1) ** 0.5))
        if n * n + 1 != n_axis:
            raise ValueError(
                f"last axis {n_axis} is not N*N+1 for an integer N"
            )
    else:
        n = int(board_size)
        if n * n + 1 != n_axis:
            raise ValueError(
                f"last axis {n_axis} does not match board_size={n} (N*N+1)"
            )
    perm = policy_permutation(k, n)
    return pi[..., perm]


def augment(s: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all 8 ``(s_k, pi_k)`` transformed pairs for one position."""
    return [
        (apply_to_features(s, k), apply_to_pi(pi, k))
        for k in range(SYMMETRY_COUNT)
    ]


def augment_batch(
    s: np.ndarray, pi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a batch ``(B, 17, N, N)``/``(B, N*N+1)`` to ``(8B, ...)``.

    Block ``k*B:(k+1)*B`` holds symmetry ``k`` applied to every sample. ``z``
    is untouched by symmetry (callers repeat it along the batch axis).
    """
    s = np.asarray(s, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)
    b = s.shape[0]
    if b != pi.shape[0]:
        raise ValueError(f"batch mismatch: s {s.shape[0]} vs pi {pi.shape[0]}")
    s_parts = [apply_to_features(s, k) for k in range(SYMMETRY_COUNT)]
    pi_parts = [apply_to_pi(pi, k) for k in range(SYMMETRY_COUNT)]
    return np.concatenate(s_parts, axis=0), np.concatenate(pi_parts, axis=0)
