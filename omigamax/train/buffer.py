"""Replay buffer over self-play npz games (todo 14).

The buffer keeps a *window* of the most recent ``max_games``
(``config replay_buffer_games`` = 1000) games on disk -- one npz per game,
written by todo 13 -- and samples AGZ training batches from it.

Sampling (AGZ, Nature 550, 2017, Methods): pick ``batch_size`` games
uniformly from the window (with replacement), then a uniformly random
recorded position within each game. ``z`` is returned in ``(B, 1)`` shape for
the value head.

There is no leakage of future positions: each npz only contains recorded
positions ``0..T-1``, and pruned (older) games are removed from the window
entirely. The window itself is enforced both on load (only the newest
``max_games`` files are visible) and on disk (older files are deleted, the
same policy self-play already applies on generation).

Memory: npz arrays are cached in RAM behind an LRU cap (``cache_limit``
games, default = ``max_games``). The plan's budget is ~4000 games x ~6 MB =
~24 GB < 32 GB RAM, so the default 1000-game window is comfortably resident;
``cache_limit`` bounds the worst case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omigamax.train.selfplay import prune_old_games


def list_game_files(data_dir: "str | Path", keep: "int | None" = None) -> list[Path]:
    """npz game files in ``data_dir``, oldest first, capped to ``keep`` newest."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime_ns)
    if keep is not None:
        keep = max(1, int(keep))
        if len(files) > keep:
            files = files[-keep:]
    return files


def load_game(path: "str | Path") -> dict:
    """Load one game npz into plain numpy arrays ``s``/``pi``/``z``."""
    path = Path(path)
    with np.load(path) as data:
        return {
            "s": np.asarray(data["s"], dtype=np.float32),
            "pi": np.asarray(data["pi"], dtype=np.float32),
            "z": np.asarray(data["z"], dtype=np.float32),
        }


class ReplayBuffer:
    """Windowed replay buffer over ``data/selfplay`` npz games.

    Args:
        data_dir: directory holding one npz per game (todo 13 format).
        max_games: keep only the newest ``max_games`` games (window).
        cache_limit: max number of games held decompressed in RAM (LRU);
            ``None`` -> ``max_games``.
        board_size: expected board edge (informational; sampling never relies
            on it -- game arrays are sliced verbatim).
    """

    def __init__(
        self,
        data_dir: "str | Path",
        max_games: int = 1000,
        cache_limit: "int | None" = None,
        board_size: "int | None" = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_games = max(1, int(max_games))
        self.cache_limit = max(
            1, int(cache_limit if cache_limit is not None else self.max_games)
        )
        self.board_size = int(board_size) if board_size is not None else None
        self._files: list[Path] = []
        self._cache: dict[str, dict] = {}
        self._lru: list[str] = []
        self.refresh()

    # -- window management --------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the data dir: prune to the window and drop stale cache."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        prune_old_games(self.data_dir, self.max_games)
        self._files = list_game_files(self.data_dir, keep=self.max_games)
        keep = {str(f) for f in self._files}
        for key in list(self._cache):
            if key not in keep:
                del self._cache[key]
        self._lru = [k for k in self._lru if k in keep]

    @property
    def num_games(self) -> int:
        """Number of games currently in the window."""
        return len(self._files)

    @property
    def num_positions(self) -> int:
        """Total recorded positions across the window (moves sum)."""
        total = 0
        for f in self._files:
            with np.load(f) as data:
                total += int(np.asarray(data["z"]).shape[0])
        return total

    # -- caching ------------------------------------------------------------

    def _get_game(self, index: int) -> dict:
        """Decompressed arrays of game ``index`` in the window (LRU cache)."""
        path = str(self._files[int(index)])
        if path in self._cache:
            self._lru.remove(path)
            self._lru.append(path)
            return self._cache[path]
        rec = load_game(path)
        self._cache[path] = rec
        self._lru.append(path)
        if len(self._lru) > self.cache_limit:
            evicted = self._lru.pop(0)
            self._cache.pop(evicted, None)
        return rec

    # -- sampling -----------------------------------------------------------

    def sample(self, batch_size: int, rng: "np.random.Generator | None" = None) -> dict:
        """Sample a training batch of ``batch_size`` random positions.

        Uniform over the window games (with replacement) and uniform over the
        recorded positions inside each chosen game (AGZ scheme). Returns a
        dict with ``s`` ``(B, 17, N, N)``, ``pi`` ``(B, N*N+1)``, ``z``
        ``(B, 1)`` plus ``game_idxs``/``position_idxs`` (for tests/tracing).
        Raises ``RuntimeError`` when no non-empty position is available.
        """
        if rng is None:
            rng = np.random.default_rng()
        batch_size = int(batch_size)
        if not self._files:
            raise RuntimeError(
                f"replay buffer {self.data_dir} has no games"
            )
        n_games = len(self._files)

        s_parts: list[np.ndarray] = []
        pi_parts: list[np.ndarray] = []
        z_parts: list[np.ndarray] = []
        g_idx: list[int] = []
        p_idx: list[int] = []
        attempts = 0
        max_attempts = n_games * 8 + batch_size
        while len(s_parts) < batch_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"replay buffer {self.data_dir} contains only empty games"
                )
            gi = int(rng.integers(0, n_games))
            rec = self._get_game(gi)
            t = rec["s"].shape[0]
            if t == 0:
                continue  # resign-on-move-0 games hold no positions
            pos = int(rng.integers(0, t))
            s_parts.append(rec["s"][pos])
            pi_parts.append(rec["pi"][pos])
            z_parts.append(rec["z"][pos])
            g_idx.append(gi)
            p_idx.append(pos)

        return {
            "s": np.stack(s_parts).astype(np.float32, copy=False),
            "pi": np.stack(pi_parts).astype(np.float32, copy=False),
            "z": np.stack(z_parts)[:, None].astype(np.float32, copy=False),
            "game_idxs": np.asarray(g_idx, dtype=np.int64),
            "position_idxs": np.asarray(p_idx, dtype=np.int64),
        }
