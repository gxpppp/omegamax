"""Self-play data generator (todo 13).

Plays full games between two sides of the *same* policy-value network (AGZ:
one network, both colours), starting from the empty board, ending at two
consecutive passes (Tromp-Taylor terminal), and scores the final position
(komi 7.5 by default) to label every move with the game outcome.

Each move records the AGZ training tuple ``(s, pi, z)``:

* ``s`` -- the 17-plane feature tensor of the position (todo 7), encoded
  with up to ``HISTORY_STEPS`` (=8) prior *game* positions (most recent
  first), per the AGZ input layout;
* ``pi`` -- the MCTS *search policy* of that position: the temperature-
  softened visit-count distribution ``pi(a) propto N(root,a)^(1/tau)``
  from :func:`omigamax.mcts.temperature_policy` at the *same* ``tau`` the
  move was sampled with -- NOT a one-hot of the chosen move (the AGZ
  training target, Oracle review correction);
* ``z`` -- the game outcome from the mover's perspective: ``+1`` if the
  mover's colour won, ``-1`` otherwise (jigo, impossible with komi 7.5,
  gives ``0``).

Temperature schedule (AGZ, todo 10): ``tau = 1.0`` (sample proportional to
visit counts) for the first ``temperature_threshold`` (=30) moves, then
``tau -> 0`` (argmax). The schedule is caller-side -- this module picks the
``tau`` per move and passes it to ``sample_action``/``temperature_policy``.
Dirichlet root noise (config ``dirichlet_alpha``/``dirichlet_eps``) is
re-sampled at every move's fresh root (todo 10 machinery).

Resignation: ``resign_threshold`` defaults to ``0.0`` = *disabled* (the
plan's locked value; the AGZ 0.5%-value resign is a documented extension
behind the same config bit, only active when ``> 0``).

Inference discipline (plan, Oracle G3): the network is put in ``eval()``
mode before the game and every forward pass runs under ``torch.no_grad()``
(the :class:`~omigamax.mcts.BatchedNetworkEvaluator` already wraps its
forward in ``no_grad``; the generator makes ``eval()`` explicit so
batch-norm running statistics are never perturbed by self-play).

Data storage: one ``npz`` file per game in ``data/selfplay/``
(``np.savez_compressed``, arrays ``s`` / ``pi`` / ``z`` + metadata),
written atomically (tmp + ``os.replace``), pruned to the most recent
``replay_buffer_games`` (=1000) games. Cross-move search-tree reuse (AGZ
standard) is a documented extension, not implemented here (plan G4).

Throughput: ``generate_games`` and the CLI report ``sims/s`` (total search
simulations / single-process wall-clock seconds, network forward included),
``positions/s`` and ``games/hour``. The plan's soft gate is ``sims/s < 150``
-> WARN + optimisation notes (never a hard failure; exit code stays 0).

Usage::

    uv run python -m omigamax.train.selfplay --games 10 --simulations 40
    uv run python -m omigamax.train.selfplay --games 100 --data-dir data/selfplay
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPS,
    DEFAULT_KOMI,
    DEFAULT_LEAF_BATCH,
    DEFAULT_TEMPERATURE_THRESHOLD,
    DEFAULT_VIRTUAL_LOSS,
    MCTS,
    BatchedNetworkEvaluator,
    sample_action,
    temperature_policy,
)
from omigamax.network.features import HISTORY_STEPS, encode, pass_index
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board

# Game / data defaults (plan todo 13).
DEFAULT_MAX_MOVES = 1000          # >1000 moves force-terminates the game
DEFAULT_DATA_DIR = "data/selfplay"
DEFAULT_GAMES = 10
# Plan soft gate (Oracle F8): sims/s below this -> WARN, never a failure.
SOFT_SIMS_PER_SEC_WARN = 150.0
SOFT_GATE_SUGGESTION = (
    "generation is functional but slow; suggestions: raise config leaf_batch, "
    "enable fp16 autocast in the batched evaluator (config fp16), or use a "
    "smaller --simulations budget for the acceptance run."
)
# P12: cap on --selfplay-workers. Each worker process owns its own model copy
# + CUDA context (~1.4GB at fp16 on the 6GB RTX 3060), so 3 workers sum to
# ~4.2GB and stay under the 5GB safety ceiling; 4 would reach ~5.6GB -> OOM.
MAX_SELFPLAY_WORKERS = 3


# ---------------------------------------------------------------------------
# z: game outcome from the mover's perspective
# ---------------------------------------------------------------------------

def z_targets(move_count: int, winner: "str | None") -> np.ndarray:
    """Outcome ``z`` for every move from the mover's perspective.

    Move ``i`` (0-based) is played by black iff ``i`` is even (black opens).
    Returns ``+1`` when that mover won, ``-1`` when it lost, ``0`` for jigo
    (``winner is None`` -- impossible with komi 7.5).
    """
    n = int(move_count)
    z = np.zeros(n, dtype=np.float32)
    if n == 0 or winner is None:
        return z
    black_to_move = np.arange(n) % 2 == 0
    if winner == "B":
        z = np.where(black_to_move, 1.0, -1.0)
    elif winner == "W":
        z = np.where(black_to_move, -1.0, 1.0)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# single game
# ---------------------------------------------------------------------------

def play_game(
    network: torch.nn.Module,
    cfg: dict,
    *,
    size: "int | None" = None,
    komi: "float | None" = None,
    simulations: "int | None" = None,
    temperature_threshold: "int | None" = None,
    dirichlet_alpha: "float | None" = None,
    dirichlet_eps: "float | None" = None,
    resign_threshold: "float | None" = None,
    max_moves: "int | None" = None,
    seed: "int | None" = None,
    evaluator=None,
    frame_callback=None,
    leaf_batch: "int | None" = None,
    fp16: "bool | None" = None,
) -> dict:
    """Play one full self-play game with ``network`` on both sides.

    Args:
        network: policy-value network (put into ``eval()`` mode here).
        cfg: config dict (defaults for every ``None`` parameter).
        size: board edge (default ``config board_size`` = 19).
        komi: komi on white (default ``config komi`` = 7.5).
        simulations: MCTS simulations per move (default ``config
            simulations`` = 200).
        temperature_threshold: moves with ``tau = 1`` before argmax
            (default ``config temperature_threshold`` = 30).
        dirichlet_alpha: AGZ root-noise concentration. ``None`` uses the
            config (0.03); pass ``0.0`` to disable noise (deterministic).
        dirichlet_eps: noise blend weight (default ``config dirichlet_eps``
            = 0.25).
        resign_threshold: resign when the mover's root value drops below
            ``-resign_threshold`` (default ``config resign_threshold`` =
            0.0 -> disabled).
        max_moves: move cap; exceeding it force-terminates and scores the
            position (plan timeout protection, default 1000).
        seed: numpy seed for the game's RNG (noise + move sampling).
        evaluator: optional leaf evaluator (default
            :class:`BatchedNetworkEvaluator` built over ``network``).
        leaf_batch: leaves collected per network forward (default ``config
            leaf_batch`` = 16). Also threaded into the evaluator and MCTS so a
            ``cfg`` override takes effect everywhere (P11).
        fp16: run leaf inference under fp16 ``autocast`` on CUDA (default
            ``config selfplay_fp16`` = False; ``None`` falls back to the
            config). Games stay legal -- only the numerics of move selection
            may shift slightly.
        frame_callback: optional ``frame_callback(board, move_number, color)``
            invoked AFTER EVERY move (each stone / pass placement) with the
            LIVE rules ``Board`` right after that move, the 1-based
            ``move_number`` just played, and the ``color`` that played it.
            Callbacks run before the next search and are wrapped in
            try/except, so a failing callback can never break generation.
            ``None`` (default) keeps the generator exactly as before.

    Returns a game record dict:

        ``features`` (T,17,N,N) float32, ``pi`` (T,N*N+1) float32,
        ``z`` (T,) float32, ``winner`` ("B"/"W"/None), ``result``,
        ``forced_terminal`` (max-moves hit), ``resigned``,
        ``move_actions`` (policy indices, pass = N*N), ``colors``,
        ``move_count`` T, ``sims`` (total simulations), plus the game
        parameters (``board_size``, ``komi``, ``simulations``,
        ``temperature_threshold``, ``seed``).
    """
    size = int(size if size is not None else cfg["board_size"])
    komi = float(komi if komi is not None else cfg.get("komi", DEFAULT_KOMI))
    simulations = int(
        simulations if simulations is not None else cfg.get("simulations", 200)
    )
    temperature_threshold = int(
        temperature_threshold
        if temperature_threshold is not None
        else cfg.get("temperature_threshold", DEFAULT_TEMPERATURE_THRESHOLD)
    )
    dirichlet_alpha = (
        float(dirichlet_alpha)
        if dirichlet_alpha is not None
        else float(cfg.get("dirichlet_alpha", DEFAULT_DIRICHLET_ALPHA))
    )
    dirichlet_eps = (
        float(dirichlet_eps)
        if dirichlet_eps is not None
        else float(cfg.get("dirichlet_eps", DEFAULT_DIRICHLET_EPS))
    )
    resign_threshold = float(
        resign_threshold
        if resign_threshold is not None
        else cfg.get("resign_threshold", 0.0)
    )
    max_moves = int(max_moves if max_moves is not None else DEFAULT_MAX_MOVES)
    seed = int(seed) if seed is not None else 0

    # Inference discipline: batch-norm running stats must never be updated by
    # self-play (plan G3); the evaluator wraps forwards in torch.no_grad().
    network.eval()
    rng = np.random.default_rng(seed)
    board = Board(size)
    # P11: leaf_batch / fp16 are threaded from the config (or explicit kwargs)
    # into BOTH the evaluator and the MCTS batch cadence, so a cfg override
    # really changes how many leaves share one forward.
    leaf_batch = int(leaf_batch if leaf_batch is not None
                     else cfg.get("leaf_batch", DEFAULT_LEAF_BATCH))
    use_fp16 = bool(fp16 if fp16 is not None
                    else cfg.get("selfplay_fp16", False))
    if evaluator is None:
        evaluator = BatchedNetworkEvaluator(
            network, batch_size=leaf_batch, fp16=use_fp16)
    virtual_loss = int(cfg.get("virtual_loss", DEFAULT_VIRTUAL_LOSS))
    mcts = MCTS(
        network=network,
        evaluator=evaluator,
        komi=komi,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        dirichlet_rng=rng,
        virtual_loss=virtual_loss,
        leaf_batch=leaf_batch,
    )

    state_history = [board.state]
    positions: list[dict] = []  # {"features", "pi", "color", "action"}
    sims_total = 0
    move_number = 0
    resigned = False
    while not board.is_terminal() and move_number < max_moves:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        root = mcts.new_root(board)
        mcts.run(simulations)
        sims_total += simulations

        tau = 1.0 if move_number < temperature_threshold else 0.0

        # Resignation (config resign_threshold=0.0 disables it; AGZ's 0.5%
        # value threshold is a documented extension behind this bit).
        if (
            resign_threshold > 0.0
            and root.visit_count > 0
            and root.q_value < -resign_threshold
        ):
            resigned = True
            break

        action = sample_action(root, tau, rng=rng)
        # s: AGZ 17 planes with up to 8 prior game positions (most recent first).
        recent_states = state_history[-HISTORY_STEPS:][::-1]
        features = encode(recent_states, color, board_size=size)
        # pi: the temperature-softened search distribution (NOT a one-hot).
        pi = temperature_policy(root, tau)
        positions.append(
            {"features": features, "pi": pi, "color": color, "action": action}
        )
        if action == pass_index(size):
            board.pass_move(color)
        else:
            board.play((action // size, action % size), color)
        state_history.append(board.state)
        move_number += 1
        # F3d: stream the LIVE board to the visualization after EVERY move (a
        # stone or a pass) so the window refreshes every ~1-2 s instead of once
        # per finished game. The callback gets the live Board + the 1-based
        # move count + the color just played; it is fully try/except-guarded so
        # a failing callback can never break generation.
        if frame_callback is not None:
            try:
                frame_callback(board, move_number, color)
            except Exception:  # noqa: BLE001 - viz must never break self-play
                pass

    if resigned:
        resigning_color = BLACK if len(board.moves) % 2 == 0 else WHITE
        winner = "W" if resigning_color == BLACK else "B"
        forced = False
    else:
        winner = board.winner(komi)
        forced = not board.is_terminal()

    t = len(positions)
    if t:
        s = np.stack([p["features"] for p in positions]).astype(np.float32)
        pi = np.stack([p["pi"] for p in positions]).astype(np.float32)
    else:  # defensive: max_moves=0 or a terminal opening
        s = np.empty((0, 17, size, size), dtype=np.float32)
        pi = np.empty((0, size * size + 1), dtype=np.float32)
    z = z_targets(t, winner)

    return {
        "features": s,
        "pi": pi,
        "z": z,
        "winner": winner,
        "result": board.result_string(komi) if not resigned else f"{winner}+Resign",
        "forced_terminal": forced,
        "resigned": resigned,
        "move_actions": [p["action"] for p in positions],
        "colors": [p["color"] for p in positions],
        "move_count": t,
        "sims": sims_total,
        "board_size": size,
        "komi": komi,
        "simulations": simulations,
        "temperature_threshold": temperature_threshold,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# npz persistence (one file per game, atomic write)
# ---------------------------------------------------------------------------

def save_game_npz(record: dict, path: "str | Path") -> str:
    """Write a game record to ``path`` as a compressed npz (atomic).

    Keys: ``s`` (T,17,N,N) float32, ``pi`` (T,N*N+1) float32, ``z`` (T,)
    float32, plus scalar metadata (``board_size``, ``komi``, ``winner``,
    ``result``, ``move_count``, ``simulations``,
    ``temperature_threshold``, ``forced_terminal``, ``seed``). Written to a
    ``.tmp`` sibling then ``os.replace``d so a crash never leaves a corrupt
    file at the final name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # np.savez_compressed appends ".npz" to a filename that does not end in it,
    # so pass a file handle -- the ".tmp" sibling is renamed atomically below.
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            s=record["features"],
            pi=record["pi"],
            z=record["z"],
            board_size=np.int64(record["board_size"]),
            komi=np.float32(record["komi"]),
            winner=record["winner"] if record["winner"] else "Jigo",
            result=str(record["result"]),
            move_count=np.int64(record["move_count"]),
            simulations=np.int64(record["simulations"]),
            temperature_threshold=np.int64(record["temperature_threshold"]),
            forced_terminal=bool(record["forced_terminal"]),
            seed=np.int64(record["seed"]),
        )
    os.replace(tmp, path)
    return str(path)


def prune_old_games(data_dir: "str | Path", keep: int) -> list[str]:
    """Delete the oldest npz files beyond ``keep`` (oldest by mtime)."""
    data_dir = Path(data_dir)
    keep = max(1, int(keep))
    files = sorted(data_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime_ns)
    excess = files[: len(files) - keep] if len(files) > keep else []
    for f in excess:
        f.unlink()
    return [f.name for f in excess]


# ---------------------------------------------------------------------------
# batch generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# multi-process batch generation (P12)
# ---------------------------------------------------------------------------

def _network_arch(network: torch.nn.Module) -> dict:
    """Serializable arch description of a :class:`PolicyValueNetwork`.

    Workers>1 need to rebuild an identical model from scratch in a fresh
    process (a live module is not picklable across a Windows spawn), so the
    parent passes ``{blocks, channels, board_size}`` + the CPU ``state_dict``
    through the spawn initializer args instead of the module itself.
    """
    try:
        return {
            "blocks": int(network.blocks),
            "channels": int(network.channels),
            "board_size": int(network.board_size),
        }
    except AttributeError as exc:  # a custom net (ConstantNet & co.) has none
        raise ValueError(
            "generate_games(workers>1) requires a real PolicyValueNetwork "
            f"(create_model); got {type(network).__name__}"
        ) from exc


def _worker_main(state_dict, arch, cfg, data_dir, device, base_seed,
                 worker_index, workers, games, play_kwargs, out_q) -> None:
    """Spawn target (module-level, picklable): play this worker's game slice.

    Each worker process owns ONE model copy (rebuilt from ``state_dict`` +
    ``arch``), its own :class:`BatchedNetworkEvaluator` and MCTS tree, and its
    own CUDA context -- so MCTS serial phases run on separate CPU cores while
    leaf-network forwards from different workers overlap on the GPU.

    Seed distribution (deterministic, worker-count independent): game ``g`` of
    the batch (0-based) always gets ``seed = base_seed + g``; this worker plays
    the strided slice ``g in {worker_index, worker_index + workers, ...}``.
    That guarantees the npz set for a given ``(seed, games)`` is identical to
    the single-process path -- games are independent per seed.

    Viz limitation (documented): the per-move ``frame_callback`` is NOT
    forwarded from workers -- each worker has its own private board, so live
    per-move frames are not feasible cheaply across processes. Callers that
    want live viz use the buffer-refresh frame path (loop.py
    :func:`push_selfplay_frame`) after the batch lands instead.

    Results (``records`` + per-worker ``stats``) come back through ``out_q``,
    one ``dict`` per worker.
    """
    device = torch.device(device)
    network = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    network.load_state_dict(state_dict)
    network.eval()
    t0 = time.perf_counter()
    records: list[dict] = []
    sims = 0
    positions = 0
    for g in range(int(worker_index), int(games), int(workers)):
        rec = play_game(network, cfg, seed=int(base_seed) + g, **play_kwargs)
        rec["npz"] = save_game_npz(
            rec, Path(data_dir) / f"game_{rec['seed']:010d}.npz")
        records.append(rec)
        sims += int(rec["sims"])
        positions += int(rec["move_count"])
    wall = time.perf_counter() - t0
    out_q.put({
        "records": records,
        "stats": {
            "worker_index": int(worker_index),
            "games": len(records),
            "positions": positions,
            "sims": sims,
            "wall_time_s": wall,
            "sims_per_sec": sims / wall if wall > 0 else 0.0,
        },
    })


def _generate_games_parallel(network, cfg, games, data_dir, seed, workers,
                             play_kwargs) -> tuple[list[dict], float, list[dict]]:
    """Spawn ``workers`` processes; each generates its strided game slice.

    Returns ``(records, wall_time_s, worker_stats)``.

    ``wall_time_s`` is the *parent* batch wall clock (process spawn + model
    reload + CUDA init per worker included) -- the honest end-to-end number
    compared against the single-process baseline. Per-worker generation-only
    timings live in ``worker_stats``. ``records`` are sorted by seed so the
    returned list matches the single-process ordering (``seed + 0 .. N-1``).

    Results are drained from ``out_q`` in a background thread WHILE the parent
    joins: several workers pushing multi-MB records through one pipe would
    otherwise fill the pipe buffer and deadlock the join (each worker blocks in
    ``put`` until the reader consumes). The thread keeps the pipe drained; a
    crashed worker is detected via its non-zero ``exitcode`` after ``join``.
    """
    ctx = multiprocessing.get_context("spawn")  # Windows-safe, picklable args
    state_dict = {k: v.detach().cpu() for k, v in network.state_dict().items()}
    arch = _network_arch(network)
    device = str(next(network.parameters()).device)
    out_q = ctx.SimpleQueue()
    t0 = time.perf_counter()
    procs = [
        ctx.Process(
            target=_worker_main,
            args=(state_dict, arch, cfg, str(data_dir), device,
                  int(seed), w, workers, int(games), play_kwargs, out_q),
        )
        for w in range(workers)
    ]
    for p in procs:
        p.start()

    # Drain results concurrently with the join so a full pipe can never
    # deadlock the parent (see docstring). The thread is daemonic so a worker
    # that crashed without putting cannot block this process's exit.
    results: list = []

    def _drain() -> None:
        for _ in procs:
            results.append(out_q.get())

    threading.Thread(target=_drain, daemon=True, name="sp-worker-drain").start()
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    failed = [i for i, p in enumerate(procs) if p.exitcode not in (0, None)]
    if failed:
        raise RuntimeError(
            f"self-play worker process(es) exited with error: {failed} "
            "(see the worker traceback above; common causes: CUDA OOM, "
            "unpicklable **play_kwargs).")
    # all workers exited 0 -> each put exactly one item; wait briefly for the
    # daemon drain thread to finish copying them into ``results``.
    deadline = time.perf_counter() + 10.0
    while len(results) < workers and time.perf_counter() < deadline:
        time.sleep(0.05)
    if len(results) != workers:
        raise RuntimeError(
            f"expected {workers} worker result(s), collected {len(results)}")

    records = []
    for item in results:
        records.extend(item["records"])
    records.sort(key=lambda r: int(r["seed"]))
    worker_stats = [item["stats"] for item in results]
    worker_stats.sort(key=lambda s: int(s["worker_index"]))
    return records, wall, worker_stats


def generate_games(
    network: torch.nn.Module,
    cfg: dict,
    games: int,
    data_dir: "str | Path" = DEFAULT_DATA_DIR,
    keep: "int | None" = None,
    seed: int = 0,
    frame_callback=None,
    workers: int = 1,
    **play_kwargs,
) -> tuple[dict, list[dict]]:
    """Generate ``games`` self-play games, save one npz per game, prune.

    Seeds are ``seed + game_index`` (deterministic per game). Returns
    ``(report, records)`` where ``report`` holds the throughput baseline
    (``sims_per_sec`` = total search simulations / wall-clock seconds,
    positions/s, games/hour), the data directory, pruning results and the
    npz files present afterwards.

    ``frame_callback`` (optional) is forwarded to :func:`play_game`: it is
    invoked with the live board after EVERY move of every game (see
    :func:`play_game`). ``None`` (default) keeps behavior identical.

    ``workers`` (P12, default 1): number of processes. ``workers=1`` is the
    exact single-process path (byte-identical to pre-P12). ``workers>1`` spawns
    ``workers`` child processes (Windows ``spawn``); each child loads the model
    checkpoint once (the parent's ``state_dict`` + arch are serialized through
    the spawn initializer args), builds its own evaluator + MCTS, and plays a
    strided slice of the games so GPU forwards overlap while CPU cores
    parallelize the serial MCTS phases. Constraints:

    * Valid range ``1 .. MAX_SELFPLAY_WORKERS`` (3): each fp16 worker holds its
      own ~1.4GB model copy on the 6GB GPU, so 3 workers (~4.2GB) is the safe
      ceiling -- ``workers<1`` or ``workers>3`` raises ``ValueError``.
    * The network must be a :class:`~omigamax.network.model.PolicyValueNetwork`
      (rebuildable from ``blocks/channels/board_size`` + ``state_dict``); a
      live module cannot cross a Windows spawn boundary.
    * Per-move ``frame_callback`` frames are NOT streamed from workers (each
      worker owns a private board) -- document this and rely on the buffer-
      refresh viz path for workers>1.
    * ``report["wall_time_s"]`` is the batch wall clock including spawn/reload;
      ``report["per_worker"]`` holds per-worker generation-only timings.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    keep = int(keep if keep is not None else cfg.get("replay_buffer_games", 1000))
    games = int(games)
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers > MAX_SELFPLAY_WORKERS:
        raise ValueError(
            f"workers={workers} exceeds the cap of {MAX_SELFPLAY_WORKERS}: "
            "each fp16 worker holds its own ~1.4GB model copy on the 6GB GPU "
            "(3 workers ~= 4.2GB is the safe ceiling; 4 risks OOM). "
            "Use --selfplay-workers 1..3.")
    t0 = time.perf_counter()
    if workers == 1:
        records: list[dict] = []
        for g in range(games):
            rec = play_game(
                network, cfg, seed=seed + g, frame_callback=frame_callback,
                **play_kwargs)
            rec["npz"] = save_game_npz(
                rec, data_dir / f"game_{rec['seed']:010d}.npz")
            records.append(rec)
        wall = time.perf_counter() - t0
        worker_stats = [{
            "worker_index": 0,
            "games": len(records),
            "positions": sum(r["move_count"] for r in records),
            "sims": sum(r["sims"] for r in records),
            "wall_time_s": wall,
            "sims_per_sec": (sum(r["sims"] for r in records) / wall
                             if wall > 0 else 0.0),
        }]
    else:
        records, wall, worker_stats = _generate_games_parallel(
            network, cfg, games, data_dir, int(seed), workers, play_kwargs)

    pruned = prune_old_games(data_dir, keep)
    total_sims = sum(r["sims"] for r in records)
    total_positions = sum(r["move_count"] for r in records)
    report = {
        "games": len(records),
        "positions": total_positions,
        "sims": total_sims,
        "wall_time_s": wall,
        "sims_per_sec": total_sims / wall if wall > 0 else 0.0,
        "positions_per_sec": total_positions / wall if wall > 0 else 0.0,
        "games_per_hour": len(records) * 3600.0 / wall if wall > 0 else 0.0,
        "data_dir": str(data_dir),
        "keep_games": keep,
        "pruned": pruned,
        "simulations_per_move": records[0]["simulations"] if records else None,
        "npz_files": sorted(p.name for p in data_dir.glob("*.npz")),
    }
    if workers > 1:
        report["workers"] = workers
        report["per_worker"] = worker_stats
    return report, records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _per_game_summary(record: dict) -> dict:
    return {
        "seed": record["seed"],
        "winner": record["winner"],
        "result": record["result"],
        "move_count": record["move_count"],
        "sims": record["sims"],
        "forced_terminal": record["forced_terminal"],
        "resigned": record["resigned"],
        "npz": record["npz"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="omigamax todo-13 self-play data generator: "
                    "plays full games, writes (s, pi, z) npz per game, "
                    "reports sims/s throughput."
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"number of games to generate (default {DEFAULT_GAMES})")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move (default: config "
                             "simulations=200; plan acceptance uses 40)")
    parser.add_argument("--board-size", type=int, default=None,
                        help="board edge (default: config board_size=19)")
    parser.add_argument("--model", type=str, default=None,
                        help="self-play with a loaded checkpoint (e.g. "
                             "models/pretrain.pt): its recorded arch "
                             "(blocks/channels/board_size) wins over the "
                             "config")
    parser.add_argument("--komi", type=float, default=None,
                        help="komi on white (default: config komi=7.5)")
    parser.add_argument("--temperature-threshold", type=int, default=None,
                        help="moves with tau=1 before argmax "
                             "(default: config temperature_threshold=30)")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"move cap per game (default {DEFAULT_MAX_MOVES})")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"npz output directory (default {DEFAULT_DATA_DIR})")
    parser.add_argument("--keep-games", type=int, default=None,
                        help="npz files to keep (default: config "
                             "replay_buffer_games=1000)")
    parser.add_argument("--leaf-batch", type=int, default=None,
                        help="leaves per network forward (default: config "
                             "leaf_batch=16; larger batches amortize the GPU "
                             "call and Python overhead)")
    parser.add_argument("--fp16", action="store_true",
                        help="run leaf inference under fp16 autocast on CUDA "
                             "(self-play only; move numerics may shift "
                             "slightly but games stay legal)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master random seed (games use seed + index)")
    parser.add_argument("--selfplay-workers", type=int, default=1,
                        help="parallel worker processes (default 1 = the "
                             "single-process path; 2-3 spawn child processes "
                             "that each hold their own model copy + MCTS so "
                             "CPU cores parallelize the serial search phases "
                             "and GPU forwards overlap; capped at 3 on the "
                             "6GB GPU ~1.4GB fp16 per worker)")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the result JSON here (default: "
                             ".omo/evidence/omigamax-go/task-13-selfplay.json)")
    parser.add_argument("--no-log", action="store_true",
                        help="skip the logs/selfplay.jsonl throughput line")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model:
        # P7: a checkpoint's recorded arch wins -- a b20c256 net self-plays
        # at its own size even though the config default is b10c128/19.
        # Lazy import: buffer.py imports selfplay.py at module load, and
        # train.py imports buffer.py -- a module-level train import would
        # be a circular-import cycle.
        from omigamax.train.train import load_checkpoint
        ckpt = load_checkpoint(args.model)
        arch = ckpt["arch"]
        size = int(arch["board_size"])
        network = create_model(
            int(arch["blocks"]), int(arch["channels"]), size
        ).to(device)
        network.load_state_dict(ckpt["model_state_dict"])
    else:
        size = int(args.board_size if args.board_size is not None
                   else cfg["board_size"])
        network = create_model(
            int(cfg["blocks"]), int(cfg["channels"]), size
        ).to(device)

    report, records = generate_games(
        network,
        cfg,
        games=args.games,
        data_dir=args.data_dir,
        keep=args.keep_games,
        seed=args.seed,
        size=size,
        komi=args.komi,
        simulations=args.simulations,
        temperature_threshold=args.temperature_threshold,
        max_moves=args.max_moves,
        leaf_batch=args.leaf_batch,
        fp16=args.fp16,
        workers=args.selfplay_workers,
    )

    warned = report["sims_per_sec"] < SOFT_SIMS_PER_SEC_WARN
    result = {
        "todo": 13,
        "device": str(device),
        "protocol": {
            "games": args.games,
            "simulations_per_move": report["simulations_per_move"],
            "board_size": size,
            "komi": float(args.komi) if args.komi is not None else float(cfg.get("komi", DEFAULT_KOMI)),
            "temperature_threshold": (
                int(args.temperature_threshold)
                if args.temperature_threshold is not None
                else int(cfg.get("temperature_threshold", DEFAULT_TEMPERATURE_THRESHOLD))
            ),
            "max_moves": args.max_moves,
            "data_dir": report["data_dir"],
            "keep_games": report["keep_games"],
            "master_seed": args.seed,
            "selfplay_workers": args.selfplay_workers,
            "dirichlet_alpha": float(cfg.get("dirichlet_alpha", DEFAULT_DIRICHLET_ALPHA)),
            "dirichlet_eps": float(cfg.get("dirichlet_eps", DEFAULT_DIRICHLET_EPS)),
        },
        "report": report,
        "games": [_per_game_summary(r) for r in records],
        "soft_gate": {
            "warn_threshold_sims_per_sec": SOFT_SIMS_PER_SEC_WARN,
            "sims_per_sec": report["sims_per_sec"],
            "warned": warned,
            "suggestion": SOFT_GATE_SUGGESTION if warned else None,
        },
        "accepted": True,  # exit 0 always (plan: soft gate only)
    }

    _print_report(result)

    if not args.no_log:
        log_path = Path("logs") / "selfplay.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "event": "selfplay_generate",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": str(device),
            "games": report["games"],
            "positions": report["positions"],
            "sims": report["sims"],
            "sims_per_sec": round(report["sims_per_sec"], 3),
            "positions_per_sec": round(report["positions_per_sec"], 3),
            "games_per_hour": round(report["games_per_hour"], 3),
            "data_dir": report["data_dir"],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"log line appended: {log_path}", flush=True)

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0  # plan: exit 0 恒通过 (sims/s soft gate never fails the run)


def _print_report(result: dict) -> None:
    proto = result["protocol"]
    rep = result["report"]
    print("=== omigamax self-play data generator (todo 13) ===", flush=True)
    print(f"device: {result['device']}", flush=True)
    print(
        f"protocol: games={proto['games']} simulations/move="
        f"{proto['simulations_per_move']} board={proto['board_size']} "
        f"komi={proto['komi']} temperature_threshold="
        f"{proto['temperature_threshold']} max_moves={proto['max_moves']} "
        f"data_dir={proto['data_dir']} keep={proto['keep_games']}", flush=True
    )
    for g in result["games"]:
        forced = " (max-moves forced)" if g["forced_terminal"] else ""
        print(
            f"  game seed={g['seed']}: moves={g['move_count']} "
            f"sims={g['sims']} result={g['result']}{forced}", flush=True
        )
    print(
        f"throughput: games={rep['games']} positions={rep['positions']} "
        f"sims={rep['sims']} wall={rep['wall_time_s']:.1f}s", flush=True
    )
    print(
        f"  sims/s = {rep['sims_per_sec']:.1f}  positions/s = "
        f"{rep['positions_per_sec']:.1f}  games/hour = "
        f"{rep['games_per_hour']:.1f}", flush=True
    )
    sg = result["soft_gate"]
    if sg["warned"]:
        print(
            f"SOFT GATE WARN: sims/s {sg['sims_per_sec']:.1f} < "
            f"{sg['warn_threshold_sims_per_sec']} -- {sg['suggestion']}",
            flush=True,
        )
    else:
        print(
            f"soft gate: sims/s {sg['sims_per_sec']:.1f} >= "
            f"{sg['warn_threshold_sims_per_sec']} (no warning)", flush=True
        )
    print(f"npz files written ({len(rep['npz_files'])}): {rep['npz_files'][:5]}"
          f"{' ...' if len(rep['npz_files']) > 5 else ''}", flush=True)
    print(f"pruned (beyond keep={rep['keep_games']}): {rep['pruned']}", flush=True)
    print("RESULT: PASS (exit 0; sims/s soft gate is a warning, not a failure)",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
