"""Training-loop orchestration: self-play -> train -> eval gate -> repeat (todo 16).

Per the plan (todo 16, authoritative) and AGZ (Nature 550, 2017, Methods fig. 1):

* the loop alternates three phases around the todo-13/14/15 components:
  ``generate_games`` (self-play, todo 13) -> ``train_steps`` (todo 14) ->
  ``evaluate_and_gate`` (todo 15 -- candidate vs ``best.pt``; a win rate
  ``>= replace_threshold`` (default 0.55) replaces it);
* **cycle rhythm** (plan, Oracle F3): the default cycle is ``100`` self-play
  games -> ``1000`` training steps -> an evaluation gate at the END OF EVERY
  CYCLE. ``eval_interval_steps`` (config = 2000) is an optional *sparser*
  interval for long training: a boundary crossed mid-cycle also fires the
  gate, but it never suppresses the cycle-end gate (``不覆盖 cycle 末评估``).
  ``--smoke`` forces a final gate so a short acceptance run always produces
  ``best.pt`` (plan: otherwise the smoke would never yield the ``>= 2
  checkpoint`` acceptance);
* **interruptible resume** (plan): ``Ctrl+C`` (KeyboardInterrupt) and
  ``SIGBREAK`` (Ctrl+Break, Windows) are caught and the run saves
  ``models/latest.pt`` immediately -- weights, SGD optimizer state,
  ``global_step``, the buffer-sampling RNG state AND the in-flight cycle
  progress (``games_generated`` / ``steps_into_cycle``) -- flushes the JSONL
  log and exits. A hard crash loses at most the un-checkpointed part of the
  current cycle. Restarting with ``--resume`` reloads ``latest.pt`` and the
  ``data/selfplay`` npz games and continues exactly where training stopped
  (deterministic-resume, Oracle F9: fixed seed + ``cudnn.deterministic=True``
  + persisted numpy RNG; compared with the plan's ``1e-4`` tolerance);
* **logging**: one JSONL line per training step in ``logs/train.jsonl`` with
  ``step`` / ``loss`` / ``lr`` / ``games`` (replay-buffer game count) /
  ``elo`` (last evaluation-gate rating) / ``timestamp`` (plan, Oracle G2 --
  the acceptance asserts loss/elo fields on >= 20 lines);
* **lazy visualization** (plan, Oracle #9/F3): when ``config viz_enabled=true``
  the loop mounts the todo-17 pygame thread through a *lazy import*
  (:func:`start_viz_if_available`): it starts a daemon
  :class:`~omigamax.viz.board_window.VizThread` over a bounded snapshot queue
  and returns a handle (``queue`` / ``thread`` / ``stop``). Any failure
  (module absent, pygame init error, ...) degrades to pure-log mode with a
  warning -- it never crashes a training run; ``--viz off`` force-disables
  the mount. The thread is stopped cleanly at the end of the run. The loop
  *feeds* the window (F2 MAJOR 2): one Snapshot frame per training step --
  a board reconstructed once per cycle from the newest self-play position
  plus the live ``loss`` / ``train_step`` / ``games`` / ``elo`` metrics --
  pushed non-blockingly through :func:`push_viz_frame` (drop-oldest,
  try/except-wrapped, so the visualization can never slow or crash training).
  The **self-play phase** feeds it too (F3b): right after each cycle's
  ``generate_games`` batch lands on disk and the buffer refreshes, one frame
  of the just-finished game is pushed so the window opens while self-play is
  still running -- not only once train steps start. F3c sharpens this: an
  empty-board **opening frame** is pushed immediately after the viz thread
  starts (window pops up within seconds of launch), and self-play now
  generates one game at a time with a frame pushed after EACH game, so the
  user watches each game finish instead of waiting for the whole batch.
  F3d sharpens this to REAL-TIME: ``generate_games`` receives a
  ``frame_callback`` that streams a Snapshot built directly from the LIVE
  board after EVERY move (each stone / pass placement), so the window
  refreshes every ~1-2 s during self-play instead of only when a game ends.

Single-process self-play only (plan Must-NOT: no multiprocessing yet).

Usage::

    uv run python -m omigamax.train.loop --smoke                 # plan acceptance
    uv run python -m omigamax.train.loop --cycles 1              # one full cycle
    uv run python -m omigamax.train.loop --resume --cycles 1     # continue
    uv run python -m omigamax.train.loop --smoke --resume --interrupt-at-steps 6
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE
from omigamax.train.buffer import ReplayBuffer, list_game_files
from omigamax.train.evaluate import (
    DEFAULT_EVAL_GAMES,
    DEFAULT_EVAL_SIMS,
    best_checkpoint_path,
    evaluate_and_gate,
    read_last_elo,
)
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.selfplay import (
    DEFAULT_DATA_DIR,
    MAX_SELFPLAY_WORKERS,
    generate_games,
)
from omigamax.train.train import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_GRAD_CLIP,
    latest_checkpoint_path,
    load_checkpoint,
    restore_from_checkpoint,
    restore_rng,
    save_checkpoint,
    train_steps,
)

log = logging.getLogger("omigamax.loop")

# plan todo 16 cycle rhythm: 100 self-play games -> 1000 train steps -> gate
DEFAULT_CYCLE_GAMES = 100
DEFAULT_CYCLE_STEPS = 1000
DEFAULT_CYCLES = 1
DEFAULT_TRAIN_LOG = "logs/train.jsonl"
DEFAULT_HISTORY = "logs/eval_history.jsonl"
DEFAULT_EVIDENCE = ".omo/evidence/omigamax-go/task-16-loop.json"
DEFAULT_EVAL_MAX_MOVES = 1000
DEFAULT_PRETRAIN_DATA_DIR = Path("data") / "pretrain"

# P16-7: mid-selfplay checkpoint cadence. A checkpoint is written every N
# completed games inside a cycle (0 disables mid-cycle saves), persisting the
# cycle-base snapshot + games_this_cycle so a resume continues generating only
# the remaining games instead of redoing the completed ones.
DEFAULT_SAVE_EVERY_GAMES = 10

# --smoke low-config preset (plan: sims=40, batch=32, ~100-games scale; the
# acceptance demo runs a smaller slice so one full cycle + gate completes in
# minutes on the 3060 while still exercising every phase). The game-length
# budgets are deliberately small: with weak models both self-play and tau=0
# eval games can run to 1000 moves (todo-13 measured 2.9 min/game on 19x19),
# so the smoke bounds both so an acceptance run stays in minutes.
SMOKE_PRESET = {
    "cycle_games": 2,
    "cycle_steps": 25,
    "simulations": 40,
    "batch_size": 32,
    "selfplay_max_moves": 150,
    "eval_games": 3,
    "eval_sims": 20,
    "eval_max_moves": 150,
}


# ---------------------------------------------------------------------------
# architecture plumbing (P7: b20c256 RL fine-tuning without touching
# config/default.yaml, which stays b10c128)
# ---------------------------------------------------------------------------

def apply_arch_overrides(
    cfg: dict,
    *,
    blocks: "int | None" = None,
    channels: "int | None" = None,
    board_size: "int | None" = None,
) -> dict:
    """Copy ``cfg`` with explicit ``--blocks`` / ``--channels`` /
    ``--board-size`` overrides applied.

    P7 design: ``config/default.yaml`` is the immutable b10c128 acceptance
    baseline; b20c256 is selected either by an explicit CLI override here or
    by a loaded checkpoint's recorded arch (see :func:`_load_or_init`, which
    gives checkpoints priority over the config). Returns a fresh dict so the
    shared config object is never mutated.
    """
    cfg = dict(cfg)
    if blocks is not None:
        cfg["blocks"] = int(blocks)
    if channels is not None:
        cfg["channels"] = int(channels)
    if board_size is not None:
        cfg["board_size"] = int(board_size)
    return cfg


# ---------------------------------------------------------------------------
# evaluation scheduling
# ---------------------------------------------------------------------------

def eval_due(step_after: int, *, cycle_end: bool, eval_interval_steps: int) -> bool:
    """Whether the evaluation gate should fire after ``step_after``.

    The cycle-end gate is the plan's primary trigger (Oracle F3: 每 cycle
    评估) and always fires. ``eval_interval_steps`` (config 2000) is the
    optional sparser interval for long training: a boundary crossed exactly at
    ``step_after`` (i.e. ``step_after`` is a positive multiple of the interval)
    also fires the gate. The interval never *suppresses* a cycle-end gate.
    """
    if cycle_end:
        return True
    interval = int(eval_interval_steps)
    if interval > 0:
        return (int(step_after) // interval) > ((int(step_after) - 1) // interval)
    return False


# ---------------------------------------------------------------------------
# lazy visualization mount-point (todo 17 does not exist yet)
# ---------------------------------------------------------------------------

def start_viz_if_available(cfg: dict, logger=None) -> dict:
    """Mount-point for the todo-17 pygame visualization (lazy, optional).

    Per the plan (Oracle #9/F3, todo 16/17): the config default
    ``viz_enabled=true`` must never crash a training run. The import of
    ``omigamax.viz.board_window`` is deferred inside this function; any
    failure (module absent, pygame init error, ...) degrades to pure-log mode
    with a warning and the training loop keeps running.

    On success a :class:`VizThread` is started over a bounded
    :class:`SnapshotQueue` (todo 17: window close only stops the thread, it
    never interrupts training). The handle carries ``queue`` (the channel the
    self-play/training loop pushes frames to), ``thread`` and a ``stop``
    callable the loop uses to clean up at the end of a run.

    Returns a small dict describing the outcome (``started`` + ``reason`` +
    handle); the training loop keeps running either way.
    """
    logger = logger or log
    if not bool(cfg.get("viz_enabled", True)):
        return {"started": False, "reason": "disabled_by_config"}
    try:
        from omigamax.viz.board_window import SnapshotQueue, VizThread
    except Exception as exc:  # ImportError / pygame / anything
        logger.warning(
            "viz module unavailable: %s -- continuing in pure-log mode", exc)
        return {"started": False, "reason": "module_unavailable",
                "error": str(exc)}
    try:
        queue = SnapshotQueue(maxlen=int(cfg.get("viz_queue_size", 32)))
        thread = VizThread(queue, logger=logger)
        thread.start()
    except Exception as exc:
        logger.warning("viz thread failed to start: %s -- "
                       "continuing in pure-log mode", exc)
        return {"started": False, "reason": "thread_failed",
                "error": str(exc)}
    logger.info("viz thread started -- visualization mount point active")
    return {"started": True, "reason": "available",
            "queue": queue, "thread": thread, "stop": thread.stop}


# ---------------------------------------------------------------------------
# viz frame feeding (F2 MAJOR 2: the live window must actually show frames)
# ---------------------------------------------------------------------------

def _reconstruct_board_state(planes, board_size: int) -> list:
    """Decode an AGZ position's 17 planes into a 0/1/2 colour-code board.

    Plane 0 holds the current player's stones, plane 1 the opponent's, and
    plane 16 tells which colour is to play (the same layout
    :func:`omigamax.network.features.encode` produces). Returns the 2-D
    ``(board_size, board_size)`` list the viz ``Snapshot`` expects. Pure
    numpy; never raises.
    """
    n = int(board_size)
    current_is_black = bool(planes[16, 0, 0] > 0.5)
    cur, opp = (BLACK, WHITE) if current_is_black else (WHITE, BLACK)
    state = np.zeros((n, n), dtype=int)
    state[planes[0] > 0.5] = cur
    state[planes[1] > 0.5] = opp
    return state.tolist()


def viz_board_info(buffer, board_size: int) -> "dict | None":
    """Best-effort board frame from the newest self-play game.

    Reads the replay buffer's most recent npz game and reconstructs its last
    recorded position, so the live window shows a real self-play board behind
    the updating metrics panel. Returns ``None`` when no position is
    available or the reconstruction fails -- the loop keeps training either
    way. Call once per cycle: per-step frames reuse this board and only
    refresh the metrics (``train_step`` / ``loss`` / ``games`` / ``elo``).
    """
    try:
        files = list_game_files(buffer.data_dir, keep=1)
        if not files:
            return None
        with np.load(files[-1]) as data:
            s = np.asarray(data["s"], dtype=np.float32)
            move_count = int(np.asarray(data["move_count"], dtype=np.int64))
        if s.shape[0] == 0:
            return None
        planes = s[-1]
        n = int(planes.shape[-1])
        return {
            "board": _reconstruct_board_state(planes, n),
            "board_size": n,
            "move_number": int(move_count),
            "current_player": int(BLACK if planes[16, 0, 0] > 0.5 else WHITE),
        }
    except Exception:  # noqa: BLE001 - viz must never break the loop
        return None


def build_viz_snapshot(board_info, *, komi, games, train_step, loss, elo,
                       win_rate=None, last_move=None):
    """A viz ``Snapshot`` over ``board_info`` + the latest training metrics.

    Lazy-imports the viz module so a missing pygame can never raise here.
    Returns ``None`` (no frame) when no board is available or the module is
    absent -- the loop keeps training either way.
    """
    if board_info is None:
        return None
    try:
        from omigamax.viz.board_window import Snapshot
    except Exception:  # noqa: BLE001 - viz must never break the loop
        return None
    return Snapshot(
        board=board_info["board"],
        board_size=int(board_info["board_size"]),
        move_number=int(board_info["move_number"]),
        current_player=int(board_info["current_player"]),
        win_rate=win_rate,
        last_move=last_move,
        komi=float(komi),
        games=int(games) if games is not None else None,
        train_step=int(train_step) if train_step is not None else None,
        loss=float(loss) if loss is not None else None,
        elo=float(elo) if elo is not None else None,
    )


def viz_snapshot_from_board(board, board_size, move_number, current_player,
                            last_move, *, komi, games, train_step, loss, elo,
                            win_rate=None):
    """A viz ``Snapshot`` built directly from a LIVE rules ``Board`` (F3d).

    Unlike ``viz_board_info`` + :func:`build_viz_snapshot` (which read the
    newest npz and therefore only exist at game end), this builds the frame
    straight from the in-memory ``Board`` right after a move, so the window
    can stream EVERY move during self-play in real time. ``board`` is a live
    ``omigamax.rules.board.Board``; ``move_number`` is the 1-based count of
    moves just played; ``current_player`` is the side to move next; and
    ``last_move`` is the ``(row, col)`` of the most recent stone (``None``
    for a pass). Lazy-imports the viz module so a missing pygame can never
    raise here; returns ``None`` when the module is absent -- the trainer
    never depends on viz. O(board_size^2) list copy + dataclass construction
    only: sub-millisecond, no encoding/render work in the trainer thread.
    """
    try:
        from omigamax.viz.board_window import Snapshot
    except Exception:  # noqa: BLE001 - viz must never break the loop
        return None
    n = int(board_size)
    flat = board.state
    grid = [list(flat[r * n:(r + 1) * n]) for r in range(n)]
    return Snapshot(
        board=grid,
        board_size=n,
        move_number=int(move_number),
        current_player=int(current_player),
        win_rate=win_rate,
        last_move=last_move,
        komi=float(komi),
        games=int(games) if games is not None else None,
        train_step=int(train_step) if train_step is not None else None,
        loss=float(loss) if loss is not None else None,
        elo=float(elo) if elo is not None else None,
    )


def push_viz_frame(viz, snap) -> bool:
    """Push one ``Snapshot`` to the viz queue; never blocks, never raises.

    Safe from the trainer thread: ``SnapshotQueue.push`` is non-blocking and
    drop-oldest, and ANY failure here (queue gone, window closed, malformed
    frame) is swallowed so visualization can never slow or crash training.
    Returns True when a frame was enqueued.
    """
    queue = viz.get("queue") if isinstance(viz, dict) else None
    if queue is None or snap is None:
        return False
    try:
        queue.push(snap)
        return True
    except Exception:  # noqa: BLE001 - viz must never break training
        return False


def push_selfplay_frame(viz, buffer, board_size, *, komi, games, train_step,
                        elo) -> bool:
    """Push one board frame from the newest self-play game (F3b).

    Called right after a cycle's ``generate_games`` batch lands and the
    buffer refreshes: the newest npz IS the just-finished game, so the live
    window shows that game's final position during the self-play phase
    instead of staying dark until the first train step pushes a frame (F2).
    Non-blocking / try-except-wrapped like :func:`push_viz_frame` -- viz can
    never slow or crash training. Returns True when a frame was enqueued.
    """
    if not viz.get("started"):
        return False
    board_info = viz_board_info(buffer, board_size)
    if board_info is None:
        return False
    return push_viz_frame(viz, build_viz_snapshot(
        board_info, komi=komi, games=games, train_step=train_step,
        loss=None, elo=elo))


def push_opening_frame(viz, board_size, *, komi) -> bool:
    """Push an empty-board opening frame so the window appears immediately.

    F3c: the pygame window is opened lazily by ``VizThread`` only once the
    FIRST frame arrives, so with viz on it used to stay invisible until the
    first self-play game (or train step) finished -- ~25 min at low sims.
    Calling this right after :func:`start_viz_if_available` returns
    ``started=True`` (before the cycle loop even runs) makes the window pop
    up within seconds of launch, showing an empty board (move 0, black to
    play, no metrics yet).

    Built directly on the ``Snapshot`` dataclass because the buffer is empty
    at launch (``viz_board_info`` would return ``None``). Non-blocking /
    try-except-wrapped like :func:`push_viz_frame` -- viz can never slow or
    crash training. Returns True when a frame was enqueued.
    """
    if not viz.get("started"):
        return False
    try:
        from omigamax.viz.board_window import Snapshot
    except Exception:  # noqa: BLE001 - viz must never break the loop
        return False
    n = int(board_size)
    empty_board = [[0] * n for _ in range(n)]
    return push_viz_frame(viz, Snapshot(
        board=empty_board,
        board_size=n,
        move_number=0,
        current_player=BLACK,
        komi=float(komi),
        games=0,
        train_step=None,
        loss=None,
        elo=None,
    ))


# ---------------------------------------------------------------------------
# JSONL helpers (explicit UTF-8, one JSON object per line)
# ---------------------------------------------------------------------------

def _append_jsonl(path: "str | Path", entry: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(path)


def _log_train_step(train_log, *, step, loss, lr, games, elo, cycle) -> str:
    """Per-training-step JSONL record (plan, Oracle G2)."""
    return _append_jsonl(train_log, {
        "event": "train_step",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step": int(step),
        "loss": round(float(loss), 6),
        "lr": round(float(lr), 6),
        "games": int(games),
        "elo": round(float(elo), 3),
        "cycle": int(cycle),
    })


def _log_loop_event(train_log, event: str, **fields) -> str:
    return _append_jsonl(
        train_log, {"event": event,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields})


# ---------------------------------------------------------------------------
# model / optimizer / RNG (re)loading
# ---------------------------------------------------------------------------

def _model_from_checkpoint(
    cfg: dict,
    ckpt: dict,
    device: torch.device,
    seed: int,
    *,
    resumed: bool,
    init_checkpoint: "str | Path | None",
) -> dict:
    """Build the run state from an existing checkpoint (resume or warm-start).

    The checkpoint's recorded ``arch`` wins over the config (P7: a
    ``models/pretrain.pt`` written at b20c256 restores a b20c256 net even
    though ``config/default.yaml`` says b10c128). The SGD optimizer is rebuilt
    with the loop's hyper-parameters (``lr``/``momentum``/``l2`` from cfg;
    ``train_steps`` re-applies the scheduled lr before every step anyway) and
    ``optimizer.load_state_dict`` restores its momentum buffers. The buffer
    sampling RNG is restored from ``rng_state`` when present so a resumed run
    stays deterministic (Oracle F9).
    """
    arch = ckpt["arch"]
    model = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"]),
    ).to(device)
    optimizer = make_sgd_optimizer(
        model,
        lr=float(cfg.get("lr", 0.2)),
        momentum=float(cfg.get("momentum", 0.9)),
        l2=float(cfg.get("l2", 1e-4)),
    )
    global_step = restore_from_checkpoint(ckpt, model, optimizer)
    rng = np.random.default_rng(int(seed))
    if "rng_state" in ckpt:
        restore_rng(rng, ckpt["rng_state"])
    extra = ckpt.get("extra") or {}
    return {
        "model": model, "optimizer": optimizer, "rng": rng,
        "global_step": int(global_step),
        "games_generated": int(extra.get("games_generated", 0)),
        "steps_into_cycle": int(extra.get("steps_into_cycle", 0)),
        "cycles_completed": int(extra.get("cycles_completed", 0)),
        # P16-7: completed games in the current cycle (missing on old
        # checkpoints -> 0, i.e. a fresh cycle / full regeneration).
        "games_this_cycle": int(extra.get("games_this_cycle", 0)),
        "resumed": resumed,
        "init_checkpoint": (
            None if init_checkpoint is None else str(init_checkpoint)
        ),
        # P13: the persisted config snapshot of the loaded checkpoint -- the
        # source for run-param restore on resume (human_mix / workers / ...).
        "ckpt_config": ckpt.get("config") or {},
    }


def _load_or_init(
    cfg: dict,
    checkpoint_path,
    device,
    seed: int,
    resume: bool,
    init_checkpoint: "str | Path | None" = None,
) -> dict:
    """Build the run state: resume, pretrained warm-start, or fresh init.

    Architecture priority (P7, from highest to lowest):

    1. ``models/latest.pt`` on ``--resume`` -- its recorded arch restores the
       exact net that trained it;
    2. ``init_checkpoint`` (e.g. ``models/pretrain.pt``) -- a *fresh* RL run
       warm-started from the pretrained weights, its recorded arch winning
       over the config;
    3. the config (with any explicit ``--blocks``/``--channels``/
       ``--board-size`` overrides the caller already applied).

    ``config/default.yaml`` is never modified: b20c256 is selected by a
    checkpoint's arch or by explicit flags, and the b10c128 config default
    stays untouched. The buffer-sampling numpy RNG is restored from the
    persisted ``rng_state`` so deterministic-resume stays exact (Oracle F9).
    """
    ckpt_path = Path(checkpoint_path)
    if resume and ckpt_path.exists():
        return _model_from_checkpoint(
            cfg, load_checkpoint(ckpt_path), device, int(seed),
            resumed=True, init_checkpoint=None,
        )
    if init_checkpoint is not None:
        return _model_from_checkpoint(
            cfg, load_checkpoint(init_checkpoint), device, int(seed),
            resumed=False, init_checkpoint=init_checkpoint,
        )
    torch.manual_seed(int(seed))
    model = create_model(
        int(cfg["blocks"]), int(cfg["channels"]), int(cfg["board_size"]),
    ).to(device)
    optimizer = make_sgd_optimizer(
        model,
        lr=float(cfg.get("lr", 0.2)),
        momentum=float(cfg.get("momentum", 0.9)),
        l2=float(cfg.get("l2", 1e-4)),
    )
    rng = np.random.default_rng(int(seed))
    return {
        "model": model, "optimizer": optimizer, "rng": rng,
        "global_step": 0, "games_generated": 0,
        "steps_into_cycle": 0, "cycles_completed": 0, "resumed": False,
        "games_this_cycle": 0,
        "init_checkpoint": None,
        "ckpt_config": {},
    }


# ---------------------------------------------------------------------------
# evaluation gate driver
# ---------------------------------------------------------------------------

def _run_eval_gate(model, optimizer, cfg, *, latest, best, history, device,
                   games, sims, seed, max_moves: int = DEFAULT_EVAL_MAX_MOVES,
                   board_size: "int | None" = None) -> dict:
    """Run the todo-15 gate: candidate (``latest``) vs ``best``, record ELO.

    ``board_size`` defaults to the *model's* architecture (P7: a b20c256 /
    9x9 warm-started net must be evaluated on its own board size, not the
    config's) -- the checkpoint arch drives both the net and the eval games.
    """
    return evaluate_and_gate(
        str(latest), str(best), cfg,
        games=games, sims=sims,
        size=int(board_size if board_size is not None
                 else model.board_size),
        komi=float(cfg.get("komi", 7.5)),
        virtual_loss=int(cfg.get("virtual_loss", 3)),
        max_moves=int(max_moves),
        seed=seed, device=device, history_path=str(history),
    )


def _ev_summary(ev: dict) -> dict:
    """Compact per-gate summary for the report (robust to mocked reports)."""
    match = ev.get("match", {}) or {}
    elo_update = ev.get("elo_update", {}) or {}
    return {
        "step": int(ev.get("protocol", {}).get("candidate_global_step", 0)),
        "winrate": round(float(match.get("winrate", ev.get("winrate", 0.0))), 4),
        "candidate_wins": int(match.get("candidate_wins",
                                        ev.get("candidate_wins", 0))),
        "games": int(match.get("games", ev.get("games", 0))),
        "replaced": bool(ev.get("replaced_best", ev.get("replaced", False))),
        "elo": round(float(elo_update.get("elo", 0.0)), 3),
    }


# ---------------------------------------------------------------------------
# the main loop
# ---------------------------------------------------------------------------

def run_loop(
    cfg: dict,
    *,
    device=None,
    data_dir: "str | Path" = DEFAULT_DATA_DIR,
    checkpoint_dir: "str | Path" = DEFAULT_CHECKPOINT_DIR,
    train_log: "str | Path" = DEFAULT_TRAIN_LOG,
    history: "str | Path" = DEFAULT_HISTORY,
    cycles: "int | None" = None,
    games_per_cycle: "int | None" = None,
    steps_per_cycle: "int | None" = None,
    steps_budget: "int | None" = None,
    simulations: "int | None" = None,
    selfplay_max_moves: "int | None" = None,
    batch_size: "int | None" = None,
    eval_games: "int | None" = None,
    eval_sims: "int | None" = None,
    eval_interval_steps: "int | None" = None,
    eval_max_moves: "int | None" = None,
    replace_threshold: "float | None" = None,
    use_symmetry: bool = True,
    use_fp16: bool = False,
    grad_clip: "float | None" = None,
    seed: int = 0,
    resume: bool = False,
    init_checkpoint: "str | Path | None" = None,
    force_final_eval: bool = False,
    viz_enabled: "bool | None" = None,
    logger=None,
    interrupt_after: "int | None" = None,
    human_mix: "float | None" = None,
    pretrain_data_dir: "str | Path | None" = None,
    leaf_batch: "int | None" = None,
    selfplay_fp16: bool = False,
    selfplay_workers: "int | None" = None,
    save_every_games: "int | None" = None,
) -> dict:
    """Run the self-play -> train -> eval-gate loop (interruptible, resumable).

    ``cycles`` full cycles are executed (``cycles is None`` -> 1, or enough to
    consume ``steps_budget`` total training steps when given). Each cycle
    generates ``games_per_cycle`` self-play games, trains ``steps_per_cycle``
    steps (one JSONL ``train_step`` line each), and fires the evaluation gate
    at the cycle end (and at ``eval_interval_steps`` boundaries, see
    :func:`eval_due`). ``latest.pt`` is checkpointed before every gate and at
    the end of the run; a graceful interrupt (KeyboardInterrupt / SIGBREAK)
    checkpoint the in-flight state so ``--resume`` continues loss-exactly.
    ``init_checkpoint`` warm-starts a *fresh* RL run from an external
    checkpoint (e.g. ``models/pretrain.pt``): its recorded arch wins over the
    config and the pretrained weights become the starting point (P7 RL
    fine-tuning at b20c256; ignored when ``resume`` finds ``latest.pt``).

    ``human_mix`` / ``selfplay_workers`` / ``pretrain_data_dir`` are run-only
    params (NOT in ``config/default.yaml``) that shape training-data
    composition. ``None`` means "resolve": an explicit arg wins, else the
    resumed checkpoint's recorded config (P13: a bare ``--resume`` after a
    machine restart restores them automatically), else ``cfg``, else the safe
    default (human_mix 0.0, workers 1). The resolved values are written into
    a *copy* of ``cfg`` which ``_save_ckpt`` persists, so every new
    checkpoint auto-restores them; old checkpoints without the keys fall back
    to the safe defaults and still need explicit flags on resume.

    ``save_every_games`` (P16-7, default 10; 0 disables) is another run-only
    param, resolved the same way and persisted the same way. When > 0, a
    mid-selfplay checkpoint is written every N completed games of a cycle
    (from the workers>1 drain thread, or from the workers==1 per-game loop).
    Such a checkpoint records ``games_generated`` as the CYCLE BASE snapshot
    plus a new ``games_this_cycle`` counter, so a later ``--resume`` with
    ``games_this_cycle > 0`` and ``steps_into_cycle == 0`` continues by
    generating only ``games_per_cycle - games_this_cycle`` games (seeds from
    ``seed + games_generated + games_this_cycle``) -- completed games are
    never regenerated. ``games_this_cycle`` is cleared ONLY when the cycle
    completes; mid-cycle eval-gate saves never clear it.

    Returns a report dict (also used as the todo-16 evidence JSON).
    """
    logger = logger or log
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    train_log = Path(train_log)
    history = Path(history)

    games_per_cycle = int(games_per_cycle if games_per_cycle is not None
                          else cfg.get("cycle_games", DEFAULT_CYCLE_GAMES))
    steps_per_cycle = int(steps_per_cycle if steps_per_cycle is not None
                          else cfg.get("cycle_steps", DEFAULT_CYCLE_STEPS))
    simulations = int(simulations if simulations is not None
                      else cfg.get("simulations", 200))
    selfplay_max_moves = (
        int(selfplay_max_moves) if selfplay_max_moves is not None
        else (int(cfg["selfplay_max_moves"]) if "selfplay_max_moves" in cfg
              else None))
    batch_size = int(batch_size if batch_size is not None
                     else cfg.get("batch_size", 128))
    eval_games = int(eval_games if eval_games is not None
                     else cfg.get("eval_games", DEFAULT_EVAL_GAMES))
    eval_sims = int(eval_sims if eval_sims is not None
                    else cfg.get("eval_sims", DEFAULT_EVAL_SIMS))
    eval_interval_steps = int(
        eval_interval_steps if eval_interval_steps is not None
        else cfg.get("eval_interval_steps", 2000))
    eval_max_moves = int(
        eval_max_moves if eval_max_moves is not None
        else cfg.get("eval_max_moves", DEFAULT_EVAL_MAX_MOVES))
    grad_clip = float(grad_clip) if grad_clip is not None else float(DEFAULT_GRAD_CLIP)
    viz_enabled = bool(cfg.get("viz_enabled", True)) if viz_enabled is None \
        else bool(viz_enabled)
    if cycles is None:
        cycles = 1 if steps_budget is None else 10 ** 6
    cycles = int(cycles)
    lr_base = float(cfg.get("lr", 0.2))
    schedule_steps = tuple(int(s) for s in cfg.get("lr_schedule_steps",
                                                   [50000, 100000]))
    keep_games = int(cfg.get("replay_buffer_games", 1000))

    torch.manual_seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    latest = latest_checkpoint_path(checkpoint_dir)
    best = best_checkpoint_path(checkpoint_dir)

    st = _load_or_init(cfg, latest, device, int(seed), bool(resume),
                       init_checkpoint=init_checkpoint)
    model, optimizer, rng = st["model"], st["optimizer"], st["rng"]
    global_step = int(st["global_step"])
    games_generated = int(st["games_generated"])
    steps_into_cycle = int(st["steps_into_cycle"])
    cycles_completed = int(st["cycles_completed"])
    resumed = bool(st["resumed"])

    # P13: human_mix / selfplay_workers / pretrain_data_dir are run-only
    # params (absent from config/default.yaml) that shape training-data
    # composition. They are persisted into the checkpoint's config so a bare
    # ``--resume`` after a machine restart restores them automatically instead
    # of silently reverting to pure-RL. Resolution order: an explicit CLI arg
    # wins; else the resumed checkpoint's recorded config; else ``cfg``; else
    # the safe default. Old checkpoints (no keys) therefore fall back to
    # human_mix=0.0 / workers=1 and still need explicit flags on resume. The
    # resolved values are written into ``persist_cfg`` -- a COPY of the
    # caller's cfg -- which is what ``_save_ckpt`` persists (the caller's dict
    # is never mutated).
    persist_cfg = dict(cfg)
    ckpt_cfg = dict(st.get("ckpt_config") or {}) if resumed else {}

    def _resolve_run_param(name, explicit, default):
        """explicit arg > resumed-checkpoint config > cfg > safe default."""
        if explicit is not None:
            return explicit, "cli"
        if name in ckpt_cfg and ckpt_cfg.get(name) is not None:
            return ckpt_cfg[name], "checkpoint"
        return cfg.get(name, default), "default"

    human_mix, hm_src = _resolve_run_param("human_mix", human_mix, 0.0)
    human_mix = float(human_mix)
    selfplay_workers, sw_src = _resolve_run_param(
        "selfplay_workers", selfplay_workers, 1)
    selfplay_workers = int(selfplay_workers)
    if selfplay_workers < 1 or selfplay_workers > MAX_SELFPLAY_WORKERS:
        raise ValueError(
            f"selfplay_workers must be 1..{MAX_SELFPLAY_WORKERS} "
            f"(6GB GPU cap), got {selfplay_workers}")
    pretrain_data_dir, pd_src = _resolve_run_param(
        "pretrain_data_dir", pretrain_data_dir, DEFAULT_PRETRAIN_DATA_DIR)
    save_every_games, seg_src = _resolve_run_param(
        "save_every_games", save_every_games, DEFAULT_SAVE_EVERY_GAMES)
    save_every_games = int(save_every_games)
    if save_every_games < 0:
        raise ValueError(f"save_every_games must be >= 0, got {save_every_games}")
    for _name, _val in (
        ("human_mix", human_mix),
        ("selfplay_workers", selfplay_workers),
        ("pretrain_data_dir", str(pretrain_data_dir)),
        ("save_every_games", save_every_games),
    ):
        persist_cfg[_name] = _val
    logger.info(
        "run config: human_mix=%s (source=%s) selfplay_workers=%s "
        "(source=%s) pretrain_data_dir=%s (source=%s) "
        "save_every_games=%s (source=%s)",
        human_mix, hm_src, selfplay_workers, sw_src,
        pretrain_data_dir, pd_src, save_every_games, seg_src)

    # P7: the board size follows the *model* (checkpoint arch wins over the
    # config) so self-play / buffer / viz / the eval gate all use the net's
    # own size -- a b20c256-19 warm-start stays on 19x19, and a 9x9 net is
    # self-played on 9x9 even if the config says 19.
    board_size = int(model.board_size)

    buffer = ReplayBuffer(data_dir, max_games=keep_games, board_size=board_size)

    # P11 human-data mixing: when enabled, each training batch blends
    # ``human_mix``-fraction human positions (sampled uniformly from the
    # data/pretrain chunk corpus) with self-play positions. The sampler draws
    # from the SAME persistent ``rng`` as the replay buffer, so deterministic
    # resume needs no new checkpoint state (the persisted ``rng_state`` covers
    # both streams). Read-only mmap handles are closed when the run ends.
    human_sampler = None
    human_chunks = None
    mix = float(human_mix)
    if mix > 0.0:
        from omigamax.train.pretrain import PretrainChunks, make_human_sampler
        human_chunks = PretrainChunks(pretrain_data_dir)
        if int(human_chunks.board_size) != int(board_size):
            human_chunks.close()
            raise ValueError(
                f"human-mix chunk corpus is {human_chunks.board_size} but the "
                f"model is {board_size} -- cannot mix boards of different sizes"
            )
        human_sampler = make_human_sampler(human_chunks)
        logger.info(
            "human-mix enabled: mix=%.3f corpus=%s (board %dx%d)",
            mix, human_chunks.data_dir, human_chunks.board_size,
            human_chunks.board_size,
        )

    current_elo = read_last_elo(history)
    viz = start_viz_if_available(
        {"viz_enabled": viz_enabled,
         "viz_queue_size": int(cfg.get("viz_queue_size", 32))},
        logger,
    )
    logger.info(
        "loop start: resume=%s global_step=%d games_generated=%d "
        "steps_into_cycle=%d viz=%s",
        resumed, global_step, games_generated, steps_into_cycle, viz["reason"])

    # F3c: the window pops up within seconds of launch -- push an empty-board
    # opening frame right after the viz thread starts (before any game has
    # been generated), so the user sees the board immediately instead of
    # waiting ~25 min for the first self-play frame. Non-blocking and
    # try/except-wrapped inside push_opening_frame: viz can never slow or
    # crash training.
    push_opening_frame(viz, board_size, komi=float(cfg.get("komi", 7.5)))

    # run bookkeeping
    interrupted = False
    cycles_done = 0
    total_steps = 0
    loss_first = None
    loss_last = None
    eval_reports: list[dict] = []
    last_eval_step = -1
    remaining_budget = None if steps_budget is None else int(steps_budget)

    # P16-7: save-every-games mid-selfplay checkpoint snapshots.
    # ``cycle_base`` is the value of ``games_generated`` when the CURRENT
    # cycle's self-play started; mid-cycle saves persist ``games_generated``
    # as this snapshot (never the advanced in-memory counter) so a resume
    # cannot double-count the in-flight cycle's games. ``games_this_cycle``
    # counts completed games in the current cycle (restored from the
    # checkpoint extra on resume) and is cleared ONLY when the cycle
    # completes. ``_save_lock`` serializes the workers>1 drain-thread save
    # against a Ctrl+C interrupt save so two concurrent writers can never
    # corrupt ``latest.pt.tmp``.
    cycle_base = int(games_generated)
    games_this_cycle = int(st.get("games_this_cycle", 0))
    _save_lock = threading.Lock()

    def _save_ckpt() -> str:
        # A checkpoint at a completed cycle persists steps_into_cycle=0 so a
        # later resume starts a fresh cycle; a mid-cycle checkpoint persists
        # the in-flight step so --resume continues inside the same cycle.
        cycle_complete = steps_into_cycle >= steps_per_cycle
        persist_into = 0 if cycle_complete else int(steps_into_cycle)
        # Completed cycles = those loaded from the checkpoint + those finished
        # during THIS run + the in-flight cycle when it just completed. This
        # stays correct even when the final save runs after the in-memory
        # ``steps_into_cycle`` has already been reset to 0.
        persist_cycles = (
            int(cycles_completed) + int(cycles_done)
            + (1 if cycle_complete else 0)
        )
        # P16-7 games state. ``games_this_cycle`` is reset to 0 ONLY when the
        # cycle is complete (mid-cycle eval-gate saves never clear it). While
        # the cycle's self-play is still in flight (games_this_cycle below
        # games_per_cycle) ``games_generated`` is persisted as the CYCLE BASE
        # snapshot; once the cycle's games are all generated (self-play done,
        # training in flight) the true counter is persisted, preserving the
        # pre-P16-7 mid-cycle checkpoint contract (test_loop.py asserts it).
        persist_games_this_cycle = (
            0 if cycle_complete else int(games_this_cycle)
        )
        if cycle_complete:
            persist_games = int(games_generated)
        elif int(games_this_cycle) < int(games_per_cycle):
            persist_games = int(cycle_base)
        else:
            # self-play done (games_this_cycle == games_per_cycle): persist the
            # TRUE counter. The workers>1 drain-thread save for the final game
            # can land before the main thread advances ``games_generated`` by
            # games_per_cycle, so compute it from the cycle base instead.
            persist_games = int(cycle_base) + int(games_this_cycle)
        with _save_lock:
            return save_checkpoint(
                latest, model, optimizer, global_step=int(global_step),
                config=persist_cfg, rng=rng,
                extra={
                    "games_generated": persist_games,
                    "steps_into_cycle": persist_into,
                    "cycles_completed": persist_cycles,
                    "games_this_cycle": persist_games_this_cycle,
                },
            )

    def _eval_now() -> dict:
        nonlocal current_elo, last_eval_step
        _save_ckpt()  # the gate reads the candidate from latest.pt on disk
        ev = _run_eval_gate(model, optimizer, cfg, latest=latest, best=best,
                            history=history, device=device, games=eval_games,
                            sims=eval_sims,
                            seed=int(seed) + int(cycles_done),
                            max_moves=eval_max_moves,
                            board_size=board_size)
        current_elo = float(ev.get("elo_update", {}).get("elo", current_elo))
        last_eval_step = int(global_step)
        eval_reports.append(ev)
        _log_loop_event(train_log, "eval_gate", step=int(global_step),
                        cycle=int(cycles_done) + 1,
                        winrate=float(ev.get("match", {}).get(
                            "winrate", ev.get("winrate", 0.0))),
                        replaced=bool(ev.get("replaced_best",
                                             ev.get("replaced", False))),
                        elo=round(current_elo, 3))
        return ev

    if resumed and steps_into_cycle > 0:
        _log_loop_event(train_log, "loop_resume", step=global_step,
                        games_generated=games_generated,
                        steps_into_cycle=steps_into_cycle)

    def _per_move_frame(board, move_number, color) -> None:
        """F3d: stream EVERY self-play move to the live window in real time.

        Passed as ``frame_callback`` to :func:`generate_games`, which invokes
        it with the live rules ``Board`` right after each stone / pass
        placement. The Snapshot is built directly from that LIVE board (a new
        helper :func:`viz_snapshot_from_board` -- the npz only exists at game
        end, so per-move frames cannot use the npz path) and pushed through
        the existing non-blocking queue. Started-guarded + try/except-wrapped:
        viz can never slow or crash training. Cost is one ``board.state`` list
        copy + a dataclass construction + a bounded-queue ``put_nowait`` --
        sub-millisecond, no encoding/render work in the trainer thread.
        """
        if not viz.get("started"):
            return
        next_player = BLACK if len(board.moves) % 2 == 0 else WHITE
        last_move = None
        if board.moves:
            mv, _ = board.moves[-1]
            if mv is not None:
                last_move = (int(mv[0]), int(mv[1]))
        snap = viz_snapshot_from_board(
            board, board_size, int(move_number), next_player, last_move,
            komi=float(cfg.get("komi", 7.5)), games=games_generated,
            train_step=None, loss=None, elo=current_elo)
        if snap is not None:
            push_viz_frame(viz, snap)

    def _on_game_progress(_count: int) -> None:
        """P14: one buffer-refresh frame per completed game for workers>1.

        Passed as ``progress_callback`` to the batched ``generate_games`` call
        (workers>1), which invokes it from its drain thread each time a worker
        game's npz lands on disk -- mirroring the workers==1 per-game path
        (refresh + push_selfplay_frame) so the window advances per finished
        game instead of staying frozen until the whole batch lands.

        P16-7: this is also the workers>1 mid-selfplay save hook -- the
        completed-game count in the current cycle advances here and a
        checkpoint is written every ``save_every_games`` games (the same
        cadence the workers==1 per-game loop uses). The drain thread is the
        ONLY thread executing while ``generate_games`` runs (the main thread
        is blocked in ``join()``), and ``_save_ckpt`` takes ``_save_lock`` so
        a Ctrl+C interrupt save in the main thread can never race the file.

        Thread-safety invariant: while ``generate_games`` runs, the main loop
        thread is blocked in ``join()`` and only the drain thread executes, so
        ``buffer.refresh()`` here never races a concurrent ``sample()``. The
        whole callback is try/except-wrapped inside ``generate_games``; the
        batched call's own post-batch refresh below is kept (idempotent), so a
        swallowed callback failure cannot leave the window permanently stale.
        ``_count`` is informational -- the frame shows ``buffer.num_games``.
        """
        nonlocal games_this_cycle
        games_this_cycle += 1
        if save_every_games > 0 and games_this_cycle % save_every_games == 0:
            _save_ckpt()
        buffer.refresh()
        if viz.get("started"):
            push_selfplay_frame(
                viz, buffer, board_size,
                komi=float(cfg.get("komi", 7.5)),
                games=buffer.num_games, train_step=None, elo=current_elo)

    try:
        while cycles_done < cycles and (
            remaining_budget is None or remaining_budget > 0
        ):
            cycle_no = int(cycles_done) + 1
            if steps_into_cycle == 0:
                # start of a cycle: self-play first (model unchanged since the
                # last checkpoint, so interrupted partial games regenerate
                # deterministically from the same seeds). F3c: generate one
                # game at a time and push a viz frame after EACH one, so the
                # user watches each game finish (~3-6 min apart) instead of
                # waiting for the whole batch. Seeds stay continuous: every
                # call passes seed=int(seed)+games_generated and increments
                # games_generated, so the npz set / buffer contents are
                # identical to one generate_games(games=N) batch call.
                #
                # P16-7: ``cycle_base`` snapshots games_generated at cycle
                # start; ``games_this_cycle`` (restored on resume) counts games
                # completed in the current cycle. When a mid-selfplay
                # checkpoint was saved (games_this_cycle > 0) we generate only
                # the REMAINING games, seeded from cycle_base + games_this_cycle
                # -- completed games are never regenerated. After the block the
                # full cycle's total is restored so the counter advances by a
                # whole cycle regardless of how many games were resumed.
                cycle_base = int(games_generated)
                remaining_games = int(games_per_cycle) - int(games_this_cycle)
                seed_offset = int(games_generated) + int(games_this_cycle)
                rep = {"games": 0, "sims": 0, "wall_time_s": 0.0,
                       "sims_per_sec": 0.0}
                if remaining_games > 0:
                    if selfplay_workers > 1:
                        # P12: multi-process batch. N worker processes generate
                        # the whole cycle's games in parallel -- each worker
                        # holds its own model copy + evaluator + MCTS, so the
                        # serial MCTS selection phases run on separate CPU cores
                        # while GPU forwards from different workers overlap
                        # (raise avg GPU util and sims/s on the idle-phase-bound
                        # loop). Seeds stay continuous (seed + seed_offset ..
                        # +games_per_cycle-1) so the npz set is identical to
                        # the single-process batch. Viz limitation: per-move
                        # frames are NOT streamed from worker processes (each
                        # worker has its own private board); instead
                        # generate_games fires _on_game_progress once per
                        # completed worker game (buffer-refresh +
                        # push_selfplay_frame + the P16-7 every-N save), so the
                        # window advances per finished game, plus one more
                        # buffer-refresh frame after the batch lands.
                        r, _records = generate_games(
                            model, cfg, games=remaining_games, data_dir=data_dir,
                            keep=keep_games, seed=int(seed) + seed_offset,
                            simulations=simulations, max_moves=selfplay_max_moves,
                            size=board_size, leaf_batch=leaf_batch,
                            fp16=selfplay_fp16, workers=selfplay_workers,
                            progress_callback=_on_game_progress)
                        buffer.refresh()
                        if viz.get("started"):
                            push_selfplay_frame(
                                viz, buffer, board_size,
                                komi=float(cfg.get("komi", 7.5)),
                                games=buffer.num_games, train_step=None,
                                elo=current_elo)
                        rep["games"] += int(r.get("games", 1))
                        rep["sims"] += int(r.get("sims", 0))
                        rep["wall_time_s"] += float(r.get("wall_time_s", 0.0))
                    else:
                        for i in range(remaining_games):
                            r, _records = generate_games(
                                model, cfg, games=1, data_dir=data_dir,
                                keep=keep_games,
                                seed=int(seed) + seed_offset + i,
                                simulations=simulations,
                                max_moves=selfplay_max_moves,
                                size=board_size,
                                frame_callback=_per_move_frame,
                                leaf_batch=leaf_batch, fp16=selfplay_fp16)
                            games_generated += 1
                            games_this_cycle += 1
                            buffer.refresh()
                            if viz.get("started"):
                                push_selfplay_frame(
                                    viz, buffer, board_size,
                                    komi=float(cfg.get("komi", 7.5)),
                                    games=buffer.num_games, train_step=None,
                                    elo=current_elo)
                            rep["games"] += int(r.get("games", 1))
                            rep["sims"] += int(r.get("sims", 0))
                            rep["wall_time_s"] += float(
                                r.get("wall_time_s", 0.0))
                            # P16-7 workers==1 hook: write a mid-selfplay
                            # checkpoint every N completed games of the cycle
                            # (progress_callback is ignored for workers==1, so
                            # the per-game loop owns the cadence here).
                            if save_every_games > 0 and \
                                    games_this_cycle % save_every_games == 0:
                                _save_ckpt()
                else:
                    # resume right after a completed self-play phase: the
                    # cycle's games are already on disk, just refresh.
                    buffer.refresh()
                # the whole cycle's games are now accounted for (whether fresh,
                # continued, or already on disk): advance by the FULL cycle so
                # games_generated never falls behind / double counts on resume.
                games_generated = int(cycle_base) + int(games_per_cycle)
                rep["sims_per_sec"] = (
                    rep["sims"] / rep["wall_time_s"]
                    if rep["wall_time_s"] > 0 else 0.0)
                logger.info(
                    "cycle %d: generated %d self-play games (step %d, "
                    "%.1f sims/s)", cycle_no, rep["games"], global_step,
                    float(rep["sims_per_sec"]))
                _log_loop_event(train_log, "cycle_start", cycle=cycle_no,
                                step=global_step, games=rep["games"],
                                games_generated=games_generated)
            else:
                # mid-cycle resume: the games are already on disk from before
                # the interrupt; just refresh the window. Defensive fallback:
                # if the data dir was wiped meanwhile, regenerate determin-
                # istically (same model, same seeds) -- again one game at a
                # time with a frame pushed after each (F3c).
                buffer.refresh()
                if buffer.num_games == 0:
                    # Regenerate deterministically (same model, same seeds).
                    # With workers>1 the whole cycle batch goes out at once
                    # (same seed continuity as the start-of-cycle path).
                    # P16-7: games_this_cycle tracks the regenerated games
                    # (the _on_game_progress drain hook / the per-game loop
                    # counter below) so mid-regeneration saves stay consistent.
                    if selfplay_workers > 1:
                        generate_games(
                            model, cfg, games=games_per_cycle, data_dir=data_dir,
                            keep=keep_games, seed=int(seed) + games_generated,
                            simulations=simulations,
                            max_moves=selfplay_max_moves,
                            size=board_size, leaf_batch=leaf_batch,
                            fp16=selfplay_fp16, workers=selfplay_workers,
                            progress_callback=_on_game_progress)
                        games_generated += games_per_cycle
                    else:
                        for _ in range(games_per_cycle):
                            generate_games(
                                model, cfg, games=1, data_dir=data_dir,
                                keep=keep_games, seed=int(seed) + games_generated,
                                simulations=simulations,
                                max_moves=selfplay_max_moves,
                                size=board_size, frame_callback=_per_move_frame,
                                leaf_batch=leaf_batch, fp16=selfplay_fp16)
                            games_generated += 1
                            games_this_cycle += 1
                            if save_every_games > 0 and \
                                    games_this_cycle % save_every_games == 0:
                                _save_ckpt()
                    buffer.refresh()
                    if viz.get("started"):
                        push_selfplay_frame(
                            viz, buffer, board_size,
                            komi=float(cfg.get("komi", 7.5)),
                            games=buffer.num_games, train_step=None,
                            elo=current_elo)
                elif viz.get("started"):
                    # games already on disk: push the newest one so the window
                    # shows a real board right away on resume.
                    push_selfplay_frame(
                        viz, buffer, board_size,
                        komi=float(cfg.get("komi", 7.5)),
                        games=buffer.num_games, train_step=None,
                        elo=current_elo)
                logger.info(
                    "cycle %d: resuming mid-cycle (step %d, %d/%d steps done)",
                    cycle_no, global_step, steps_into_cycle, steps_per_cycle)

            remaining_in_cycle = steps_per_cycle - steps_into_cycle
            if remaining_budget is not None:
                n_train = min(remaining_in_cycle, remaining_budget)
            else:
                n_train = remaining_in_cycle
            n_train = max(0, int(n_train))
            if n_train <= 0:
                break

            # One board frame per cycle (rebuilt from the newest self-play
            # position); every train step below reuses it and only refreshes
            # the metrics -- no per-step disk I/O in the viz path.
            viz_board = viz_board_info(buffer, board_size)
            viz_komi = float(cfg.get("komi", 7.5))

            for _ in range(n_train):
                losses, lrs, global_step, rng = train_steps(
                    model, optimizer, buffer, steps=1, rng=rng, seed=int(seed),
                    global_step=global_step, batch_size=batch_size,
                    device=device, use_fp16=use_fp16, grad_clip=grad_clip,
                    symmetry=use_symmetry, lr_base=lr_base,
                    schedule_steps=schedule_steps,
                    human_sampler=human_sampler, human_mix=mix)
                loss = float(losses[0])
                lr = float(lrs[0])
                if loss_first is None:
                    loss_first = loss
                loss_last = loss
                total_steps += 1
                steps_into_cycle += 1
                _log_train_step(train_log, step=global_step, loss=loss, lr=lr,
                                games=buffer.num_games, elo=current_elo,
                                cycle=cycle_no)
                # F2 MAJOR 2: feed the live window a frame per train step
                # (non-blocking, drop-oldest, wrapped -- viz can never slow or
                # crash training).
                if viz.get("started"):
                    push_viz_frame(viz, build_viz_snapshot(
                        viz_board, komi=viz_komi, games=buffer.num_games,
                        train_step=global_step, loss=loss, elo=current_elo,
                    ))
                if interrupt_after is not None and total_steps >= int(interrupt_after):
                    raise KeyboardInterrupt(
                        f"simulated Ctrl+C after {total_steps} training steps "
                        f"(--interrupt-at-steps)")
                if eval_due(global_step,
                            cycle_end=(steps_into_cycle >= steps_per_cycle),
                            eval_interval_steps=eval_interval_steps):
                    _eval_now()

            if steps_into_cycle >= steps_per_cycle:
                steps_into_cycle = 0
                cycles_done += 1
                # P16-7: cycle complete -> the cycle's games are fully
                # accounted for; clear the per-cycle counter and advance the
                # base so a subsequent save (incl. the final one) persists the
                # true totals instead of a stale mid-cycle snapshot.
                games_this_cycle = 0
                cycle_base = int(games_generated)
            if remaining_budget is not None:
                remaining_budget -= n_train
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        if steps_into_cycle >= steps_per_cycle:
            steps_into_cycle = 0
            cycles_done += 1
            games_this_cycle = 0
            cycle_base = int(games_generated)
        _save_ckpt()
        _log_loop_event(train_log, "interrupt", step=global_step,
                        total_steps=total_steps, cycles_completed=cycles_done,
                        games_generated=games_generated)
        logger.warning("interrupted: checkpoint saved at step %d (%s)",
                       global_step, latest)
    else:
        if force_final_eval and global_step > last_eval_step:
            _eval_now()
        _save_ckpt()
        if steps_into_cycle >= steps_per_cycle:
            steps_into_cycle = 0
            cycles_done += 1
            games_this_cycle = 0
            cycle_base = int(games_generated)
        _log_loop_event(train_log, "loop_end", step=global_step,
                        total_steps=total_steps, cycles_completed=cycles_done,
                        games_generated=games_generated,
                        elo=round(current_elo, 3))
        logger.info(
            "loop end: %d steps trained this run (global step %d, %d cycles, "
            "final elo %.3f)", total_steps, global_step, cycles_done,
            current_elo)

    # tidy up the todo-17 viz thread (daemon anyway, but a clean stop prevents
    # leaked windows/threads on every run); never breaks the loop itself.
    stop_viz = viz.get("stop")
    if stop_viz is not None:
        try:
            stop_viz()
            viz.get("thread").join(timeout=2)
        except Exception:  # pragma: no cover - defensive
            logger.warning("viz stop raised (ignored)", exc_info=True)

    # P11: release the human-chunk mmap handles (idempotent; only open when
    # human-mix was enabled).
    if human_chunks is not None:
        try:
            human_chunks.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("human-chunk close raised (ignored)", exc_info=True)

    return {
        "todo": 16,
        "device": str(device),
        "protocol": {
            "cycles": int(cycles),
            "games_per_cycle": games_per_cycle,
            "steps_per_cycle": steps_per_cycle,
            "steps_budget": steps_budget,
            "simulations": simulations,
            "selfplay_max_moves": selfplay_max_moves,
            "batch_size": batch_size,
            "eval_games": eval_games,
            "eval_sims": eval_sims,
            "eval_interval_steps": eval_interval_steps,
            "eval_max_moves": eval_max_moves,
            "replace_threshold": float(
                replace_threshold if replace_threshold is not None
                else cfg.get("replace_threshold", 0.55)),
            "symmetry_aug": use_symmetry,
            "fp16": use_fp16,
            "human_mix": mix,
            "pretrain_data_dir": str(pretrain_data_dir),
            "leaf_batch": (
                int(leaf_batch) if leaf_batch is not None
                else int(cfg.get("leaf_batch", 16))),
            "selfplay_fp16": bool(selfplay_fp16),
            "selfplay_workers": int(selfplay_workers),
            "save_every_games": int(save_every_games),
            "grad_clip": grad_clip,
            "seed": int(seed),
            "resume": bool(resume),
            "resumed": resumed,
            "init_checkpoint": (
                None if init_checkpoint is None else str(init_checkpoint)
            ),
            "board_size": board_size,
            "force_final_eval": force_final_eval,
            "viz_enabled": viz_enabled,
            "viz": viz,
            "lr": lr_base,
            "lr_schedule_steps": list(schedule_steps),
            "data_dir": str(data_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "train_log": str(train_log),
            "history": str(history),
        },
        "loop": {
            "interrupted": interrupted,
            "resumed": resumed,
            "steps_trained": total_steps,
            "global_step_final": global_step,
            "games_generated": games_generated,
            "cycles_done": cycles_done,
            "loss_first": loss_first,
            "loss_last": loss_last,
            "loss_decrease": (
                loss_last < loss_first if loss_first is not None
                and loss_last is not None else None
            ),
            "eval_gates": len(eval_reports),
            "eval_summaries": [_ev_summary(ev) for ev in eval_reports],
        },
        "checkpoint": {
            "latest": str(latest),
            "latest_exists": latest.exists(),
            "best": str(best),
            "best_exists": best.exists(),
        },
        "accepted": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="omigamax todo-16 training loop: self-play -> train -> "
                    "eval gate, interruptible and resumable. Per-cycle "
                    "rhythm: games -> train steps -> gate; Ctrl+C/SIGBREAK "
                    "checkpoint and exit; --resume continues."
    )
    parser.add_argument("--smoke", action="store_true",
                        help="low-config preset (sims=40, batch=32) that "
                             "runs one full cycle with a forced final "
                             "evaluation gate")
    parser.add_argument("--cycles", type=int, default=None,
                        help="number of full cycles (default 1; ignored when "
                             "--steps is given)")
    parser.add_argument("--games", type=int, default=None,
                        help="self-play games per cycle (default 100)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="training steps per cycle (default 1000)")
    parser.add_argument("--steps", type=int, default=None,
                        help="total training steps to run this invocation "
                             "(overrides --cycles when given)")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per self-play move "
                             "(default: config simulations=200)")
    parser.add_argument("--selfplay-max-moves", type=int, default=None,
                        help="move cap per self-play game (default: "
                             "config selfplay_max_moves=1000; --smoke bounds "
                             "it to 150 so generation completes in minutes)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="training batch size (default: config 128)")
    parser.add_argument("--eval-games", type=int, default=None,
                        help="evaluation games per gate (default: config 21)")
    parser.add_argument("--eval-sims", type=int, default=None,
                        help="MCTS simulations per eval move "
                             "(default: config eval_sims=200)")
    parser.add_argument("--eval-interval-steps", type=int, default=None,
                        help="sparser eval interval for long training "
                             "(default: config eval_interval_steps=2000; "
                             "cycle-end gates always fire)")
    parser.add_argument("--eval-max-moves", type=int, default=None,
                        help="move cap per evaluation game (default: config "
                             "eval_max_moves=1000; --smoke bounds it to 150 "
                             "so the gate completes in minutes)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from models/latest.pt + data/selfplay "
                             "(requires an existing checkpoint)")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="start a FRESH RL run warm-started from this "
                             "checkpoint (e.g. models/pretrain.pt): its "
                             "recorded arch (blocks/channels/board_size) wins "
                             "over the config. Ignored when --resume finds "
                             "models/latest.pt.")
    parser.add_argument("--blocks", type=int, default=None,
                        help="override config blocks for a fresh-init run "
                             "(config default 10; checkpoint arch wins when "
                             "--resume/--init-checkpoint loads one)")
    parser.add_argument("--channels", type=int, default=None,
                        help="override config channels for a fresh-init run "
                             "(config default 128)")
    parser.add_argument("--board-size", type=int, default=None,
                        help="override config board_size for a fresh-init "
                             "run (config default 19)")
    parser.add_argument("--interrupt-at-steps", type=int, default=None,
                        help="simulate Ctrl+C after this many training steps "
                             "(exercises the graceful checkpoint path)")
    parser.add_argument("--viz", choices=["on", "off"], default=None,
                        help="visualization mount (default: config "
                             "viz_enabled=true; gracefully degraded while "
                             "todo 17 is absent)")
    parser.add_argument("--no-symmetry", action="store_true",
                        help="disable the 8-fold symmetry augmentation")
    parser.add_argument("--fp16", action="store_true",
                        help="exercise the FP16 (autocast) toggle")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help=f"gradient-norm clip (default {DEFAULT_GRAD_CLIP})")
    parser.add_argument("--human-mix", type=float, default=None,
                        help="fraction of each training batch drawn from the "
                             "human data/pretrain chunk corpus (KataGo-style "
                             "human-data mixing; default: auto -- restored "
                             "from the checkpoint on --resume, else 0.0 = "
                             "pure self-play, byte-identical to the pre-P11 "
                             "behavior)")
    parser.add_argument("--pretrain-data-dir", type=str, default=None,
                        help=f"human chunk corpus dir for --human-mix "
                             f"(default: auto -- restored from the checkpoint "
                             f"on --resume, else {DEFAULT_PRETRAIN_DATA_DIR}; "
                             f"read-only)")
    parser.add_argument("--leaf-batch", type=int, default=None,
                        help="self-play leaves per network forward (default: "
                             "config leaf_batch=16; P11 speedup knob)")
    parser.add_argument("--selfplay-fp16", action="store_true",
                        help="run self-play leaf inference under fp16 "
                             "autocast on CUDA (move numerics may shift "
                             "slightly but games stay legal)")
    parser.add_argument("--selfplay-workers", type=int, default=None,
                        help="parallel self-play worker processes (default: "
                             "auto -- restored from the checkpoint on "
                             "--resume, else 1 = today's per-game loop; 2-3 "
                             "spawn child processes that each hold their own "
                             "model copy + MCTS so CPU cores parallelize the "
                             "serial search phases and GPU forwards overlap "
                             "-- per-move live viz is disabled; capped at 3 "
                             "on the 6GB GPU, ~1.4GB fp16 per worker)")
    parser.add_argument("--save-every-games", type=int, default=None,
                        help="write a mid-selfplay checkpoint every N "
                             "completed games of a cycle (default 10; 0 "
                             "disables). The checkpoint records the cycle "
                             "base + games_this_cycle so a later --resume "
                             "continues generating only the remaining games "
                             "instead of redoing the completed ones. "
                             "Resolved on --resume: CLI arg > checkpoint "
                             "config > default.")
    parser.add_argument("--seed", type=int, default=0,
                        help="master random seed (default 0)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"self-play npz directory (default {DEFAULT_DATA_DIR})")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=DEFAULT_CHECKPOINT_DIR,
                        help=f"checkpoint dir (default {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--train-log", type=str, default=DEFAULT_TRAIN_LOG,
                        help=f"per-step JSONL log (default {DEFAULT_TRAIN_LOG})")
    parser.add_argument("--history", type=str, default=DEFAULT_HISTORY,
                        help=f"eval-history JSONL (default {DEFAULT_HISTORY})")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--evidence", type=str, default=DEFAULT_EVIDENCE,
                        help=f"write the result JSON here (default {DEFAULT_EVIDENCE})")
    return parser


def _install_sigbreak() -> bool:
    """Route Windows SIGBREAK (Ctrl+Break) through the KeyboardInterrupt path.

    The plan's Windows interrupt note (Oracle): Ctrl+C raises KeyboardInterrupt
    in-process (already handled by ``run_loop``); SIGBREAK must be caught too.
    """
    if not hasattr(signal, "SIGBREAK"):
        return False
    try:
        def _raise(_signum, _frame):
            raise KeyboardInterrupt("SIGBREAK (Ctrl+Break) received")
        signal.signal(signal.SIGBREAK, _raise)
        return True
    except (ValueError, OSError, TypeError):  # non-main thread / OS limits
        return False


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _install_sigbreak()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = dict(cfg)          # never mutate the shared config dict
        cfg.update(SMOKE_PRESET)
    cfg = apply_arch_overrides(
        cfg,
        blocks=args.blocks, channels=args.channels, board_size=args.board_size,
    )

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    viz_enabled = None if args.viz is None else (args.viz == "on")

    try:
        report = run_loop(
            cfg,
            device=device,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            train_log=args.train_log,
            history=args.history,
            cycles=args.cycles,
            games_per_cycle=args.games,
            steps_per_cycle=args.train_steps,
            steps_budget=args.steps,
            simulations=args.simulations,
            selfplay_max_moves=args.selfplay_max_moves,
            batch_size=args.batch_size,
            eval_games=args.eval_games,
            eval_sims=args.eval_sims,
            eval_interval_steps=args.eval_interval_steps,
            eval_max_moves=args.eval_max_moves,
            use_symmetry=not args.no_symmetry,
            use_fp16=args.fp16,
            grad_clip=args.grad_clip,
            seed=args.seed,
            resume=args.resume,
            init_checkpoint=args.init_checkpoint,
            force_final_eval=bool(args.smoke),
            viz_enabled=viz_enabled,
            interrupt_after=args.interrupt_at_steps,
            human_mix=args.human_mix,
            pretrain_data_dir=args.pretrain_data_dir,
            leaf_batch=args.leaf_batch,
            selfplay_fp16=args.selfplay_fp16,
            selfplay_workers=args.selfplay_workers,
            save_every_games=args.save_every_games,
        )
    except KeyboardInterrupt:
        # Interrupt landed outside run_loop's protected region (model load,
        # buffer scan, ...); no checkpoint was written this time.
        print("interrupted during setup -- no checkpoint was written; "
              "rerun with --resume to continue from the last checkpoint",
              flush=True)
        return 0

    _print_report(report)

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON-safe report for the evidence dump: the viz handle carries live
        # queue/thread/stop objects that are NOT serializable and would crash
        # the dump after a successful run (plan: viz must never break training).
        def _json_safe(value):
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(type(value).__name__)  # queue/thread/stop -> type name

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0


def _print_report(result: dict) -> None:
    proto = result["protocol"]
    loop_ = result["loop"]
    ck = result["checkpoint"]
    print("=== omigamax training loop (todo 16) ===", flush=True)
    print(f"device: {result['device']}", flush=True)
    print(
        f"protocol: cycles={proto['cycles']} games/cycle="
        f"{proto['games_per_cycle']} steps/cycle={proto['steps_per_cycle']} "
        f"sims/move={proto['simulations']} batch={proto['batch_size']} "
        f"eval_games={proto['eval_games']} eval_sims={proto['eval_sims']} "
        f"eval_interval={proto['eval_interval_steps']} seed={proto['seed']} "
        f"viz={proto['viz']['reason']}", flush=True
    )
    print(
        f"resume: requested={proto['resume']} effective={loop_['resumed']}",
        flush=True,
    )
    print(
        f"loop: {loop_['steps_trained']} steps trained this run "
        f"(global step {loop_['global_step_final']}, "
        f"{loop_['cycles_done']} cycles done, "
        f"{loop_['games_generated']} games generated)", flush=True
    )
    if loop_["interrupted"]:
        print("loop: INTERRUPTED -- checkpoint saved, logs flushed", flush=True)
    if loop_["loss_first"] is not None:
        print(
            f"loss: {loop_['loss_first']:.4f} -> {loop_['loss_last']:.4f} "
            f"(decrease={loop_['loss_decrease']})", flush=True
        )
    for i, s in enumerate(loop_["eval_summaries"]):
        decision = "REPLACE best.pt" if s["replaced"] else "KEEP best.pt"
        print(
            f"eval gate {i + 1} (step {s['step']}): "
            f"{s['candidate_wins']}/{s['games']} wins "
            f"(win rate {s['winrate']:.3f}) elo={s['elo']:.3f} -> {decision}",
            flush=True,
        )
    print(
        f"checkpoints: latest={ck['latest']} exists={ck['latest_exists']} | "
        f"best={ck['best']} exists={ck['best_exists']}", flush=True
    )
    print(f"train log: {proto['train_log']}", flush=True)
    print(f"eval history: {proto['history']}", flush=True)
    print("RESULT: PASS (exit 0)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
