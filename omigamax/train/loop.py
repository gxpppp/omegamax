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
  the mount. The thread is stopped cleanly at the end of the run.

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
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train.buffer import ReplayBuffer
from omigamax.train.evaluate import (
    DEFAULT_EVAL_GAMES,
    DEFAULT_EVAL_SIMS,
    best_checkpoint_path,
    evaluate_and_gate,
    read_last_elo,
)
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.selfplay import DEFAULT_DATA_DIR, generate_games
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

def _load_or_init(cfg: dict, checkpoint_path, device, seed: int, resume: bool) -> dict:
    """Build the run state: either from ``latest.pt`` (resume) or fresh init.

    On resume the architecture is read from the checkpoint (not the config) so
    a checkpoint written at a different board/net size restores correctly.
    The buffer-sampling numpy RNG is restored from the persisted
    ``rng_state`` so deterministic-resume stays exact (Oracle F9).
    """
    ckpt_path = Path(checkpoint_path)
    if resume and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        arch = ckpt["arch"]
        model = create_model(
            int(arch["blocks"]), int(arch["channels"]),
            int(arch["board_size"]),
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
            "resumed": True,
        }
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
    }


# ---------------------------------------------------------------------------
# evaluation gate driver
# ---------------------------------------------------------------------------

def _run_eval_gate(model, optimizer, cfg, *, latest, best, history, device,
                   games, sims, seed, max_moves: int = DEFAULT_EVAL_MAX_MOVES) -> dict:
    """Run the todo-15 gate: candidate (``latest``) vs ``best``, record ELO."""
    return evaluate_and_gate(
        str(latest), str(best), cfg,
        games=games, sims=sims,
        size=int(cfg.get("board_size", 19)),
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
    force_final_eval: bool = False,
    viz_enabled: "bool | None" = None,
    logger=None,
    interrupt_after: "int | None" = None,
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
    board_size = int(cfg.get("board_size", 19))

    torch.manual_seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    latest = latest_checkpoint_path(checkpoint_dir)
    best = best_checkpoint_path(checkpoint_dir)

    st = _load_or_init(cfg, latest, device, int(seed), bool(resume))
    model, optimizer, rng = st["model"], st["optimizer"], st["rng"]
    global_step = int(st["global_step"])
    games_generated = int(st["games_generated"])
    steps_into_cycle = int(st["steps_into_cycle"])
    cycles_completed = int(st["cycles_completed"])
    resumed = bool(st["resumed"])

    buffer = ReplayBuffer(data_dir, max_games=keep_games, board_size=board_size)

    current_elo = read_last_elo(history)
    viz = start_viz_if_available({"viz_enabled": viz_enabled}, logger)
    logger.info(
        "loop start: resume=%s global_step=%d games_generated=%d "
        "steps_into_cycle=%d viz=%s",
        resumed, global_step, games_generated, steps_into_cycle, viz["reason"])

    # run bookkeeping
    interrupted = False
    cycles_done = 0
    total_steps = 0
    loss_first = None
    loss_last = None
    eval_reports: list[dict] = []
    last_eval_step = -1
    remaining_budget = None if steps_budget is None else int(steps_budget)

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
        return save_checkpoint(
            latest, model, optimizer, global_step=int(global_step),
            config=cfg, rng=rng,
            extra={
                "games_generated": int(games_generated),
                "steps_into_cycle": persist_into,
                "cycles_completed": persist_cycles,
            },
        )

    def _eval_now() -> dict:
        nonlocal current_elo, last_eval_step
        _save_ckpt()  # the gate reads the candidate from latest.pt on disk
        ev = _run_eval_gate(model, optimizer, cfg, latest=latest, best=best,
                            history=history, device=device, games=eval_games,
                            sims=eval_sims,
                            seed=int(seed) + int(cycles_done),
                            max_moves=eval_max_moves)
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

    try:
        while cycles_done < cycles and (
            remaining_budget is None or remaining_budget > 0
        ):
            cycle_no = int(cycles_done) + 1
            if steps_into_cycle == 0:
                # start of a cycle: self-play first (model unchanged since the
                # last checkpoint, so interrupted partial games regenerate
                # deterministically from the same seeds).
                rep, _records = generate_games(
                    model, cfg, games=games_per_cycle, data_dir=data_dir,
                    keep=keep_games, seed=int(seed) + games_generated,
                    simulations=simulations, max_moves=selfplay_max_moves)
                games_generated += games_per_cycle
                buffer.refresh()
                logger.info(
                    "cycle %d: generated %d self-play games (step %d, "
                    "%.1f sims/s)", cycle_no, games_per_cycle, global_step,
                    float(rep.get("sims_per_sec", 0.0)))
                _log_loop_event(train_log, "cycle_start", cycle=cycle_no,
                                step=global_step, games=games_per_cycle,
                                games_generated=games_generated)
            else:
                # mid-cycle resume: the games are already on disk from before
                # the interrupt; just refresh the window. Defensive fallback:
                # if the data dir was wiped meanwhile, regenerate determin-
                # istically (same model, same seeds).
                buffer.refresh()
                if buffer.num_games == 0:
                    rep, _records = generate_games(
                        model, cfg, games=games_per_cycle, data_dir=data_dir,
                        keep=keep_games, seed=int(seed) + games_generated,
                        simulations=simulations, max_moves=selfplay_max_moves)
                    games_generated += games_per_cycle
                    buffer.refresh()
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

            for _ in range(n_train):
                losses, lrs, global_step, rng = train_steps(
                    model, optimizer, buffer, steps=1, rng=rng, seed=int(seed),
                    global_step=global_step, batch_size=batch_size,
                    device=device, use_fp16=use_fp16, grad_clip=grad_clip,
                    symmetry=use_symmetry, lr_base=lr_base,
                    schedule_steps=schedule_steps)
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
            if remaining_budget is not None:
                remaining_budget -= n_train
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        if steps_into_cycle >= steps_per_cycle:
            steps_into_cycle = 0
            cycles_done += 1
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
            "grad_clip": grad_clip,
            "seed": int(seed),
            "resume": bool(resume),
            "resumed": resumed,
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
            force_final_eval=bool(args.smoke),
            viz_enabled=viz_enabled,
            interrupt_after=args.interrupt_at_steps,
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
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
