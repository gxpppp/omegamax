"""P16-7: --save-every-games mid-selfplay checkpoints + resume continuation.

The loop's mid-cycle saves (every N completed games, ``save_every_games``,
default 10, 0 = disabled) write ``games_generated`` as the CYCLE BASE snapshot
plus a new ``games_this_cycle`` counter, so:

* a resume with ``games_this_cycle > 0`` and ``steps_into_cycle == 0``
  continues by generating only ``games_per_cycle - games_this_cycle`` games,
  seeded from ``seed + games_generated + games_this_cycle`` -- completed games
  are never regenerated;
* ``games_this_cycle`` is cleared ONLY when the cycle completes; mid-cycle
  eval-gate saves never clear it;
* the workers>1 drain-thread save and a Ctrl+C interrupt save are serialized
  by a lock so ``latest.pt`` stays loadable under concurrency.

Covers (all tiny 9x9 CPU fakes, temp dirs):
(a) workers=1 AND workers=2: mid-selfplay save every N -> checkpoint exists,
    games_this_cycle increments, extra games_generated == cycle_base;
(b) workers=1 AND workers=2: kill mid-selfplay -> --resume -> total distinct
    seeds per cycle == games_per_cycle (both paths), only the remaining games
    generated, training continues;
(c) cycle completion -> games_this_cycle 0 (incl. the _eval_now checkpoint);
    mid-cycle eval-gate saves do NOT clear it;
(d) concurrency: a drain-thread save racing a simulated interrupt save leaves
    latest.pt always loadable;
(e) --save-every-games 0 -> no mid-cycle saves;
(f) an old checkpoint without the key -> resume works (games_this_cycle
    defaults to 0 -> full regeneration).
"""

import threading
import time

import numpy as np
import pytest
import torch

import omigamax.train.loop as loop
import omigamax.train.train as train
from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train.loss import make_sgd_optimizer

SIZE = 9
BLOCKS = 1
CHANNELS = 8
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_loop_cfg():
    """Tiny 9x9 CPU loop config (fast cycle params)."""
    cfg = dict(load_config())
    cfg.update({"board_size": SIZE, "blocks": BLOCKS, "channels": CHANNELS,
                "simulations": 4, "eval_games": 2, "eval_sims": 4,
                "eval_interval_steps": 2000, "replace_threshold": 0.55,
                "batch_size": 8, "cycle_steps": 2, "cycle_games": 6,
                "lr": 0.1})
    return cfg


def fake_train(model, optimizer, buffer, steps, **kwargs):
    return [0.5], [0.2], int(kwargs["global_step"]) + 1, kwargs["rng"]


def fake_eval(candidate_path, best_path, cfg, **kw):
    return {"replaced_best": False,
            "match": {"winrate": 0.5, "candidate_wins": 1, "games": 2},
            "elo_update": {"elo": 3.0}}


def make_fake_gen(calls, *, kill_after=None, leave_drain_racing=False):
    """A generate_games fake that simulates the workers>1 drain callback.

    ``calls`` collects ``(games, seed, workers)`` per invocation. For
    ``workers>1`` the per-game ``progress_callback`` is fired from a separate
    drain thread (one callback per completed game), matching the real
    ``_generate_games_parallel``. With ``kill_after`` the main thread raises
    ``KeyboardInterrupt`` once ``kill_after`` games have been reported --
    simulating Ctrl+C landing mid-self-play (``steps_into_cycle == 0``). For
    ``workers==1`` the interrupt is raised when the call count reaches
    ``kill_after`` (the per-game loop calls once per remaining game).
    ``leave_drain_racing`` keeps the drain thread firing after the interrupt
    so its save races the interrupt save (concurrency test).
    """
    def fake_gen(network, cfg, games, data_dir, keep, seed, simulations, **kw):
        games = int(games)
        workers = int(kw.get("workers", 1))
        pc = kw.get("progress_callback")
        if workers > 1:
            calls.append({"games": games, "seed": int(seed), "workers": workers})
            done = [0]
            stop = threading.Event()

            def drain():
                for i in range(1, games + 1):
                    if stop.is_set():
                        return
                    time.sleep(0.001)
                    done[0] = i
                    if pc is not None:
                        try:
                            pc(i)
                        except Exception:  # noqa: BLE001 - like the real drain
                            pass

            t = threading.Thread(target=drain, daemon=True)
            t.start()
            if kill_after is not None:
                while t.is_alive() and done[0] < kill_after:
                    time.sleep(0.001)
                if not leave_drain_racing:
                    stop.set()
                raise KeyboardInterrupt(
                    f"simulated Ctrl+C after {done[0]} self-play games")
            t.join(timeout=10.0)
        else:
            # per-game call: kill_after games must COMPLETE first (the game in
            # flight when Ctrl+C lands is never counted), so the interrupt
            # fires on the call AFTER the kill_after-th one.
            if kill_after is not None and len(calls) >= kill_after:
                raise KeyboardInterrupt(
                    f"simulated Ctrl+C after {kill_after} self-play games")
            calls.append({"games": games, "seed": int(seed), "workers": workers})
        return {"games": games,
                "sims": int(games) * 30 * int(simulations),
                "wall_time_s": 1.0, "sims_per_sec": 1.0,
                "data_dir": str(data_dir)}, []

    return fake_gen


def run_loop(tmp_path, monkeypatch, calls, *, resume=False, workers=1,
             save_every_games=10, kill_after=None, leave_drain_racing=False,
             games_per_cycle=6, steps_per_cycle=2, eval_interval_steps=2000,
             cycles=1, seed=0):
    """Run the loop on fakes; return (report, captured save extras)."""
    extras: list[dict] = []
    real_save = train.save_checkpoint

    def wrapped_save(*args, **kwargs):
        if kwargs.get("extra") is not None:
            extras.append(dict(kwargs["extra"]))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(loop, "generate_games",
                        make_fake_gen(calls, kill_after=kill_after,
                                      leave_drain_racing=leave_drain_racing))
    monkeypatch.setattr(loop, "train_steps", fake_train)
    monkeypatch.setattr(loop, "evaluate_and_gate", fake_eval)
    monkeypatch.setattr(loop, "save_checkpoint", wrapped_save)

    cfg = make_loop_cfg()
    report = loop.run_loop(
        cfg, device=DEVICE,
        data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
        train_log=tmp_path / "train.jsonl",
        history=tmp_path / "eval_history.jsonl",
        cycles=cycles, games_per_cycle=games_per_cycle,
        steps_per_cycle=steps_per_cycle, eval_interval_steps=eval_interval_steps,
        selfplay_workers=workers, save_every_games=save_every_games,
        viz_enabled=False, resume=resume, seed=seed,
    )
    return report, extras


def mid_selfplay_saves(extras, games_per_cycle):
    """Saves whose games_this_cycle shows self-play still in flight."""
    return [e for e in extras
            if 0 < e["games_this_cycle"] < games_per_cycle]


# ---------------------------------------------------------------------------
# (a) mid-selfplay save every N: checkpoint exists, games_this_cycle
#     increments, extra games_generated == cycle_base
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_mid_selfplay_save_snapshots_cycle_base(tmp_path, monkeypatch, workers):
    calls = []
    report, extras = run_loop(
        tmp_path, monkeypatch, calls, workers=workers,
        save_every_games=2, games_per_cycle=6, steps_per_cycle=1)
    assert report["protocol"]["save_every_games"] == 2

    # the checkpoint really landed on disk (mid-selfplay saves + final)
    latest = tmp_path / "models" / "latest.pt"
    assert latest.exists()

    # mid-selfplay saves: one per N games, games_this_cycle increments, and
    # the persisted games_generated is the CYCLE BASE snapshot (0 here).
    mid = mid_selfplay_saves(extras, 6)
    assert [e["games_this_cycle"] for e in mid] == [2, 4]
    assert all(e["games_generated"] == 0 for e in mid)  # == cycle_base

    # when self-play finishes exactly at a multiple of N (game 6) the true
    # counter is persisted (the "self-play done" contract, test_loop.py:668).
    assert any(e["games_this_cycle"] == 6 and e["games_generated"] == 6
               for e in extras)


# ---------------------------------------------------------------------------
# (b) kill mid-selfplay -> --resume -> total distinct seeds == games_per_cycle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_resume_continues_only_remaining_games(tmp_path, monkeypatch, workers):
    calls = []
    first, _ = run_loop(
        tmp_path, monkeypatch, calls, workers=workers,
        save_every_games=2, games_per_cycle=6, steps_per_cycle=2,
        kill_after=4, seed=0)
    assert first["loop"]["interrupted"] is True
    assert first["loop"]["steps_trained"] == 0  # killed during self-play

    # the mid-selfplay checkpoint recorded the cycle-base snapshot
    ckpt = loop.load_checkpoint(tmp_path / "models" / "latest.pt")
    assert ckpt["extra"]["games_generated"] == 0          # == cycle_base
    assert ckpt["extra"]["games_this_cycle"] == 4
    assert ckpt["extra"]["steps_into_cycle"] == 0

    # resume WITHOUT flags: save_every_games restores from the checkpoint cfg
    resumed_calls = []
    resumed, extras2 = run_loop(
        tmp_path, monkeypatch, resumed_calls, resume=True, workers=workers,
        save_every_games=None, games_per_cycle=6, steps_per_cycle=2, seed=0)
    assert resumed["loop"]["resumed"] is True
    assert resumed["loop"]["interrupted"] is False
    assert resumed["protocol"]["save_every_games"] == 2  # restored from ckpt
    assert resumed["loop"]["steps_trained"] == 2         # training continued
    assert resumed["loop"]["global_step_final"] == 2

    # only the remaining 2 games were generated this session, seeded from
    # cycle_base + games_this_cycle = 4
    if workers > 1:
        assert [c["games"] for c in resumed_calls] == [2]
        assert [c["seed"] for c in resumed_calls] == [4]
    else:
        assert [c["games"] for c in resumed_calls] == [1, 1]
        assert [c["seed"] for c in resumed_calls] == [4, 5]

    # total distinct seeds per cycle == games_per_cycle (never regenerated).
    # workers>1 records one batch call per run, so reconstruct the per-game
    # seed range each call covers: first batch spans 0..games-1 (only 4 of its
    # games completed), the resumed batch continues from games_this_cycle.
    def game_seeds(call_list):
        return sorted({s for c in call_list
                       for s in range(c["seed"], c["seed"] + c["games"])})

    all_seeds = game_seeds(calls)
    resume_seeds = game_seeds(resumed_calls)
    assert min(resume_seeds) == 4                 # continues from game 4
    assert max(resume_seeds) == 5                 # == games_per_cycle - 1
    assert len(set(all_seeds) | set(resume_seeds)) == 6

    # cycle completed on resume: games_this_cycle cleared, full total persisted
    ckpt2 = loop.load_checkpoint(tmp_path / "models" / "latest.pt")
    assert ckpt2["extra"]["games_this_cycle"] == 0
    assert ckpt2["extra"]["games_generated"] == 6
    assert ckpt2["extra"]["cycles_completed"] == 1


# ---------------------------------------------------------------------------
# (c) reset guard: cycle completion clears games_this_cycle (incl. the
#     _eval_now checkpoint); mid-cycle eval-gate saves do NOT clear it
# ---------------------------------------------------------------------------

def test_cycle_completion_clears_mid_cycle_eval_does_not(tmp_path, monkeypatch):
    calls = []
    report, extras = run_loop(
        tmp_path, monkeypatch, calls, workers=1, save_every_games=10,
        games_per_cycle=4, steps_per_cycle=3, eval_interval_steps=1)
    assert report["loop"]["eval_gates"] == 3  # step 1, 2, and cycle-end

    # the eval saves fire in order: mid-cycle (steps 1, 2) then cycle-end.
    # Mid-cycle eval saves keep games_this_cycle at games_per_cycle (4).
    mid_cycle = [e for e in extras if 0 < e["steps_into_cycle"] < 3]
    assert len(mid_cycle) == 2
    assert all(e["games_this_cycle"] == 4 for e in mid_cycle)  # NOT cleared

    # the cycle-end _eval_now checkpoint clears it (steps_into_cycle >= 3)
    cycle_end = [e for e in extras
                 if e["steps_into_cycle"] == 0 and e["cycles_completed"] == 1]
    assert cycle_end
    assert all(e["games_this_cycle"] == 0 for e in cycle_end)

    # final checkpoint on disk: cycle complete -> games_this_cycle 0
    ckpt = loop.load_checkpoint(tmp_path / "models" / "latest.pt")
    assert ckpt["extra"]["games_this_cycle"] == 0
    assert ckpt["extra"]["games_generated"] == 4


# ---------------------------------------------------------------------------
# (d) concurrency: drain-thread save racing an interrupt save keeps latest.pt
#     loadable
# ---------------------------------------------------------------------------

def test_drain_and_interrupt_saves_keep_ckpt_loadable(tmp_path, monkeypatch):
    calls = []
    report, extras = run_loop(
        tmp_path, monkeypatch, calls, workers=2, save_every_games=2,
        games_per_cycle=8, steps_per_cycle=2, kill_after=3,
        leave_drain_racing=True)
    assert report["loop"]["interrupted"] is True

    # at least the drain-thread mid-selfplay saves happened while racing
    assert any(e["games_this_cycle"] > 0 for e in extras)

    # latest.pt must always be a complete, loadable checkpoint
    last = None
    for _ in range(50):
        try:
            last = loop.load_checkpoint(tmp_path / "models" / "latest.pt")
            break
        except Exception:  # noqa: BLE001 - keep retrying while a save races
            time.sleep(0.02)
    assert last is not None
    assert last["format"] == train.CHECKPOINT_FORMAT
    assert "games_this_cycle" in last["extra"]
    # give any still-running daemon drain thread a chance to finish
    time.sleep(0.3)
    loadable = loop.load_checkpoint(tmp_path / "models" / "latest.pt")
    assert loadable["format"] == train.CHECKPOINT_FORMAT


# ---------------------------------------------------------------------------
# (e) --save-every-games 0 -> no mid-cycle saves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_save_every_games_zero_disables_mid_cycle_saves(tmp_path, monkeypatch,
                                                        workers):
    calls = []
    report, extras = run_loop(
        tmp_path, monkeypatch, calls, workers=workers, save_every_games=0,
        games_per_cycle=6, steps_per_cycle=1)
    assert report["protocol"]["save_every_games"] == 0
    # the ONLY saves are cycle-complete ones (cycle-end eval + final): none
    # carries a non-zero games_this_cycle, i.e. no mid-selfplay checkpoint.
    assert all(e["games_this_cycle"] == 0 for e in extras)
    assert len(mid_selfplay_saves(extras, 6)) == 0


# ---------------------------------------------------------------------------
# (f) old checkpoint without the key -> resume works, full regeneration
# ---------------------------------------------------------------------------

def test_old_checkpoint_without_key_resumes_full_regen(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = create_model(BLOCKS, CHANNELS, SIZE).to(DEVICE)
    opt = make_sgd_optimizer(model, lr=0.1, momentum=0.9, l2=1e-4)
    train.save_checkpoint(
        tmp_path / "models" / "latest.pt", model, opt, global_step=0,
        config={"seed": 42}, rng=np.random.default_rng(42),
        extra={"games_generated": 0, "steps_into_cycle": 0,
               "cycles_completed": 0})  # no games_this_cycle key

    calls = []
    report, extras = run_loop(
        tmp_path, monkeypatch, calls, resume=True, save_every_games=None,
        games_per_cycle=4, steps_per_cycle=2, seed=0)
    assert report["loop"]["resumed"] is True
    # missing key -> games_this_cycle defaults to 0 -> the FULL cycle is
    # regenerated from the persisted games_generated (no continuation)
    assert [c["seed"] for c in calls] == [0, 1, 2, 3]
    assert report["loop"]["games_generated"] == 4
    # save_every_games falls back to the default (10) on old checkpoints
    assert report["protocol"]["save_every_games"] == 10
    ckpt = loop.load_checkpoint(tmp_path / "models" / "latest.pt")
    assert ckpt["extra"]["games_this_cycle"] == 0
    assert ckpt["extra"]["games_generated"] == 4
