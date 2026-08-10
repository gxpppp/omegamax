"""P13: run-param persistence into checkpoints (human_mix / selfplay_workers).

The loop's run-only params (human_mix, selfplay_workers, pretrain_data_dir)
are absent from ``config/default.yaml``, so before P13 a machine restart +
``--resume`` silently reverted a mixing run to pure-RL. The fix persists the
resolved values into the checkpoint's ``config`` (via ``persist_cfg``, a copy
of the caller's cfg) and resolves them on resume as:
explicit CLI arg > resumed checkpoint config > cfg > safe default.

Covers:
(a) a run with human_mix=0.25 + selfplay_workers=2 writes a checkpoint whose
    config carries both (and the caller's cfg dict is NOT mutated);
(b) resume from that checkpoint WITHOUT explicit flags resolves 0.25 / 2 /
    the persisted corpus dir (asserted via the returned report protocol);
(c) explicit CLI values override the checkpoint's on resume;
(d) an old-style checkpoint without the keys resolves to the safe defaults
    (human_mix=0.0, workers=1) without crashing.

All runs use a tiny 9x9 b1c8 net on CPU with fake self-play/training so the
checkpoint machinery is exercised without a real RL run.
"""

import numpy as np
import pytest
import torch

import omigamax.train.loop as loop
from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.train import load_checkpoint, save_checkpoint

SIZE = 9
BLOCKS = 1
CHANNELS = 8
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_human_chunks(dir_path, board=SIZE, sizes=(32, 32), seed=7):
    """Write valid ``chunk_%04d.npz`` human-corpus files (P3 format)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, n in enumerate(sizes):
        s = rng.integers(0, 2, size=(n, 17, board, board)).astype(np.uint8)
        pi = rng.integers(0, board * board + 1, size=n).astype(np.uint16)
        z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        np.savez(dir_path / f"chunk_{i:04d}.npz", s=s, pi=pi, z=z)
    return dir_path


def make_loop_cfg():
    """Tiny 9x9 CPU loop config (arch + fast cycle params)."""
    cfg = dict(load_config())
    cfg.update({"board_size": SIZE, "blocks": BLOCKS, "channels": CHANNELS,
                "simulations": 4, "eval_games": 2, "eval_sims": 4,
                "eval_interval_steps": 2000, "replace_threshold": 0.55,
                "batch_size": 8, "cycle_steps": 1, "cycle_games": 2,
                "lr": 0.1})
    return cfg


def fake_gen(network, cfg, games, data_dir, keep, seed, simulations, **kw):
    return {"games": int(games), "sims": int(games) * 30 * simulations,
            "positions": int(games) * 30, "wall_time_s": 1.0,
            "sims_per_sec": 1.0, "data_dir": str(data_dir)}, []


def fake_train(model, optimizer, buffer, steps, **kwargs):
    return [0.5], [0.2], int(kwargs["global_step"]) + 1, kwargs["rng"]


def run_smoke(cfg, tmp_path, monkeypatch, **run_kwargs):
    """Run the loop on fakes; return (report, kwargs already applied)."""
    monkeypatch.setattr(loop, "generate_games", fake_gen)
    monkeypatch.setattr(loop, "train_steps", fake_train)
    defaults = dict(
        device=DEVICE,
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "models",
        train_log=tmp_path / "train.jsonl",
        history=tmp_path / "eval_history.jsonl",
        cycles=1, games_per_cycle=2, steps_per_cycle=1,
        viz_enabled=False,
    )
    defaults.update(run_kwargs)
    return loop.run_loop(cfg, **defaults)


# ---------------------------------------------------------------------------
# (a) run params are written into the checkpoint config
# ---------------------------------------------------------------------------

def test_checkpoint_persists_run_params(tmp_path, monkeypatch):
    corpus = make_human_chunks(tmp_path / "human")
    cfg = make_loop_cfg()
    report = run_smoke(cfg, tmp_path, monkeypatch,
                       human_mix=0.25, selfplay_workers=2,
                       pretrain_data_dir=corpus)
    assert report["protocol"]["human_mix"] == 0.25
    assert report["protocol"]["selfplay_workers"] == 2

    ckpt = load_checkpoint(tmp_path / "models" / "latest.pt")
    assert ckpt["config"]["human_mix"] == 0.25
    assert ckpt["config"]["selfplay_workers"] == 2
    assert ckpt["config"]["pretrain_data_dir"] == str(corpus)

    # the caller's cfg dict is never mutated (persist_cfg is a copy)
    assert "human_mix" not in cfg
    assert "selfplay_workers" not in cfg


# ---------------------------------------------------------------------------
# (b) resume WITHOUT flags restores the checkpoint's values
# ---------------------------------------------------------------------------

def test_resume_restores_run_params(tmp_path, monkeypatch):
    corpus = make_human_chunks(tmp_path / "human")
    cfg = make_loop_cfg()
    run_smoke(cfg, tmp_path, monkeypatch,
              human_mix=0.25, selfplay_workers=2, pretrain_data_dir=corpus)

    # bare --resume: no run-param flags -> resolve from the checkpoint
    report = run_smoke(cfg, tmp_path, monkeypatch, resume=True)
    assert report["protocol"]["resumed"] is True
    assert report["protocol"]["human_mix"] == 0.25
    assert report["protocol"]["selfplay_workers"] == 2
    assert report["protocol"]["pretrain_data_dir"] == str(corpus)


# ---------------------------------------------------------------------------
# (c) explicit CLI values override the checkpoint's on resume
# ---------------------------------------------------------------------------

def test_explicit_flags_override_checkpoint(tmp_path, monkeypatch):
    corpus = make_human_chunks(tmp_path / "human")
    cfg = make_loop_cfg()
    run_smoke(cfg, tmp_path, monkeypatch,
              human_mix=0.25, selfplay_workers=2, pretrain_data_dir=corpus)

    report = run_smoke(cfg, tmp_path, monkeypatch,
                       resume=True,
                       human_mix=0.5, selfplay_workers=1)
    assert report["protocol"]["human_mix"] == 0.5
    assert report["protocol"]["selfplay_workers"] == 1


# ---------------------------------------------------------------------------
# (d) old-style checkpoint without the keys -> safe defaults, no crash
# ---------------------------------------------------------------------------

def test_old_checkpoint_resolves_safe_defaults(tmp_path, monkeypatch):
    # hand-write an old-style checkpoint whose config has NO run-param keys
    torch.manual_seed(0)
    model = create_model(BLOCKS, CHANNELS, SIZE).to(DEVICE)
    opt = make_sgd_optimizer(model, lr=0.1, momentum=0.9, l2=1e-4)
    ckpt_dir = tmp_path / "models"
    save_checkpoint(ckpt_dir / "latest.pt", model, opt, global_step=5,
                    config={"seed": 42},
                    rng=np.random.default_rng(42))

    cfg = make_loop_cfg()
    report = run_smoke(cfg, tmp_path, monkeypatch, resume=True)
    assert report["protocol"]["resumed"] is True
    assert report["protocol"]["human_mix"] == 0.0
    assert report["protocol"]["selfplay_workers"] == 1
