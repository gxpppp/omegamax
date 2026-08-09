"""P10b: async-prefetch tests for the pretrain pipeline.

Covers:
(a) a tiny 9x9 CPU run (blocks=1 channels=8, 20 steps) with prefetch on:
    loss finite + decreasing (the async sampler must feed valid batches);
(b) determinism: same seed -> identical loss curve (the prefetch worker's
    generator is seeded from the persistent rng, so a fixed seed reproduces
    a run exactly even though sampling now runs in a background thread);
(c) batch contents sane: the prefetch worker hands over the same shapes /
    values as ``PretrainChunks.sample_batch``, in torch-ready dtypes;
(d) escape hatch: ``prefetch=False`` keeps the original fully-serial,
    rng-driven sampling (same seed -> same loss, deterministic).
"""

import numpy as np
import pytest
import torch

from omigamax.network.model import create_model
from omigamax.train.pretrain import (
    PretrainChunks,
    _PrefetchSampler,
    _prepare_batch,
    make_pretrain_optimizer,
    run_pretrain,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_chunks(dir_path, board, sizes, seed=0):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, n in enumerate(sizes):
        s = rng.integers(0, 2, size=(n, 17, board, board)).astype(np.uint8)
        pi = rng.integers(0, board * board + 1, size=n).astype(np.uint16)
        z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        np.savez(dir_path / f"chunk_{i:04d}.npz", s=s, pi=pi, z=z)
    return dir_path


def tiny_run_losses(tmp_path, *, seed, steps=20, prefetch=True):
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[64, 64])
    torch.manual_seed(seed)
    model = create_model(1, 8, board)
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    with PretrainChunks(data_dir) as chunks:
        metrics, step, rng = run_pretrain(
            model, optimizer, chunks, steps=steps, seed=seed, global_step=0,
            batch_size=16, device=torch.device("cpu"), lr_base=0.02,
            lr_steps=(), prefetch=prefetch,
        )
    assert step == steps
    return [m["loss_total"] for m in metrics], rng


# ---------------------------------------------------------------------------
# (a) tiny run with prefetch on: loss finite + decreasing
# ---------------------------------------------------------------------------

def test_prefetch_tiny_run_20_steps_finite_and_decreasing(tmp_path):
    losses, _ = tiny_run_losses(tmp_path, seed=11, steps=20, prefetch=True)
    assert len(losses) == 20
    assert all(np.isfinite(losses)), f"non-finite losses: {losses}"
    assert losses[-1] < losses[0] * 0.98, (
        f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    )


# ---------------------------------------------------------------------------
# (b) determinism: same seed -> same loss curve under prefetch
# ---------------------------------------------------------------------------

def test_prefetch_determinism_same_seed_same_loss(tmp_path):
    losses1, _ = tiny_run_losses(tmp_path, seed=123, steps=20, prefetch=True)
    losses2, _ = tiny_run_losses(tmp_path, seed=123, steps=20, prefetch=True)
    assert len(losses1) == len(losses2) == 20
    assert all(
        a == pytest.approx(b, abs=0.0) for a, b in zip(losses1, losses2)
    ), f"same seed diverged: {losses1} vs {losses2}"


# ---------------------------------------------------------------------------
# (c) batch contents sane (worker -> _prepare_batch -> torch-ready dtypes)
# ---------------------------------------------------------------------------

def test_prepare_batch_preserves_values_shapes(tmp_path):
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[40, 40])
    with PretrainChunks(data_dir) as chunks:
        raw = chunks.sample_batch(np.random.default_rng(5), 12)
        prepared = _prepare_batch(raw)
    assert prepared["s"].shape == (12, 17, board, board)
    assert prepared["pi"].shape == (12,)
    assert prepared["z"].shape == (12,)
    assert prepared["s"].dtype == np.float32
    assert prepared["pi"].dtype == np.int64
    assert prepared["z"].dtype == np.float32
    # values preserved exactly across the dtype conversion
    assert np.array_equal(prepared["s"], raw["s"].astype(np.float32))
    assert np.array_equal(prepared["pi"], raw["pi"].astype(np.int64))
    assert np.array_equal(prepared["z"], raw["z"].astype(np.float32))
    assert set(np.unique(prepared["s"])).issubset({0.0, 1.0})
    assert set(np.unique(prepared["z"])).issubset({-1.0, 1.0})
    assert int(prepared["pi"].min()) >= 0 and int(prepared["pi"].max()) <= board * board


def test_prefetch_sampler_get_yields_sane_batches(tmp_path):
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[30, 40])
    with PretrainChunks(data_dir) as chunks:
        sampler = _PrefetchSampler(chunks, batch_size=16, seed=7).start()
        try:
            for _ in range(3):
                b = sampler.get()
                assert b["s"].dtype == np.float32 and b["s"].shape == (16, 17, board, board)
                assert b["pi"].dtype == np.int64 and b["pi"].shape == (16,)
                assert b["z"].dtype == np.float32 and b["z"].shape == (16,)
                assert set(np.unique(b["z"])).issubset({-1.0, 1.0})
        finally:
            sampler.stop()
        assert not sampler._thread.is_alive(), "worker did not exit on stop()"


# ---------------------------------------------------------------------------
# (d) escape hatch: prefetch=False keeps serial rng-driven sampling
# ---------------------------------------------------------------------------

def test_no_prefetch_serial_determinism(tmp_path):
    losses1, _ = tiny_run_losses(tmp_path, seed=42, steps=20, prefetch=False)
    losses2, _ = tiny_run_losses(tmp_path, seed=42, steps=20, prefetch=False)
    assert losses1 == losses2, "prefetch=False must be bit-reproducible"
    assert all(np.isfinite(losses1))
    assert losses1[-1] < losses1[0] * 0.98


def test_prefetch_rng_advanced_by_one_draw_only(tmp_path):
    """With prefetch on, the persistent rng must advance by exactly one draw
    per run_pretrain call (the worker seed), preserving checkpoint semantics."""
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[32, 32])
    rng = np.random.default_rng(99)
    torch.manual_seed(0)
    model = create_model(1, 8, board)
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    with PretrainChunks(data_dir) as chunks:
        _, step, rng_out = run_pretrain(
            model, optimizer, chunks, steps=5, rng=rng, seed=99,
            global_step=0, batch_size=16, device=torch.device("cpu"),
            lr_base=0.02, lr_steps=(), prefetch=True,
        )
    assert step == 5
    assert rng_out is rng  # same object returned, only advanced by the seed draw
    # a fresh rng with the same seed must land in the same state after the one
    # worker-seed draw -- proving exactly one draw (and no per-step sampling)
    rng_probe = np.random.default_rng(99)
    _ = int(rng_probe.integers(0, 2**32 - 1))
    assert rng_probe.bit_generator.state == rng_out.bit_generator.state
