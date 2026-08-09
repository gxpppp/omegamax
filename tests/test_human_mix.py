"""P11: human-data RL mixing tests (KataGo-style (1-mix)/mix batches).

Covers:
(a) ``human_mix == 0`` is byte-identical to the pure-RL path -- same losses
    and the same final RNG state, and the human sampler is never invoked;
(b) ``human_mix == 0.25`` on a tiny 9x9 setup: batch composition is exactly
    ``rl_n = batch_size - round(mix*batch_size)`` self-play + ``human_n``
    human samples per step, the composed batch flows through the existing
    ``train_on_batch`` unchanged, losses stay finite and decrease over 20
    steps;
(c) ``convert_to_rl_batch``: uint8->float32 value-preserving, move-index ->
    one-hot (pass index lands in the last slot), int8 z -> (B, 1) float32;
(d) resume reproducibility with mixing on: (k save+reload k) == 2k
    uninterrupted, same seed (the single persisted RNG drives both streams);
(e) symmetry + mixed batch: the 8x-augmented batch keeps shapes consistent
    (8B x 17 x N x N / 8B x N*N+1 / 8B x 1) and trains with finite loss.

All runs use a tiny 9x9 b1c8 net on the fastest available device (CPU keeps
the resume test bit-deterministic).
"""

import numpy as np
import pytest
import torch

import omigamax.train.train as train_mod
from omigamax.network.model import create_model
from omigamax.train.buffer import ReplayBuffer
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.pretrain import (
    PretrainChunks,
    convert_to_rl_batch,
    make_human_sampler,
)
from omigamax.train.train import (
    load_checkpoint,
    restore_from_checkpoint,
    restore_rng,
    save_checkpoint,
    train_steps,
)

SIZE = 9
BLOCKS = 1
CHANNELS = 8
DEVICE = torch.device("cpu")  # deterministic resume needs bit-exact ops
TOL = 1e-6


# ---------------------------------------------------------------------------
# helpers (synthetic corpus + replay games, same shapes as the real ones)
# ---------------------------------------------------------------------------

def make_human_chunks(dir_path, board=SIZE, sizes=(64, 64), seed=7):
    """Write valid ``chunk_%04d.npz`` human-corpus files (P3 format)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, n in enumerate(sizes):
        s = rng.integers(0, 2, size=(n, 17, board, board)).astype(np.uint8)
        pi = rng.integers(0, board * board + 1, size=n).astype(np.uint16)
        z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        np.savez(dir_path / f"chunk_{i:04d}.npz", s=s, pi=pi, z=z)
    return dir_path


def make_replay_games(data_dir, games=2, seed_base=11, size=SIZE, t=40):
    """Deterministic synthetic npz self-play games (RL-buffer format)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 + int(seed_base))
    for g in range(int(games)):
        s = rng.random((t, 17, size, size)).astype(np.float32)
        pi = rng.random((t, size * size + 1)).astype(np.float32)
        pi /= pi.sum(axis=1, keepdims=True)
        z = rng.choice([-1.0, 1.0], size=t).astype(np.float32)
        np.savez(
            data_dir / f"game_{int(seed_base) + g:010d}.npz",
            s=s, pi=pi, z=z,
            board_size=np.int64(size), move_count=np.int64(t),
        )
    return data_dir


def make_tiny_setup(tmp_path):
    """(model, optimizer, buffer, human_chunks) on the tiny 9x9 fixture.

    The model is created under a fixed ``torch.manual_seed`` so every setup
    shares the SAME initialization (needed by the resume test, where the
    interrupted and uninterrupted runs must start from identical weights).
    """
    chunk_dir = make_human_chunks(tmp_path / "human")
    game_dir = make_replay_games(tmp_path / "selfplay")
    torch.manual_seed(0)
    model = create_model(BLOCKS, CHANNELS, SIZE).to(DEVICE)
    optimizer = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    buffer = ReplayBuffer(game_dir, max_games=50, board_size=SIZE)
    chunks = PretrainChunks(chunk_dir)
    return model, optimizer, buffer, chunks


# ---------------------------------------------------------------------------
# (a) mix == 0 is the pure-RL path
# ---------------------------------------------------------------------------

def test_mix_zero_is_pure_rl(tmp_path):
    def must_not_be_called(rng, n):  # pragma: no cover - failure probe
        raise AssertionError("human sampler must not be called at mix=0")

    # Two independent fresh setups: both runs start from identical weights and
    # data so the pure-RL path must reproduce the same trajectory exactly.
    model_a, opt_a, buffer_a, _ = make_tiny_setup(tmp_path / "a")
    losses_a, _, step_a, rng_a = train_steps(
        model_a, opt_a, buffer_a, steps=8, rng=np.random.default_rng(3),
        batch_size=32, device=DEVICE, symmetry=False,
        human_sampler=None, human_mix=0.0,
    )
    model_b, opt_b, buffer_b, _ = make_tiny_setup(tmp_path / "b")
    losses_b, _, step_b, rng_b = train_steps(
        model_b, opt_b, buffer_b, steps=8, rng=np.random.default_rng(3),
        batch_size=32, device=DEVICE, symmetry=False,
        human_sampler=must_not_be_called, human_mix=0.0,
    )
    assert losses_a == losses_b
    assert step_a == step_b == 8
    # the final RNG state is identical too -> resume stays exact
    assert rng_a.bit_generator.state == rng_b.bit_generator.state


# ---------------------------------------------------------------------------
# (b) mix == 0.25 composition + finite/decreasing loss
# ---------------------------------------------------------------------------

def test_mix_quarter_composition(tmp_path, monkeypatch):
    model, optimizer, buffer, chunks = make_tiny_setup(tmp_path)
    human_sampler = make_human_sampler(chunks)
    batch_size = 64
    rl_ns, human_ns, batch_sizes = [], [], []
    real_sample = buffer.sample
    real_train = train_mod.train_on_batch

    def spy_sample(n, rng):
        rl_ns.append(int(n))
        return real_sample(int(n), rng)

    def spy_sampler(rng, n):
        human_ns.append(int(n))
        return human_sampler(rng, int(n))

    def spy_train(model, optimizer, batch, **kwargs):
        batch_sizes.append(int(batch["s"].shape[0]))
        return real_train(model, optimizer, batch, **kwargs)

    monkeypatch.setattr(buffer, "sample", spy_sample)
    monkeypatch.setattr(train_mod, "train_on_batch", spy_train)
    losses, _, _, _ = train_steps(
        model, optimizer, buffer, steps=4,
        rng=np.random.default_rng(5), batch_size=batch_size,
        device=DEVICE, symmetry=False,
        human_sampler=spy_sampler, human_mix=0.25,
    )
    # human_n = round(0.25 * 64) = 16; rl_n keeps the batch at exactly 64.
    assert rl_ns == [48, 48, 48, 48]
    assert human_ns == [16, 16, 16, 16]
    assert batch_sizes == [64, 64, 64, 64]
    assert all(np.isfinite(losses) for losses in losses)


def test_mix_quarter_loss_decreases(tmp_path):
    model, optimizer, buffer, chunks = make_tiny_setup(tmp_path)
    human_sampler = make_human_sampler(chunks)
    # One-hot human labels are sparse: the pretrain-style SL lr (0.02) keeps
    # the shared trunk stable while the policy head memorizes the targets.
    losses, _, _, _ = train_steps(
        model, optimizer, buffer, steps=20,
        rng=np.random.default_rng(5), batch_size=64,
        device=DEVICE, symmetry=False, lr_base=0.02,
        human_sampler=human_sampler, human_mix=0.25,
    )
    assert len(losses) == 20
    assert all(np.isfinite(losses) for losses in losses)
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# (c) conversion helper: s / pi / z to the RL batch format
# ---------------------------------------------------------------------------

def test_convert_to_rl_batch():
    n, board = 7, 9
    s_u8 = np.random.default_rng(0).integers(0, 2, (n, 17, board, board)).astype(np.uint8)
    pi_idx = np.array([0, 1, 40, 81, 80, 81, 17], dtype=np.uint16)  # 81 = pass
    z_i8 = np.array([1, -1, 1, -1, 1, -1, 1], dtype=np.int8)
    out = convert_to_rl_batch({"s": s_u8, "pi": pi_idx, "z": z_i8})

    assert out["s"].dtype == np.float32
    assert out["s"].shape == (n, 17, board, board)
    np.testing.assert_array_equal(out["s"], s_u8.astype(np.float32))  # value-preserving

    assert out["pi"].dtype == np.float32
    assert out["pi"].shape == (n, board * board + 1)
    for i, idx in enumerate(pi_idx):
        row = out["pi"][i]
        assert row.sum() == pytest.approx(1.0)
        assert row[int(idx)] == pytest.approx(1.0)  # index 81 (pass) lands last
        assert int(np.argmax(row)) == int(idx)
    # the pass slot (index N*N) is exactly the one-hot of index 81
    np.testing.assert_array_equal(out["pi"][3], out["pi"][5])

    assert out["z"].dtype == np.float32
    assert out["z"].shape == (n, 1)
    np.testing.assert_array_equal(out["z"][:, 0], z_i8.astype(np.float32))


# ---------------------------------------------------------------------------
# (d) resume reproducibility with mixing on
# ---------------------------------------------------------------------------

def test_resume_reproducible_with_mix(tmp_path):
    k = 10
    batch_size = 64
    seed = 42

    model_a, opt_a, buffer, chunks = make_tiny_setup(tmp_path)
    sampler = make_human_sampler(chunks)
    rng_a = np.random.default_rng(seed)
    losses_a, _, step_a, _ = train_steps(
        model_a, opt_a, buffer, steps=k, rng=rng_a,
        batch_size=batch_size, device=DEVICE, symmetry=False,
        human_sampler=sampler, human_mix=0.25,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model_a, opt_a, global_step=step_a,
                    config={"seed": seed}, rng=rng_a)
    ckpt = load_checkpoint(ckpt_path)
    step_restored = restore_from_checkpoint(ckpt, model_a, opt_a)
    restore_rng(rng_a, ckpt["rng_state"])
    losses_a2, _, step_a2, _ = train_steps(
        model_a, opt_a, buffer, steps=k, rng=rng_a, global_step=step_restored,
        batch_size=batch_size, device=DEVICE, symmetry=False,
        human_sampler=sampler, human_mix=0.25,
    )
    interrupted = losses_a + losses_a2
    assert step_a2 == 2 * k

    model_b, opt_b, buffer_b, chunks_b = make_tiny_setup(tmp_path)
    sampler_b = make_human_sampler(chunks_b)
    rng_b = np.random.default_rng(seed)
    uninterrupted, _, _, _ = train_steps(
        model_b, opt_b, buffer_b, steps=2 * k, rng=rng_b,
        batch_size=batch_size, device=DEVICE, symmetry=False,
        human_sampler=sampler_b, human_mix=0.25,
    )

    assert len(interrupted) == len(uninterrupted) == 2 * k
    np.testing.assert_allclose(
        np.asarray(interrupted), np.asarray(uninterrupted), atol=TOL, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# (e) symmetry + mixed batch: shapes survive the 8x augmentation
# ---------------------------------------------------------------------------

def test_symmetry_mixed_batch_shapes(tmp_path, monkeypatch):
    model, optimizer, buffer, chunks = make_tiny_setup(tmp_path)
    human_sampler = make_human_sampler(chunks)
    batch_size = 64
    seen = {}
    real_train = train_mod.train_on_batch

    def spy_train(model, optimizer, batch, **kwargs):
        seen["s"] = batch["s"].shape
        seen["pi"] = batch["pi"].shape
        seen["z"] = batch["z"].shape
        return real_train(model, optimizer, batch, **kwargs)

    monkeypatch.setattr(train_mod, "train_on_batch", spy_train)
    losses, _, _, _ = train_steps(
        model, optimizer, buffer, steps=3,
        rng=np.random.default_rng(9), batch_size=batch_size,
        device=DEVICE, symmetry=True,
        human_sampler=human_sampler, human_mix=0.25,
    )
    assert seen["s"] == (8 * batch_size, 17, SIZE, SIZE)
    assert seen["pi"] == (8 * batch_size, SIZE * SIZE + 1)
    assert seen["z"] == (8 * batch_size, 1)
    assert all(np.isfinite(losses) for losses in losses)
