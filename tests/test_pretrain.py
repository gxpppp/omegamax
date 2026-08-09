"""P5: supervised-pretraining tests (loader / loss / tiny run / GPU smoke).

Covers:
(a) ``PretrainChunks``: stored shapes/dtypes/value ranges, seeded
    deterministic sampling, cross-chunk batch gathering;
(b) loss: one-hot policy CE equals ``F.cross_entropy``, value MSE exact on
    hand-made tensors, total = policy + value;
(c) a tiny real run (9x9, blocks=1 channels=8, batch 8, 30 steps on 3
    synthetic mini-games) -- loss decreases, checkpoint round-trips with its
    ``arch``, and ``--resume``-style continuation advances ``global_step``;
(d) full b20c256 20-step batch-64 GPU smoke -- loss decreases, peak memory
    <= 5.5 GB (P4 ground truth: 2.15 GB @ batch 64).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from omigamax.network.model import create_model
from omigamax.train.loss import (
    agz_loss,
    policy_cross_entropy,
    value_mse,
)
from omigamax.train.pretrain import (
    PretrainChunks,
    make_pretrain_optimizer,
    pretrain_lr,
    pretrain_step,
    resume_from_checkpoint,
    run_pretrain,
    save_pretrain_checkpoint,
)
from omigamax.train.train import load_checkpoint, restore_from_checkpoint

GIB = 1024 ** 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_synthetic_chunks(dir_path, board, sizes, seed=0):
    """Write ``chunk_%04d.npz`` files with valid random SL samples."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, n in enumerate(sizes):
        s = rng.integers(0, 2, size=(n, 17, board, board)).astype(np.uint8)
        pi = rng.integers(0, board * board + 1, size=n).astype(np.uint16)
        z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        np.savez(dir_path / f"chunk_{i:04d}.npz", s=s, pi=pi, z=z)
    return dir_path


def random_batch_tensors(batch, board, device=DEVICE):
    """(inputs float, pi_idx int64, z (B,1) float) random smoke tensors."""
    n_logits = board * board + 1
    inputs = torch.randn(batch, 17, board, board, device=device)
    pi_idx = torch.randint(0, n_logits, (batch,), device=device)
    z = torch.randint(0, 2, (batch, 1), device=device).float() * 2.0 - 1.0
    return inputs, pi_idx, z


# ---------------------------------------------------------------------------
# (a) loader
# ---------------------------------------------------------------------------

def test_loader_shapes_dtypes_values(tmp_path):
    board = 9
    make_synthetic_chunks(tmp_path, board, sizes=[20, 30])
    with PretrainChunks(tmp_path) as chunks:
        assert chunks.total_n == 50
        assert chunks.num_chunks == 2
        assert chunks.sizes == [20, 30]
        report = chunks.validate()
        assert report["total_positions"] == 50
        rng = np.random.default_rng(1)
        b = chunks.sample_batch(rng, 8)
        assert b["s"].dtype == np.uint8 and b["s"].shape == (8, 17, board, board)
        assert b["pi"].dtype == np.uint16 and b["pi"].shape == (8,)
        assert b["z"].dtype == np.int8 and b["z"].shape == (8,)
        assert set(np.unique(b["s"])).issubset({0, 1})
        assert int(b["pi"].min()) >= 0 and int(b["pi"].max()) <= board * board
        assert set(np.unique(b["z"])).issubset({-1, 1})


def test_loader_sampling_deterministic_with_seed(tmp_path):
    board = 9
    make_synthetic_chunks(tmp_path, board, sizes=[17, 11, 23])
    with PretrainChunks(tmp_path) as chunks:
        b1 = chunks.sample_batch(np.random.default_rng(7), 16)
        b2 = chunks.sample_batch(np.random.default_rng(7), 16)
        assert np.array_equal(b1["s"], b2["s"])
        assert np.array_equal(b1["pi"], b2["pi"])
        assert np.array_equal(b1["z"], b2["z"])
        b3 = chunks.sample_batch(np.random.default_rng(8), 16)
        # fixed seeds -> fixed (deterministically different) batches
        assert not np.array_equal(b1["pi"], b3["pi"])


def test_loader_batch_spans_chunks_and_rows_exist(tmp_path):
    """A batch larger than the first chunk must gather across chunks, and every
    returned row must come from the underlying corpus."""
    board = 9
    make_synthetic_chunks(tmp_path, board, sizes=[5, 7, 9])
    with PretrainChunks(tmp_path) as chunks:
        full = {}
        for ci in range(chunks.num_chunks):
            full.setdefault("s", []).append(chunks._get(ci, "s"))
            full.setdefault("pi", []).append(chunks._get(ci, "pi"))
            full.setdefault("z", []).append(chunks._get(ci, "z"))
        full_s = np.concatenate(full["s"])  # (21, 17, 9, 9)
        full_pi = np.concatenate(full["pi"])
        full_z = np.concatenate(full["z"])
        b = chunks.sample_batch(np.random.default_rng(3), 12)
        assert b["s"].shape[0] == 12
        for i in range(b["s"].shape[0]):
            hits = np.all(full_s == b["s"][i], axis=(1, 2, 3))
            assert hits.any(), f"row {i} not in corpus"
            j = int(np.flatnonzero(hits)[0])
            assert full_pi[j] == b["pi"][i]
            assert full_z[j] == b["z"][i]


def test_loader_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PretrainChunks(tmp_path)


# ---------------------------------------------------------------------------
# (b) loss
# ---------------------------------------------------------------------------

def test_onehot_policy_ce_equals_cross_entropy():
    torch.manual_seed(0)
    board = 9
    batch, n_logits = 8, board * board + 1
    logits = torch.randn(batch, n_logits)
    pi_idx = torch.randint(0, n_logits, (batch,))
    pi_onehot = F.one_hot(pi_idx, num_classes=n_logits).float()
    assert torch.allclose(
        policy_cross_entropy(logits, pi_onehot),
        F.cross_entropy(logits, pi_idx),
        atol=1e-6,
    )


def test_value_mse_correct_hand_made():
    value = torch.tensor([[0.5], [-1.0], [0.25]])
    z = torch.tensor([[1.0], [-1.0], [-1.0]])
    expected = float(
        ((value - z) ** 2).mean()
    )
    assert value_mse(value, z).item() == pytest.approx(expected)


def test_agz_loss_equals_policy_plus_value():
    torch.manual_seed(1)
    board = 19
    batch, n_logits = 4, board * board + 1
    logits = torch.randn(batch, n_logits)
    value = torch.randn(batch, 1).tanh()
    pi_idx = torch.randint(0, n_logits, (batch,))
    pi_onehot = F.one_hot(pi_idx, num_classes=n_logits).float()
    z = torch.randint(0, 2, (batch, 1)).float() * 2.0 - 1.0
    total = agz_loss(logits, value, pi_onehot, z)
    assert total.item() == pytest.approx(
        policy_cross_entropy(logits, pi_onehot).item()
        + value_mse(value, z).item()
    )


def test_pretrain_step_metrics_components_consistent(tmp_path):
    """pretrain_step's total loss == policy + value, acc in [0,1]."""
    board = 9
    make_synthetic_chunks(tmp_path, board, sizes=[16])
    torch.manual_seed(0)
    model = create_model(1, 8, board)
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    with PretrainChunks(tmp_path) as chunks:
        batch = chunks.sample_batch(np.random.default_rng(0), 8)
        m = pretrain_step(model, optimizer, batch, device=torch.device("cpu"))
    assert m["loss_total"] == pytest.approx(m["loss_policy"] + m["loss_value"])
    assert 0.0 <= m["acc_top1"] <= 1.0
    assert all(np.isfinite(m[k]) for k in ("loss_total", "loss_policy", "loss_value"))


# ---------------------------------------------------------------------------
# (c) tiny real pretrain run + checkpoint round-trip + resume
# ---------------------------------------------------------------------------

def test_pretrain_lr_schedule():
    assert pretrain_lr(0, 0.02, (50000,)) == pytest.approx(0.02)
    assert pretrain_lr(50000, 0.02, (50000,)) == pytest.approx(0.01)
    assert pretrain_lr(100000, 0.02, (50000,)) == pytest.approx(0.01)
    assert pretrain_lr(100000, 0.02, (50000, 100000)) == pytest.approx(0.005)


def test_tiny_pretrain_run_checkpoint_roundtrip_and_resume(tmp_path):
    """~30 steps on 3 synthetic mini-games: loss decreases, checkpoint
    round-trips with arch, resume continues the global step counter."""
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[20, 20, 20])
    ckpt_path = tmp_path / "pretrain.pt"
    cpu = torch.device("cpu")

    torch.manual_seed(0)
    model = create_model(1, 8, board)
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    with PretrainChunks(data_dir) as chunks:
        rng = np.random.default_rng(3)
        metrics, step1, rng = run_pretrain(
            model, optimizer, chunks, steps=30, rng=rng, global_step=0,
            batch_size=8, device=cpu, lr_base=0.02, lr_steps=(),
        )
    assert step1 == 30
    assert len(metrics) == 30
    assert metrics[-1]["loss_total"] < metrics[0]["loss_total"] * 0.98, (
        f"loss did not decrease: {metrics[0]['loss_total']} -> {metrics[-1]['loss_total']}"
    )
    save_pretrain_checkpoint(
        ckpt_path, model, optimizer, global_step=step1, rng=rng,
        config={"blocks": 1, "channels": 8, "board_size": board},
    )

    # round-trip through the shared checkpoint format (P7 path)
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["arch"] == {"blocks": 1, "channels": 8, "board_size": board}
    assert ckpt["global_step"] == 30
    model2 = create_model(**ckpt["arch"])
    model2.load_state_dict(ckpt["model_state_dict"])
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), model2.state_dict().values())
    )

    # resume: restore step counter + optimizer + rng, continue
    optimizer2 = make_pretrain_optimizer(model2, lr=0.02, momentum=0.9, l2=1e-4)
    rng2 = np.random.default_rng(3)
    step_restored = resume_from_checkpoint(ckpt, model2, optimizer2, rng2)
    assert step_restored == 30
    with PretrainChunks(data_dir) as chunks2:
        metrics2, step2, _ = run_pretrain(
            model2, optimizer2, chunks2, steps=10, rng=rng2,
            global_step=step_restored, batch_size=8, device=cpu,
            lr_base=0.02, lr_steps=(),
        )
    assert step2 == 40
    assert len(metrics2) == 10
    assert metrics2[0]["step"] == 30


# ---------------------------------------------------------------------------
# (d) full b20c256 GPU smoke (mirrors the P4 memory protocol)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_b20c256_gpu_smoke_20_steps_loss_and_memory(tmp_path):
    """b20c256 @ batch 64, 20 steps on synthetic chunks: loss decreases and
    peak GPU memory stays <= 5.5 GB (P4 measured 2.15 GB)."""
    board, blocks, channels = 19, 20, 256
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[512])
    torch.manual_seed(0)
    model = create_model(blocks, channels, board).cuda()
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    torch.cuda.reset_peak_memory_stats()
    with PretrainChunks(data_dir) as chunks:
        metrics, step, _ = run_pretrain(
            model, optimizer, chunks, steps=20, rng=np.random.default_rng(0),
            global_step=0, batch_size=64, device=DEVICE, lr_base=0.02, lr_steps=(),
        )
    peak_gb = torch.cuda.max_memory_allocated() / GIB
    assert peak_gb <= 5.5, f"peak {peak_gb:.3f} GB > 5.5 GB"
    assert step == 20 and len(metrics) == 20
    assert metrics[-1]["loss_total"] < metrics[0]["loss_total"] * 0.98, (
        f"loss did not decrease: {metrics[0]['loss_total']} -> {metrics[-1]['loss_total']}"
    )
    assert torch.isfinite(torch.tensor([m["loss_total"] for m in metrics])).all()


def test_pretrain_lr_schedule_empty_steps_is_constant():
    assert pretrain_lr(12345, 0.02, ()) == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# (e) AMP (autocast + GradScaler) -- opt-in, fp32 default unchanged
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pretrain_run_amp_tiny_9x9_20_steps_finite_and_decreasing(tmp_path):
    """AMP run on a tiny 9x9 model (blocks=1, channels=8), 20 steps: loss stays
    finite (no NaNs) and decreases -- the autocast+GradScaler path trains.

    Uses ``prefetch=False`` so the assertion keys off the deterministic serial
    stream (the intent here is AMP correctness, not sampling). The async
    prefetch path is covered by tests/test_pretrain_prefetch.py.
    """
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[256])
    torch.manual_seed(0)
    model = create_model(1, 8, board).cuda()
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    with PretrainChunks(data_dir) as chunks:
        metrics, step, _ = run_pretrain(
            model, optimizer, chunks, steps=20, rng=np.random.default_rng(0),
            global_step=0, batch_size=16, device=DEVICE, lr_base=0.02,
            lr_steps=(), amp=True, prefetch=False,
        )
    assert step == 20 and len(metrics) == 20
    losses = [m["loss_total"] for m in metrics]
    assert all(np.isfinite(losses)), f"non-finite losses under AMP: {losses}"
    assert losses[-1] < losses[0] * 0.98, (
        f"loss did not decrease under AMP: {losses[0]} -> {losses[-1]}"
    )


def test_pretrain_amp_requires_cuda(tmp_path):
    """amp=True on a CPU run is rejected loudly instead of silently ignoring."""
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[16])
    model = create_model(1, 8, board)
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    with PretrainChunks(data_dir) as chunks:
        with pytest.raises(ValueError, match="CUDA"):
            run_pretrain(
                model, optimizer, chunks, steps=2, batch_size=8,
                device=torch.device("cpu"), amp=True,
            )
