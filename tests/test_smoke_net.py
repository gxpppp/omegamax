"""Tests for the AGZ training loss and the network smoke (todo 8).

Per the plan's todo-8 acceptance criteria and the training-smoke protocol:

* loss-function correctness -- analytic checks on hand-made tensors:
  policy CE of uniform logits vs uniform pi == ln(362); value MSE equals the
  manual mean-squared error; the AGZ loss equals CE + MSE (the value the
  optimizer minimizes); soft-target CE matches ``F.cross_entropy`` on one-hot
  targets; the analytic ``weight_l2`` monitor only sums conv/linear *weights*
  (``ndim >= 2``), never biases / batch-norm scales;
* L2 wiring -- :func:`make_sgd_optimizer` applies the plan's ``l2=1e-4`` as
  SGD ``weight_decay`` (the canonical, stable AGZ implementation; see
  ``omigamax/train/loss.py`` docstring for the empirical rationale);
* loss decreases over 200 SGD steps on the smoke dataset (batch 128),
  ``final < initial * 0.8`` -- the plan's "loss 有限且下降" made measurable;
* backward produces no NaN gradients;
* peak GPU memory of a batch-128 (b10c128, board 19) forward+backward stays
  <= 5.5 GB (the 6GB-card budget; measured via
  ``torch.cuda.max_memory_allocated`` per the plan's method);
* FP16 (autocast) toggle smoke -- runs without error and stays finite.

The 200-step decrease / no-NaN tests use a small topology (b2c32@9, still
batch 128) so the suite stays fast; the real b10c128@19 batch-128 run with
the full 200 steps is exercised by ``omigamax.cli.smoke_net`` (the plan's
acceptance command).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from omigamax.network.model import create_model
from omigamax.train.loss import (
    agz_loss,
    make_sgd_optimizer,
    policy_cross_entropy,
    train_step,
    value_mse,
    weight_l2,
)

# Batch-128 19x19 default topology (from config/default.yaml).
B10C128 = (10, 128, 19)
# Small topology for the fast 200-step tests (still batch 128).
SMALL = (2, 32, 9)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GIB = 1024**3


def synthetic_batch(batch: int, blocks: int, channels: int, board: int):
    """Build (inputs, pi, z) synthetic smoke data on the test device.

    Matches the CLI recipe (plan "随机数据"): random inputs, a random one-hot
    policy label per sample, random ``{-1, +1}`` value outcome.
    """
    n_logits = board * board + 1
    inputs = torch.randn(batch, 17, board, board, device=DEVICE)
    target_idx = torch.randint(0, n_logits, (batch,), device=DEVICE)
    pi = torch.zeros(batch, n_logits, device=DEVICE)
    pi[torch.arange(batch), target_idx] = 1.0
    z = torch.randint(0, 2, (batch, 1), device=DEVICE).float() * 2.0 - 1.0
    return inputs, pi, z


# ---------------------------------------------------------------------------
# loss-function correctness (analytic, CPU -- fast and exact)
# ---------------------------------------------------------------------------

def test_policy_ce_uniform_logits_equals_ln_362():
    logits = torch.zeros(3, 362)
    pi = torch.full((3, 362), 1.0 / 362.0)
    ce = policy_cross_entropy(logits, pi)
    assert math.isclose(float(ce), math.log(362), rel_tol=1e-6)


def test_value_mse_matches_manual():
    value = torch.tensor([[0.5], [-0.25], [1.0]])
    z = torch.tensor([[1.0], [0.5], [-1.0]])
    mse = value_mse(value, z)
    manual = float(((value - z) ** 2).mean())
    assert math.isclose(float(mse), manual, rel_tol=1e-6)


def test_agz_loss_equals_ce_plus_mse():
    torch.manual_seed(0)
    logits = torch.randn(3, 362)
    pi = torch.softmax(torch.randn(3, 362), dim=-1)
    value = torch.randn(3, 1)
    z = torch.rand(3, 1)
    total = agz_loss(logits, value, pi, z)
    ce = policy_cross_entropy(logits, pi)
    mse = value_mse(value, z)
    assert math.isclose(float(total), float(ce) + float(mse), rel_tol=1e-6)


def test_policy_ce_matches_f_cross_entropy_onehot():
    torch.manual_seed(1)
    logits = torch.randn(4, 82)
    target_idx = torch.tensor([0, 81, 40, 41])
    pi = torch.zeros(4, 82)
    pi[torch.arange(4), target_idx] = 1.0
    ours = policy_cross_entropy(logits, pi)
    ref = F.cross_entropy(logits, target_idx)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_weight_l2_monitor_covers_only_ndim_ge_2_params():
    torch.manual_seed(0)
    model = create_model(1, 16, 5)
    manual = torch.zeros(())
    for p in model.parameters():
        if p.ndim >= 2:
            manual = manual + p.detach().pow(2).sum()
    assert torch.allclose(weight_l2(model), manual)
    # there is at least one 1-D param (biases/BN scales) being excluded
    assert any(p.ndim == 1 for p in model.parameters())
    assert float(weight_l2(model)) > 0.0


def test_make_sgd_optimizer_applies_l2_as_weight_decay():
    torch.manual_seed(0)
    model = create_model(1, 16, 5)
    opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    group = opt.param_groups[0]
    assert group["lr"] == 0.2
    assert group["momentum"] == 0.9
    assert group["weight_decay"] == 1e-4  # the plan's l2 regularizer


# ---------------------------------------------------------------------------
# smoke training on the synthetic dataset
# ---------------------------------------------------------------------------

def test_train_step_loss_decreases_over_200_steps():
    torch.manual_seed(0)
    model = create_model(*SMALL).to(DEVICE)
    optimizer = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    inputs, pi, z = synthetic_batch(128, *SMALL)
    losses = [train_step(model, optimizer, inputs, pi, z)
              for _ in range(200)]
    assert all(math.isfinite(l) for l in losses)
    assert losses[-1] < losses[0] * 0.8, (
        f"loss did not decrease measurably: first={losses[0]} last={losses[-1]}"
    )


def test_backward_produces_no_nan_gradients():
    torch.manual_seed(1)
    model = create_model(*SMALL).to(DEVICE)
    optimizer = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    inputs, pi, z = synthetic_batch(128, *SMALL)
    for _ in range(20):
        train_step(model, optimizer, inputs, pi, z)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has NaN/inf gradient"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_memory_batch128_b10c128_within_5_5gb():
    """Plan gate: peak ``max_memory_allocated`` <= 5.5 GB @ batch 128 b10c128."""
    torch.manual_seed(2)
    model = create_model(*B10C128).cuda()
    optimizer = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    inputs, pi, z = synthetic_batch(128, *B10C128)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):  # step 1 allocates optimizer state; measure over both
        train_step(model, optimizer, inputs, pi, z)
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / GIB
    assert peak_gb <= 5.5, (
        f"peak {peak_gb:.3f} GB > 5.5 GB budget at batch=128 b10c128"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_autocast_toggle_smoke():
    """FP16 (autocast) toggle: runs without error and stays finite on CUDA."""
    torch.manual_seed(3)
    model = create_model(*SMALL).cuda()
    optimizer = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    inputs, pi, z = synthetic_batch(128, *SMALL)
    losses = [train_step(model, optimizer, inputs, pi, z, use_fp16=True)
              for _ in range(20)]
    assert all(math.isfinite(l) for l in losses)
