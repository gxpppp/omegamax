"""P4: b20c256 architecture verification (supervised-pretraining target net).

The pretraining CLI (P5) will drive this architecture via explicit args --
``create_model(blocks=20, channels=256, board_size=19)`` -- NOT by changing
``config/default.yaml`` (whose b10c128 topology remains the default). These
tests lock the ground-truth numbers measured on this machine (RTX 3060
Laptop GPU, 6 GB):

* parameter count: 23,962,085 -- a deliberate, user-approved architecture
  upgrade from the b10c128 default (3,282,661); ~8.5x the parameters,
  still inside the plan's "~3M family scaled up" 10M-30M band;
* forward pass: policy ``(1, 362)``, value ``(1, 1)``, all finite;
* ``state_dict`` save/load round-trip: ``torch.equal`` exact;
* peak GPU memory of a batch-64 forward+backward (with optimizer state,
  mirroring the todo-8 protocol) stays <= 5.5 GB -- measured 2.149 GB at
  batch 64 (and 4.143 GB at batch 128, both within budget).

Measured on 2026-08-08, device ``NVIDIA GeForce RTX 3060 Laptop GPU``
(5.999 GiB). Evidence: ``.omo/evidence/omigamax-go/task-P4-arch.txt``.
"""

import math

import pytest
import torch

from omigamax.network.model import create_model
from omigamax.train.loss import make_sgd_optimizer, train_step

# b20c256 @ 19x19: the supervised-pretraining target architecture.
B20C256 = (20, 256, 19)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GIB = 1024**3

# Exact measured value; kept in range so accidental drift is caught either way.
EXACT_PARAM_COUNT = 23_962_085


def synthetic_batch(batch: int, blocks: int, channels: int, board: int):
    """Build (inputs, pi, z) synthetic smoke data on the test device.

    Matches the todo-8 smoke recipe ("随机数据"): random inputs, a random
    one-hot policy label per sample, random ``{-1, +1}`` value outcome.
    """
    n_logits = board * board + 1
    inputs = torch.randn(batch, 17, board, board, device=DEVICE)
    target_idx = torch.randint(0, n_logits, (batch,), device=DEVICE)
    pi = torch.zeros(batch, n_logits, device=DEVICE)
    pi[torch.arange(batch), target_idx] = 1.0
    z = torch.randint(0, 2, (batch, 1), device=DEVICE).float() * 2.0 - 1.0
    return inputs, pi, z


# ---------------------------------------------------------------------------
# parameter count / forward sanity (CPU -- fast and exact)
# ---------------------------------------------------------------------------

def test_b20c256_parameter_count_in_range():
    """P4 gate: b20c256 must stay inside the planned 10M-30M band."""
    model = create_model(*B20C256)
    n = sum(p.numel() for p in model.parameters())
    assert 10_000_000 <= n <= 30_000_000, f"param count {n} outside 10M-30M band"
    assert n == EXACT_PARAM_COUNT, (
        f"param count {n} drifted from measured {EXACT_PARAM_COUNT}"
    )


def test_b20c256_forward_shapes_and_finite():
    """Forward pass emits policy (1, 362) and value (1, 1), all finite."""
    torch.manual_seed(0)
    model = create_model(*B20C256).eval()
    x = torch.randn(1, 17, 19, 19)
    with torch.no_grad():
        logits, value = model(x)
    assert tuple(logits.shape) == (1, 362)
    assert tuple(value.shape) == (1, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_b20c256_state_dict_round_trip(tmp_path):
    """save() -> load() reproduces an exactly equal state_dict."""
    torch.manual_seed(1)
    model = create_model(*B20C256)
    path = tmp_path / "b20c256.pt"
    model.save(path)
    loaded = create_model(*B20C256)
    state = torch.load(path, map_location="cpu", weights_only=True)
    loaded.load_state_dict(state)
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), loaded.state_dict().values())
    )


# ---------------------------------------------------------------------------
# GPU memory gate (mirrors test_smoke_net protocol)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_memory_batch64_b20c256_within_5_5gb():
    """P4 gate: peak ``max_memory_allocated`` <= 5.5 GB @ batch 64 b20c256.

    Mirror of the todo-8 memory protocol: measure forward+backward with the
    optimizer (SGD momentum + L2 weight decay) -- step 1 allocates the
    optimizer state, so peak is captured over the first few steps.
    """
    torch.manual_seed(2)
    model = create_model(*B20C256).cuda()
    optimizer = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    inputs, pi, z = synthetic_batch(64, *B20C256)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        train_step(model, optimizer, inputs, pi, z)
    peak_gb = torch.cuda.max_memory_allocated() / GIB
    assert math.isfinite(peak_gb)
    assert peak_gb <= 5.5, (
        f"peak {peak_gb:.3f} GB > 5.5 GB budget at batch=64 b20c256"
    )
