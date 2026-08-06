"""Tests for the ResNet policy-value network (todo 6).

Covers, per the plan's todo-6 acceptance criteria:
  * forward shapes: (B, 17, N, N) -> policy (B, N**2 + 1), value (B, 1)
    (b10c128 -> policy (1/2, 362), value (1/2, 1))
  * value tanh bounded in [-1, 1], policy logits finite
  * forward + backward produces no NaN gradients
  * determinism under a fixed seed (same weights, same outputs)
  * parameter count ~3M for b10c128 (plan expectation)
  * save -> load -> forward identical to before saving (round-trip consistency)
  * small-board configurable topology (blocks/channels/board_size)
  * create_model_from_config reads blocks/channels/board_size from config
"""

import io

import pytest
import torch
import torch.nn.functional as F

from omigamax.config import load_config
from omigamax.network.model import (
    INPUT_PLANES,
    PolicyValueNetwork,
    create_model,
    create_model_from_config,
    policy_logit_count,
)

B10C128 = (10, 128, 19)  # default topology from config/default.yaml


def make_input(batch: int, planes: int = INPUT_PLANES, board: int = 19) -> torch.Tensor:
    return torch.randn(batch, planes, board, board)


# ---------------------------------------------------------------------------
# forward shapes
# ---------------------------------------------------------------------------

def test_forward_shapes_b10c128():
    model = create_model(*B10C128)
    policy, value = model(make_input(2))
    assert policy.shape == (2, 362)  # 361 intersections + 1 pass
    assert value.shape == (2, 1)
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_forward_single_batch_b10c128():
    model = create_model(*B10C128)
    policy, value = model(make_input(1))
    assert policy.shape == (1, 362)
    assert value.shape == (1, 1)


def test_small_board_configurable_topology():
    # 9x9 -> 81 + 1 = 82 logits; 5x5 -> 25 + 1 = 26 logits.
    model = create_model(blocks=2, channels=32, board_size=9)
    policy, value = model(make_input(1, board=9))
    assert policy.shape == (1, 82)
    assert value.shape == (1, 1)

    model = create_model(blocks=1, channels=16, board_size=5)
    policy, value = model(make_input(2, board=5))
    assert policy.shape == (2, 26)
    assert value.shape == (2, 1)


def test_policy_logit_count_helper():
    assert policy_logit_count(19) == 362
    assert policy_logit_count(9) == 82


# ---------------------------------------------------------------------------
# numerical properties
# ---------------------------------------------------------------------------

def test_value_tanh_bounded_in_minus_one_one():
    model = create_model(blocks=2, channels=32, board_size=9)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _, value = model(make_input(4, board=9))
            assert torch.isfinite(value).all()
            assert value.ge(-1.0).all() and value.le(1.0).all()


def test_policy_logits_finite():
    model = create_model(*B10C128)
    model.eval()
    with torch.no_grad():
        policy, _ = model(make_input(3))
    assert torch.isfinite(policy).all()


def test_forward_backward_no_nan():
    model = create_model(blocks=2, channels=32, board_size=9)
    logits, value = model(make_input(4, board=9))
    target_p = torch.zeros(4, policy_logit_count(9))
    target_p[:, 40] = 1.0
    loss = F.cross_entropy(logits, target_p.argmax(dim=1)) + F.mse_loss(value, torch.ones(4, 1))
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has NaN/inf gradient"


# ---------------------------------------------------------------------------
# determinism / round-trip
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    x = make_input(2)
    torch.manual_seed(1234)
    model_a = create_model(*B10C128)
    pa, va = model_a(x)

    torch.manual_seed(1234)
    model_b = create_model(*B10C128)
    pb, vb = model_b(x)

    for pa_w, pb_w in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa_w, pb_w)
    assert torch.allclose(pa, pb)
    assert torch.allclose(va, vb)


def test_round_trip_state_dict():
    """save -> load -> forward matches the forward before saving (全等)."""
    torch.manual_seed(7)
    model = create_model(*B10C128)
    x = make_input(2)
    p_before, v_before = model(x)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    loaded = create_model(*B10C128)
    loaded.load_state_dict(torch.load(buf, weights_only=True))
    p_after, v_after = loaded(x)

    assert torch.equal(p_before, p_after)
    assert torch.equal(v_before, v_after)


def test_save_method_and_static_load(tmp_path):
    torch.manual_seed(1)
    model = create_model(*B10C128)
    x = make_input(2)
    p_before, v_before = model(x)

    path = tmp_path / "nested" / "model.pt"
    model.save(path)
    assert path.exists()

    loaded = PolicyValueNetwork.load(path, *B10C128)
    p_after, v_after = loaded(x)
    assert torch.equal(p_before, p_after)
    assert torch.equal(v_before, v_after)


# ---------------------------------------------------------------------------
# parameter count (plan expectation: ~3M for b10c128)
# ---------------------------------------------------------------------------

def test_parameter_count_b10c128():
    model = create_model(*B10C128)
    n = sum(p.numel() for p in model.parameters())
    # Plan: b10c128 is "~3M 参数"; the exact count is ~3.28M.
    assert 2_900_000 <= n <= 3_600_000, f"param count {n} outside ~3M band"


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------

def test_create_model_from_config():
    cfg = {"blocks": 4, "channels": 32, "board_size": 9}
    model = create_model_from_config(cfg)
    assert isinstance(model, torch.nn.Module)
    assert model.blocks == 4 and model.channels == 32 and model.board_size == 9
    policy, value = model(make_input(1, board=9))
    assert policy.shape == (1, 82)
    assert value.shape == (1, 1)


def test_create_model_from_default_config_is_b10c128():
    cfg = load_config()
    model = create_model_from_config()
    assert model.blocks == cfg["blocks"] == 10
    assert model.channels == cfg["channels"] == 128
    assert model.board_size == cfg["board_size"] == 19


def test_create_model_rejects_bad_arguments():
    with pytest.raises(ValueError):
        create_model(blocks=0, channels=128, board_size=19)
    with pytest.raises(ValueError):
        create_model(blocks=10, channels=0, board_size=19)
    with pytest.raises(ValueError):
        create_model(blocks=10, channels=128, board_size=1)
