"""P7: RL 训练循环 b20c256 适配 (CPU-only tests).

Tests the architecture plumbing that lets the RL loop warm-start from the
b20c256 pretrained checkpoint (``models/pretrain.pt``, P5) while keeping
``config/default.yaml`` (b10c128, the acceptance baseline) untouched:

(a) a loaded checkpoint's recorded arch wins over config defaults -- a fake
    b20c256 checkpoint makes the loop build a 20/256/19 model even though the
    config says b10c128;
(b) explicit ``--blocks`` / ``--channels`` / ``--board-size`` overrides beat
    the config for a fresh-init run, without mutating the shared config;
(c) ``models/pretrain.pt`` loads through the RL loop's warm-start path on CPU
    with arch {20,256,19} and exactly 23,962,085 parameters;
(d) the self-play agent net built from the b20c256 arch produces right-sized
    policy/value outputs on a CPU forward (small batch), and a tiny self-play
    game runs on 19x19;
(e) a tiny CPU RL micro-run (9x9, blocks=1, channels=8): 2 self-play games +
    10 train steps -- loss finite and decreasing, well under 3 minutes.

All tests are CPU-only: the GPU is reserved for the detached P6 pretraining
run, and the b20c256 GPU smoke is deferred to a later task.
"""

import math
from pathlib import Path

import pytest
import torch

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train import loop
from omigamax.train.loss import make_sgd_optimizer

PRETRAIN = Path("models/pretrain.pt")
B20C256_PARAMS = 23_962_085
CPU = torch.device("cpu")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _b20c256_checkpoint() -> dict:
    """A genuine b20c256 checkpoint dict (real weights, pre-step SGD state)."""
    model = create_model(20, 256, 19)
    opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
    return {
        "arch": {"blocks": 20, "channels": 256, "board_size": 19},
        "global_step": 42,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }


# ---------------------------------------------------------------------------
# (a) checkpoint arch wins over config defaults
# ---------------------------------------------------------------------------

class TestCheckpointArchWins:
    def test_b20c256_checkpoint_beats_b10c128_config(self, tmp_path, monkeypatch):
        """A loaded checkpoint's recorded arch (20/256/19) wins over the
        config defaults (10/128/19) -- P7's pretrained-start selection."""
        cfg = load_config()
        assert cfg["blocks"] == 10 and cfg["channels"] == 128  # b10c128 baseline
        fake = _b20c256_checkpoint()
        monkeypatch.setattr(loop, "load_checkpoint", lambda *a, **k: fake)
        st = loop._load_or_init(
            cfg, tmp_path / "models" / "latest.pt", CPU, 0,
            resume=False, init_checkpoint=str(PRETRAIN),
        )
        model = st["model"]
        assert (model.blocks, model.channels, model.board_size) == (20, 256, 19)
        assert st["global_step"] == 42
        assert st["resumed"] is False
        assert st["init_checkpoint"] == str(PRETRAIN)


# ---------------------------------------------------------------------------
# (b) explicit CLI flags override config
# ---------------------------------------------------------------------------

class TestCliArchOverrides:
    def test_apply_arch_overrides_maps_and_keeps_config(self):
        cfg = load_config()
        assert cfg["blocks"] == 10 and cfg["board_size"] == 19
        over = loop.apply_arch_overrides(cfg, blocks=20, channels=256,
                                         board_size=19)
        assert (over["blocks"], over["channels"], over["board_size"]) == (
            20, 256, 19)
        # the shared config dict is never mutated
        assert cfg["blocks"] == 10 and cfg["board_size"] == 19

    def test_fresh_init_uses_overridden_arch(self, tmp_path):
        cfg = loop.apply_arch_overrides(load_config(), blocks=2, channels=16,
                                        board_size=9)
        st = loop._load_or_init(cfg, tmp_path / "models" / "latest.pt",
                                CPU, 0, resume=False)
        model = st["model"]
        assert (model.blocks, model.channels, model.board_size) == (2, 16, 9)
        assert st["resumed"] is False and st["init_checkpoint"] is None


# ---------------------------------------------------------------------------
# (c) pretrain.pt loads in the RL loop (param count on CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PRETRAIN.exists(),
                    reason="models/pretrain.pt (P5 artifact) not present")
class TestPretrainLoadsInRL:
    def test_pretrain_pt_warm_start_param_count(self):
        st = loop._load_or_init(
            load_config(), "models/does-not-exist-latest.pt", CPU, 0,
            resume=False, init_checkpoint=str(PRETRAIN),
        )
        model = st["model"]
        assert (model.blocks, model.channels, model.board_size) == (20, 256, 19)
        # the checkpoint's recorded global_step is loaded (the P5 smoke
        # artifact recorded 200; the real 60k-step run records 60000 -- assert
        # against the checkpoint itself, not a pinned step count)
        from omigamax.train.train import load_checkpoint
        assert st["global_step"] == int(load_checkpoint(PRETRAIN)["global_step"])
        assert st["resumed"] is False
        assert st["init_checkpoint"] == str(PRETRAIN)
        n = sum(p.numel() for p in model.parameters())
        assert n == B20C256_PARAMS

    def test_pretrain_pt_optimizer_state_compatible(self):
        """SGD momentum state from the SL run loads into the RL loop's SGD
        optimizer (same make_sgd_optimizer recipe)."""
        cfg = load_config()
        st = loop._load_or_init(
            cfg, "models/does-not-exist-latest.pt", CPU, 0,
            resume=False, init_checkpoint=str(PRETRAIN),
        )
        opt = st["optimizer"]
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["momentum"] == float(cfg.get("momentum", 0.9))
        # one tiny optimizer step proves the restored momentum buffers work
        model = st["model"]
        x = torch.randn(2, 17, 19, 19)
        pi = torch.rand(2, 362)
        pi = pi / pi.sum(dim=-1, keepdim=True)
        z = torch.ones(2, 1)
        from omigamax.train.loss import agz_loss
        opt.zero_grad(set_to_none=True)
        logits, value = model(x)
        loss = agz_loss(logits, value, pi, z)
        loss.backward()
        opt.step()
        assert math.isfinite(float(loss.detach()))


# ---------------------------------------------------------------------------
# (d) self-play agent construction from the b20c256 arch (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PRETRAIN.exists(),
                    reason="models/pretrain.pt (P5 artifact) not present")
class TestSelfplayFromPretrainArch:
    def test_selfplay_net_forward_shapes(self):
        st = loop._load_or_init(
            load_config(), "models/does-not-exist-latest.pt", CPU, 0,
            resume=False, init_checkpoint=str(PRETRAIN),
        )
        net = st["model"]
        net.eval()
        x = torch.randn(1, 17, 19, 19)   # AGZ 17 planes, small batch, CPU
        with torch.no_grad():
            pi, v = net(x)
        assert pi.shape == (1, 19 * 19 + 1)   # (1, 362) policy logits
        assert v.shape == (1, 1)              # scalar value
        assert torch.isfinite(pi).all() and torch.isfinite(v).all()
        assert sum(p.numel() for p in net.parameters()) == B20C256_PARAMS

    def test_selfplay_game_runs_on_pretrain_net(self):
        """A tiny self-play game on 19x19 through the b20c256 net: the agent
        (MCTS + net inference) builds the board/feature tensors from the net's
        own arch, so (s, pi) come out at the right size."""
        from omigamax.train.selfplay import play_game
        st = loop._load_or_init(
            load_config(), "models/does-not-exist-latest.pt", CPU, 0,
            resume=False, init_checkpoint=str(PRETRAIN),
        )
        rec = play_game(st["model"], load_config(), simulations=1,
                        max_moves=4, temperature_threshold=0, seed=0)
        assert rec["features"].shape[1] == 17
        assert rec["features"].shape[2:] == (19, 19)
        assert rec["pi"].shape[1] == 362
        assert rec["z"].shape[0] == rec["features"].shape[0]


# ---------------------------------------------------------------------------
# (e) tiny CPU RL micro-run: 2 self-play games + 10 train steps
# ---------------------------------------------------------------------------

class TestMicroRLRun:
    def test_micro_run_loss_finite_and_decreases(self, tmp_path):
        """9x9, blocks=1, channels=8, self-play 2 games + 10 train steps on
        CPU: loss finite and decreasing, complete well under 3 minutes."""
        cfg = load_config()
        cfg.update({
            "board_size": 9, "blocks": 1, "channels": 8,
            "batch_size": 8, "simulations": 4,
            "temperature_threshold": 0, "selfplay_max_moves": 60,
            "viz_enabled": False, "lr": 0.1,
        })
        report = loop.run_loop(
            cfg, device=CPU,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl", history=tmp_path / "hist.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=10,
            batch_size=8, use_symmetry=True, seed=0,
            viz_enabled=False, interrupt_after=10,
        )
        L = report["loop"]
        assert L["interrupted"] is True          # interrupt path, no eval gate
        assert L["steps_trained"] == 10
        assert L["global_step_final"] == 10
        assert math.isfinite(L["loss_first"]) and math.isfinite(L["loss_last"])
        assert L["loss_decrease"] is True        # loss_last < loss_first
        assert report["protocol"]["board_size"] == 9
        assert report["checkpoint"]["latest_exists"] is True
