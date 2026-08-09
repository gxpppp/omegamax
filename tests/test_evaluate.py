"""Tests for the evaluation gate and ELO recording (todo 15).

Per the plan's todo-15 acceptance criteria:

* **gate logic** -- fabricated win counts decide replace/keep exactly at the
  ``replace_threshold`` = 0.55 boundary: 12/21 = 0.5714 replaces, 11/21 =
  0.5238 keeps, and 11/20 = 0.55 replaces (the plan's 含等于替换 -- the
  boundary is included);
* **ELO formula** -- the standard rating difference
  ``400 * log10(p / (1 - p))`` (plan References: Wikipedia Elo) on known win
  rates: ``0.5 -> 0``, ``0.55 -> ~34.9``, ``0.75 -> ~190.9`` (note: a
  constant-200 variant would give ~95.3 at p=0.75; the plan's standard-400
  formula is what is implemented here); plus the K=32 running update
  ``R' = R + 32 * (score - E(R))`` with ``E(0) = 0.5``;
* **no-noise / tau=0 discipline** -- a mock of ``run_search`` /
  ``sample_action`` proves the evaluator never applies Dirichlet root noise
  and always selects moves at ``tau = 0`` (argmax);
* **bootstrap** -- the first evaluation writes random-init weights to a
  missing ``best.pt`` and re-uses them as the baseline opponent afterwards
  (plan Oracle G1);
* **end-to-end** -- a short real evaluation (tiny differently-seeded nets,
  low sims) plays full legal games and reports a consistent win rate, ELO and
  gate decision; and the full ``evaluate_and_gate`` orchestration writes
  ``best.pt`` (replace or bootstrap) plus a JSONL history entry.
"""

import json

import pytest
import torch

from omigamax.config import load_config
from omigamax.network.features import pass_index
from omigamax.network.model import create_model
from omigamax.train import evaluate as ev
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.train import save_checkpoint

SIZE = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _net(seed: int = 0, blocks: int = 2, channels: int = 16, size: int = SIZE):
    torch.manual_seed(seed)
    return create_model(blocks=blocks, channels=channels, board_size=size).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

class TestGate:
    def test_12_of_21_replaces(self):
        # 12/21 = 0.5714 >= 0.55 -> replace
        assert ev.gate_decision(12, 21, 0.55) is True

    def test_11_of_21_keeps(self):
        # 11/21 = 0.5238 < 0.55 -> keep
        assert ev.gate_decision(11, 21, 0.55) is False

    def test_equality_boundary_replaces(self):
        # 11/20 = 0.55 == threshold -> replace (plan: 含等于替换)
        assert ev.gate_decision(11, 20, 0.55) is True

    def test_zero_wins_keeps(self):
        assert ev.gate_decision(0, 21, 0.55) is False

    def test_all_wins_replaces(self):
        assert ev.gate_decision(21, 21, 0.55) is True

    def test_config_threshold_used(self):
        cfg = load_config()
        assert float(cfg["replace_threshold"]) == pytest.approx(0.55)
        assert ev.gate_decision(12, 21, float(cfg["replace_threshold"])) is True
        assert ev.gate_decision(11, 21, float(cfg["replace_threshold"])) is False

    def test_invalid_games_keeps(self):
        assert ev.gate_decision(5, 0, 0.55) is False


# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------

class TestElo:
    def test_winrate_half_is_zero(self):
        assert ev.elo_from_winrate(0.5) == 0.0

    def test_winrate_055(self):
        # 400*log10(0.55/0.45) = 400*log10(1.2222) ~ 34.86
        assert ev.elo_from_winrate(0.55) == pytest.approx(34.862, abs=0.1)

    def test_winrate_075(self):
        # standard Elo: 400*log10(0.75/0.25) = 400*log10(3) ~ 190.85
        assert ev.elo_from_winrate(0.75) == pytest.approx(190.85, abs=0.1)

    def test_symmetric(self):
        assert ev.elo_from_winrate(0.25) == pytest.approx(-ev.elo_from_winrate(0.75))

    def test_clamped_at_extremes(self):
        # all-wins / all-losses -> +/-2400, never inf
        assert ev.elo_from_winrate(1.0) == pytest.approx(2400.0, abs=1.0)
        assert ev.elo_from_winrate(0.0) == pytest.approx(-2400.0, abs=1.0)

    def test_expected_score_equal_ratings(self):
        assert ev.expected_score(0.0, 0.0) == pytest.approx(0.5)

    def test_update_elo_k32(self):
        # K=32: R' = R + 32*(score - 0.5) at equal ratings
        assert ev.update_elo(0.0, 0.5) == pytest.approx(0.0)
        assert ev.update_elo(0.0, 0.75) == pytest.approx(8.0)
        assert ev.update_elo(0.0, 0.0) == pytest.approx(-16.0)

    def test_update_elo_tracks(self):
        # a win-rate above expectation pushes the rating up
        r1 = ev.update_elo(0.0, 0.6)
        r2 = ev.update_elo(r1, 0.6)
        assert r2 > r1 > 0.0


# ---------------------------------------------------------------------------
# evaluation discipline: no noise, tau=0 (probed via mocks)
# ---------------------------------------------------------------------------

class TestEvaluationDiscipline:
    def test_no_dirichlet_noise_and_tau_zero(self, monkeypatch):
        black = _net(1)
        white = _net(2)
        captured = {"dirichlet_alphas": [], "temperatures": []}

        class _FakeRoot:
            def __init__(self, size):
                self.size = size

        def fake_run_search(root, network, simulations, **kwargs):
            captured["dirichlet_alphas"].append(kwargs.get("dirichlet_alpha"))
            return root

        def fake_sample_action(root, temperature, rng=None):
            captured["temperatures"].append(float(temperature))
            return pass_index(root.board.size)

        monkeypatch.setattr(ev, "run_search", fake_run_search)
        monkeypatch.setattr(ev, "sample_action", fake_sample_action)

        rec = ev.play_eval_game(black, white, sims=4, size=SIZE, komi=7.5,
                                seed=3, virtual_loss=3)

        # both sides searched at least once and the game finished legally
        assert len(captured["dirichlet_alphas"]) >= 2
        assert rec["winner"] in ("B", "W")
        # the evaluator never applies Dirichlet root noise (always None)
        assert all(alpha is None for alpha in captured["dirichlet_alphas"])
        # every move is selected at tau = 0 (argmax)
        assert len(captured["temperatures"]) == len(captured["dirichlet_alphas"])
        assert all(t == 0.0 for t in captured["temperatures"])

    def test_eval_games_never_written_to_buffer(self, monkeypatch):
        """Evaluation must not touch the replay buffer (plan Must-NOT)."""
        # play_eval_game takes only networks -- no data_dir/keep anywhere in
        # the evaluate module's game path.
        src = open(ev.__file__, "r", encoding="utf-8").read()
        assert "data/selfplay" not in src
        assert "save_game_npz" not in src


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_first_eval_bootstraps_random_best(self, tmp_path):
        cfg = load_config()
        arch = {"blocks": 1, "channels": 8, "board_size": SIZE}
        best_path = tmp_path / "best.pt"
        assert not best_path.exists()

        model, bootstrapped = ev.ensure_best_model(best_path, arch, cfg, DEVICE)
        assert bootstrapped is True
        assert best_path.exists()

        # a second call loads the existing baseline (not a fresh one)
        model2, bootstrapped2 = ev.ensure_best_model(best_path, arch, cfg, DEVICE)
        assert bootstrapped2 is False
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1.detach().cpu(), p2.detach().cpu())


# ---------------------------------------------------------------------------
# a short real evaluation, end to end
# ---------------------------------------------------------------------------

class TestShortRealEvaluation:
    def test_full_evaluation_gate_end_to_end(self):
        candidate = _net(11)
        best = _net(22)  # differently-seeded tiny nets
        report = ev.run_evaluation(
            candidate, best, load_config(),
            games=3, sims=4, size=SIZE, komi=7.5, virtual_loss=3,
            seed=5, max_moves=120,
        )
        assert report["games"] == 3
        assert report["sims"] == 4
        assert 0 <= report["winrate"] <= 1.0
        assert report["candidate_wins"] in (0, 1, 2, 3)
        assert report["draws"] == 0  # komi 7.5 => no jigo
        # the gate decision is consistent with the win count
        assert report["replaced"] == ev.gate_decision(
            report["candidate_wins"], report["games"], report["threshold"])
        # ELO diff consistent with the win rate
        assert report["elo_diff"] == pytest.approx(
            ev.elo_from_winrate(report["winrate"]), abs=0.001)
        # colours alternate: candidate black on even games, white on odd
        for i, rec in enumerate(report["games_detail"]):
            assert rec["winner"] in ("B", "W")
            assert rec["candidate_color"] == ("B" if i % 2 == 0 else "W")
            assert rec["moves"] > 0
            assert isinstance(rec["result"], str)


class TestEvaluateAndGate:
    def test_full_orchestration_with_history(self, tmp_path):
        torch.manual_seed(0)
        model = create_model(2, 16, SIZE).to(DEVICE)
        opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
        candidate_path = tmp_path / "latest.pt"
        save_checkpoint(candidate_path, model, opt, global_step=7,
                        config={"lr": 0.2, "momentum": 0.9, "l2": 1e-4})
        best_path = tmp_path / "best.pt"
        history = tmp_path / "eval_history.jsonl"

        result = ev.evaluate_and_gate(
            candidate_path, best_path, load_config(),
            games=2, sims=3, size=SIZE, komi=7.5, seed=9,
            device=DEVICE, history_path=history,
        )
        assert best_path.exists()  # bootstrapped
        assert result["protocol"]["bootstrapped_best"] is True
        assert result["protocol"]["candidate_global_step"] == 7
        # best.pt is written iff the gate replaced
        assert (result["best_written"] == str(best_path)) == result["replaced_best"]

        lines = [json.loads(l) for l in
                 history.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "evaluate_gate"
        assert entry["games"] == 2
        assert entry["global_step"] == 7
        assert entry["elo_before"] == 0.0
        assert entry["elo"] == pytest.approx(entry["elo_before"] + entry["elo_delta"])

    def test_read_last_elo_empty_and_after_append(self, tmp_path):
        hp = tmp_path / "eval_history.jsonl"
        assert ev.read_last_elo(hp) == 0.0
        ev.append_eval_history({"event": "x", "elo": 12.5}, hp)
        assert ev.read_last_elo(hp) == 12.5
        ev.append_eval_history({"event": "x", "elo": 7.25}, hp)
        assert ev.read_last_elo(hp) == 7.25


# ---------------------------------------------------------------------------
# P8: acceptance-evaluation CLI (omigamax/cli/evaluate.py) -- CPU-only.
# (a) human-match on a tiny 9x9 model + tiny synthetic chunk -> metrics sane
#     and deterministic under a fixed seed;
# (b) random baseline on the real 19x19 geometry -> top-1 ~ 1/362;
# (c) bench with a tiny config runs end-to-end and returns a win-rate dict;
# (d) the report writer creates the file.
# ---------------------------------------------------------------------------

import math as _math
from pathlib import Path

import numpy as _np

from omigamax.cli import evaluate as p8
from omigamax.train.pretrain import (
    PretrainChunks,
    make_pretrain_optimizer,
    save_pretrain_checkpoint,
)

CPU = torch.device("cpu")


def _p8_synthetic_chunks(dir_path, board, sizes, seed=0):
    """Write tiny valid ``chunk_%04d.npz`` SL samples (test_pretrain-style)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = _np.random.default_rng(seed)
    for i, n in enumerate(sizes):
        s = rng.integers(0, 2, size=(n, 17, board, board)).astype(_np.uint8)
        pi = rng.integers(0, board * board + 1, size=n).astype(_np.uint16)
        z = rng.choice(_np.array([-1, 1], dtype=_np.int8), size=n)
        _np.savez(dir_path / f"chunk_{i:04d}.npz", s=s, pi=pi, z=z)
    return dir_path


def _p8_tiny_checkpoint(path, board, blocks=1, channels=8, seed=0):
    torch.manual_seed(seed)
    model = create_model(blocks, channels, board)
    optimizer = make_pretrain_optimizer(model, lr=0.02, momentum=0.9, l2=1e-4)
    save_pretrain_checkpoint(
        path, model, optimizer, global_step=0,
        rng=_np.random.default_rng(seed),
        config={"blocks": blocks, "channels": channels, "board_size": board},
    )
    return path


class TestP8HumanMatch:
    def test_tiny_run_metrics_sane_and_deterministic(self, tmp_path):
        board = 9
        ckpt = _p8_tiny_checkpoint(tmp_path / "tiny.pt", board)
        data_dir = _p8_synthetic_chunks(tmp_path / "data", board, sizes=[40, 40])
        r1 = p8.run_human_match(ckpt, data_dir, samples=32, seed=12345,
                                device=CPU)
        r2 = p8.run_human_match(ckpt, data_dir, samples=32, seed=12345,
                                device=CPU)
        m = r1["model"]
        # deterministic under a fixed seed
        assert m == r2["model"]
        assert r1["eval_samples"] == 32 and m["n"] == 32
        assert 0.0 <= m["top1"] <= 1.0
        assert 0.0 <= m["top5"] <= 1.0
        assert m["top5"] >= m["top1"]
        assert all(_math.isfinite(m[k]) for k in
                   ("top1", "top5", "policy_ce", "value_mse", "pearson"))
        assert r1["arch"] == {"blocks": 1, "channels": 8, "board_size": board}
        # the random floor is reported alongside
        assert 0.0 <= r1["random_baseline"]["top1"] <= 1.0

    def test_random_baseline_19x19_top1_about_1_over_362(self, tmp_path):
        board = 19
        data_dir = _p8_synthetic_chunks(tmp_path / "data", board, sizes=[2000])
        with PretrainChunks(data_dir) as chunks:
            batch = chunks.sample_batch(_np.random.default_rng(0), 2000)
        base = p8.random_baseline(batch, seed=99)
        assert base["D"] == 362
        assert abs(base["top1"] - 1.0 / 362.0) < 0.01
        assert abs(base["top5"] - 5.0 / 362.0) < 0.02
        assert base["policy_ce"] == pytest.approx(_math.log(362), rel=1e-6)
        assert base["value_mse"] == pytest.approx(1.0, abs=1e-6)  # z in +-1

    def test_evaluate_model_chunking_is_metric_invariant(self, tmp_path,
                                                         monkeypatch):
        """Chunked eval (P8b fix) is bit-identical to a single full-batch pass.

        Regression: the original evaluate_model forwarded the WHOLE eval set
        at once -- a 7.4GB activation tensor at b20c256 on 20k samples, which
        OOMs the 6GB GPU and is pathologically slow on CPU. Metrics must not
        depend on how the batch is sliced into EVAL_CHUNK chunks.
        """
        board = 9
        ckpt = _p8_tiny_checkpoint(tmp_path / "tiny.pt", board)
        data_dir = _p8_synthetic_chunks(tmp_path / "data", board,
                                        sizes=[50, 50])
        with PretrainChunks(data_dir) as chunks:
            batch = chunks.sample_batch(_np.random.default_rng(7), 300)
        model, _arch, _step = p8.load_eval_model(ckpt, CPU)

        monkeypatch.setattr(p8, "EVAL_CHUNK", 300)  # one single chunk
        one = p8.evaluate_model(model, batch, CPU)
        monkeypatch.setattr(p8, "EVAL_CHUNK", 64)   # five chunks
        many = p8.evaluate_model(model, batch, CPU)

        assert many["n"] == one["n"] == 300
        for k in ("top1", "top5", "policy_ce", "value_mse", "pearson"):
            assert many[k] == pytest.approx(one[k], abs=1e-6)


class TestP8Bench:
    def test_tiny_bench_end_to_end_winrate_dict(self, tmp_path):
        board = 9
        ckpt = _p8_tiny_checkpoint(tmp_path / "a.pt", board)
        cfg = {"komi": 7.5, "virtual_loss": 3}
        rep = p8.run_bench(
            ckpt, None, cfg,
            games=2, sims=50, size=None, komi=None, virtual_loss=None,
            max_moves=200, seed=7, device=CPU,
        )
        assert rep["games"] == 2 and rep["sims"] == 50
        assert rep["board_size"] == board and rep["komi"] == 7.5
        assert 0 <= rep["a_wins"] <= 2
        assert rep["a_wins"] + rep["b_wins"] + rep["draws"] == 2
        assert 0.0 <= rep["winrate_a"] <= 1.0
        assert rep["winrate_a"] + rep["winrate_b"] + \
            rep["draws"] / rep["games"] == pytest.approx(1.0)
        assert rep["avg_game_length"] > 0
        assert rep["opponent"]["checkpoint"] is None  # random-init baseline
        assert len(rep["games_detail"]) == 2
        for rec in rep["games_detail"]:
            assert rec["winner"] in ("B", "W", None)
            assert rec["moves"] > 0


class TestP8Report:
    def test_report_writer_creates_file(self, tmp_path):
        human = {
            "mode": "human-match", "checkpoint": "models/pretrain.pt",
            "arch": {"blocks": 1, "channels": 8, "board_size": 9},
            "eval_samples": 4, "data_dir": "data/pretrain", "eval_seed": 1,
            "model": {"n": 4, "top1": 0.25, "top5": 0.5, "policy_ce": 2.3,
                      "value_mse": 0.9, "pearson": 0.1},
            "random_baseline": {"n": 4, "D": 82, "top1": 0.0122,
                                "top5": 0.06, "policy_ce": 4.4,
                                "value_mse": 1.0, "pearson": 0.0},
        }
        bench = {
            "mode": "bench", "checkpoint": "models/pretrain.pt",
            "games": 2, "a_wins": 1, "b_wins": 1, "draws": 0,
            "winrate_a": 0.5, "winrate_b": 0.5,
            "avg_game_length": 120.0, "sims": 50, "board_size": 9,
            "komi": 7.5, "virtual_loss": 3,
            "opponent": {"checkpoint": None, "note": "random-init baseline"},
        }
        out = tmp_path / "report.txt"
        path = p8.write_report(out, human, bench)
        text = Path(path).read_text(encoding="utf-8")
        assert Path(path).exists()
        assert "human-match" in text
        assert "bench" in text
        assert "top-1" in text and "win rate" in text
        assert len(text.strip()) > 50

    def test_report_includes_rl_smoke_section(self, tmp_path):
        human = {"checkpoint": "models/pretrain.pt", "arch": {"blocks": 20,
                 "channels": 256, "board_size": 19}, "model": {"top1": 0.4,
                 "top5": 0.7, "policy_ce": 2.3, "value_mse": 0.23,
                 "pearson": 0.8}}
        bench = {"checkpoint": "models/pretrain.pt", "games": 2, "a_wins": 2,
                 "b_wins": 0, "draws": 0, "winrate_a": 1.0, "winrate_b": 0.0,
                 "avg_game_length": 120.0, "sims": 150, "board_size": 19,
                 "komi": 7.5, "virtual_loss": 3,
                 "opponent": {"checkpoint": None, "note": "random-init"}}
        smoke = {
            "todo": 16, "device": "cuda",
            "init_checkpoint": "models/pretrain.pt",
            "protocol": {"batch_size": 64, "simulations": 100},
            "loop": {"steps_trained": 200, "games_generated": 1,
                     "global_step_final": 60200, "loss_first": 9.77,
                     "loss_last": 4.86, "loss_decrease": True},
            "checkpoint": {"latest_exists": True},
            "arch": {"blocks": 20, "channels": 256, "board_size": 19},
        }
        out = tmp_path / "report.txt"
        path = p8.write_report(out, human, bench, smoke=smoke)
        text = Path(path).read_text(encoding="utf-8")
        assert "[3] RL warm-start smoke" in text
        assert "9.77" in text and "4.86" in text
        assert "60200" in text and "arch blocks=20 channels=256 board=19" in text
        # the P8-code-phase "deferred until P6 finishes" note is gone now that
        # the acceptance run actually executed
        assert "deferred" not in text
        assert "every number above is measured" in text
