"""P12: multi-process parallel self-play (``--selfplay-workers``) tests.

Covered (per the P12 plan):

  (a) determinism -- ``workers=2`` vs ``workers=1`` produce the SAME npz set
      for the same seeds on a tiny 5x5 net (games are independent per seed, so
      the strided worker slices add up to the identical set; loaded arrays are
      compared for exact equality);
  (b) the ``workers`` parameter defaults to 1 and ``run_loop``/CLI thread it
      through (``run_loop(selfplay_workers=N)`` sends one batched
      ``generate_games(..., workers=N)`` call; the loop CLI parser exposes
      ``--selfplay-workers``);
  (c) worker-count validation -- ``workers <= 0`` and ``workers > 3`` are
      rejected with ``ValueError`` (6GB GPU cap, ~1.4GB fp16 per worker);
  (d) the aggregate report shape for a ``workers>1`` batch (``workers`` and
      ``per_worker`` breakdown present; games/sims add up).

The tiny 1x4/5x5 network runs on CPU (the worker device mirrors the parent
network's device), so the suite is fast and fully deterministic.
"""

import inspect
import json

import numpy as np
import pytest
import torch

from omigamax.config import load_config
from omigamax.network.model import create_model
import omigamax.train.loop as loop
import omigamax.train.selfplay as sp

SIZE = 5
KWARGS = dict(size=SIZE, simulations=8, dirichlet_alpha=0.0)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return create_model(blocks=1, channels=4, board_size=SIZE)


# ---------------------------------------------------------------------------
# (a) determinism: workers=2 == workers=1 npz set (same seeds)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_workers2_matches_workers1_npz_set(self, net, cfg, tmp_path):
        d1 = tmp_path / "w1"
        d2 = tmp_path / "w2"
        rep1, rec1 = sp.generate_games(
            net, cfg, games=2, data_dir=d1, keep=10, seed=0, **KWARGS)
        rep2, rec2 = sp.generate_games(
            net, cfg, games=2, data_dir=d2, keep=10, seed=0,
            workers=2, **KWARGS)
        assert rep1["games"] == rep2["games"] == 2
        assert {p.name for p in d1.glob("*.npz")} == \
            {p.name for p in d2.glob("*.npz")} == \
            {"game_0000000000.npz", "game_0000000001.npz"}
        by_seed1 = {int(r["seed"]): r for r in rec1}
        # same seeds -> same data, in both the in-memory records and the npz
        for r in rec2:
            ref = by_seed1[int(r["seed"])]
            np.testing.assert_array_equal(r["features"], ref["features"])
            np.testing.assert_array_equal(r["pi"], ref["pi"])
            np.testing.assert_array_equal(r["z"], ref["z"])
            assert r["move_actions"] == ref["move_actions"]
            assert r["winner"] == ref["winner"]
            a = np.load(d2 / f"game_{r['seed']:010d}.npz")
            b = np.load(d1 / f"game_{r['seed']:010d}.npz")
            np.testing.assert_array_equal(a["s"], b["s"])
            np.testing.assert_array_equal(a["pi"], b["pi"])
            np.testing.assert_array_equal(a["z"], b["z"])

    def test_workers3_matches_workers1(self, net, cfg, tmp_path):
        d1 = tmp_path / "w1"
        d3 = tmp_path / "w3"
        _, rec1 = sp.generate_games(
            net, cfg, games=3, data_dir=d1, keep=10, seed=5, **KWARGS)
        _, rec3 = sp.generate_games(
            net, cfg, games=3, data_dir=d3, keep=10, seed=5,
            workers=3, **KWARGS)
        by_seed1 = {int(r["seed"]): r for r in rec1}
        assert {int(r["seed"]) for r in rec3} == {5, 6, 7}
        for r in rec3:
            ref = by_seed1[int(r["seed"])]
            np.testing.assert_array_equal(r["features"], ref["features"])
            np.testing.assert_array_equal(r["pi"], ref["pi"])


# ---------------------------------------------------------------------------
# (b) default 1 + loop/CLI plumbing
# ---------------------------------------------------------------------------

class TestDefaultsAndPlumbing:
    def test_workers_defaults_to_one(self):
        sig = inspect.signature(sp.generate_games)
        assert sig.parameters["workers"].default == 1

    def test_loop_parser_exposes_selfplay_workers(self):
        parser = loop._build_parser()
        args = parser.parse_args(["--selfplay-workers", "2"])
        assert args.selfplay_workers == 2
        args = parser.parse_args([])
        assert args.selfplay_workers == 1

    def test_run_loop_batches_with_workers(self, tmp_path, monkeypatch):
        """workers>1: run_loop sends ONE batched generate_games(workers=N)
        call for the whole cycle instead of N per-game calls; viz still gets a
        buffer-refresh frame via push_selfplay_frame (which returns False when
        viz is off) and per-move frames are dropped."""
        calls: list[dict] = []

        def fake_gen(network, cfg, games, data_dir, keep, seed, simulations,
                     **kw):
            calls.append({"games": games, "seed": seed, "kw": kw})
            return {"games": int(games), "sims": int(games) * 30 * simulations,
                    "positions": int(games) * 30, "wall_time_s": 1.0,
                    "sims_per_sec": 1.0, "data_dir": str(data_dir)}, []

        def fake_train(model, optimizer, buffer, steps, **kwargs):
            return [0.5], [0.2], int(kwargs["global_step"]) + 1, kwargs["rng"]

        def fake_eval(candidate_path, best_path, cfg, **kw):
            return {"replaced_best": False,
                    "match": {"winrate": 0.5, "candidate_wins": 1, "games": 2},
                    "elo_update": {"elo": 3.0}}

        monkeypatch.setattr(loop, "generate_games", fake_gen)
        monkeypatch.setattr(loop, "train_steps", fake_train)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_eval)
        cfg = dict(load_config())
        cfg.update({"board_size": 5, "blocks": 1, "channels": 4,
                    "simulations": 4, "eval_games": 2,
                    "eval_sims": 4, "eval_interval_steps": 2000,
                    "replace_threshold": 0.55, "batch_size": 8,
                    "cycle_steps": 1, "cycle_games": 4, "lr": 0.1})
        report = loop.run_loop(
            cfg, device=torch.device("cpu"),
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=4, steps_per_cycle=1,
            selfplay_workers=2, viz_enabled=False,
        )
        # one batched call for the whole cycle, carrying workers=2
        assert len(calls) == 1
        assert calls[0]["games"] == 4
        assert calls[0]["kw"]["workers"] == 2
        # the selfplay-workers value is reported in the protocol
        assert report["protocol"]["selfplay_workers"] == 2

    def test_run_loop_default_keeps_per_game_calls(self, tmp_path, monkeypatch):
        """workers=1 (default) keeps today's per-game generate_games(games=1)
        loop (byte-identical behavior)."""
        calls: list[int] = []

        def fake_gen(network, cfg, games, data_dir, keep, seed, simulations,
                     **kw):
            calls.append(int(games))
            return {"games": int(games), "sims": int(games) * 30 * simulations,
                    "positions": int(games) * 30, "wall_time_s": 1.0,
                    "sims_per_sec": 1.0, "data_dir": str(data_dir)}, []

        def fake_train(model, optimizer, buffer, steps, **kwargs):
            return [0.5], [0.2], int(kwargs["global_step"]) + 1, kwargs["rng"]

        def fake_eval(candidate_path, best_path, cfg, **kw):
            return {"replaced_best": False,
                    "match": {"winrate": 0.5, "candidate_wins": 1, "games": 2},
                    "elo_update": {"elo": 3.0}}

        monkeypatch.setattr(loop, "generate_games", fake_gen)
        monkeypatch.setattr(loop, "train_steps", fake_train)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_eval)
        cfg = dict(load_config())
        cfg.update({"board_size": 5, "blocks": 1, "channels": 4,
                    "simulations": 4, "eval_games": 2,
                    "eval_sims": 4, "eval_interval_steps": 2000,
                    "replace_threshold": 0.55, "batch_size": 8,
                    "cycle_steps": 1, "cycle_games": 4, "lr": 0.1})
        loop.run_loop(
            cfg, device=torch.device("cpu"),
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=4, steps_per_cycle=1,
            selfplay_workers=1, viz_enabled=False,
        )
        assert calls == [1, 1, 1, 1]  # today's per-game loop

    def test_selfplay_cli_parser_accepts_workers(self):
        parser = sp._build_parser()
        args = parser.parse_args(["--selfplay-workers", "3"])
        assert args.selfplay_workers == 3
        assert parser.parse_args([]).selfplay_workers == 1

    def test_selfplay_cli_threads_workers_into_evidence(self, tmp_path):
        """CLI end-to-end: --selfplay-workers 2 spawns the parallel path and
        the evidence JSON records the worker count."""
        data_dir = tmp_path / "data"
        evidence = tmp_path / "ev.json"
        rc = sp.main([
            "--games", "1", "--simulations", "5", "--board-size", "5",
            "--data-dir", str(data_dir), "--keep-games", "1",
            "--seed", "0", "--selfplay-workers", "2",
            "--evidence", str(evidence), "--no-log",
        ])
        assert rc == 0
        assert list(data_dir.glob("*.npz"))
        result = json.loads(evidence.read_text(encoding="utf-8"))
        assert result["protocol"]["selfplay_workers"] == 2
        assert result["report"]["workers"] == 2


# ---------------------------------------------------------------------------
# (c) worker-count validation
# ---------------------------------------------------------------------------

class TestWorkerValidation:
    @pytest.mark.parametrize("bad", [0, -1, -3])
    def test_nonpositive_rejected(self, net, cfg, tmp_path, bad):
        with pytest.raises(ValueError):
            sp.generate_games(
                net, cfg, games=1, data_dir=tmp_path, keep=10, seed=0,
                workers=bad, **KWARGS)

    @pytest.mark.parametrize("bad", [4, 10])
    def test_above_cap_rejected(self, net, cfg, tmp_path, bad):
        with pytest.raises(ValueError):
            sp.generate_games(
                net, cfg, games=1, data_dir=tmp_path, keep=10, seed=0,
                workers=bad, **KWARGS)

    def test_run_loop_rejects_out_of_range(self, tmp_path):
        cfg = dict(load_config())
        with pytest.raises(ValueError):
            loop.run_loop(
                cfg, device=torch.device("cpu"),
                data_dir=tmp_path / "data",
                checkpoint_dir=tmp_path / "models",
                selfplay_workers=4, steps_per_cycle=0, cycles=0)


# ---------------------------------------------------------------------------
# (d) aggregate report shape for workers>1
# ---------------------------------------------------------------------------

class TestAggregateReport:
    def test_report_shape_and_totals(self, net, cfg, tmp_path):
        data_dir = tmp_path / "data"
        report, records = sp.generate_games(
            net, cfg, games=2, data_dir=data_dir, keep=10, seed=0,
            workers=2, **KWARGS)
        assert report["workers"] == 2
        assert report["games"] == 2
        assert report["sims"] == sum(r["sims"] for r in records)
        assert report["positions"] == sum(r["move_count"] for r in records)
        assert report["wall_time_s"] > 0
        assert report["sims_per_sec"] > 0
        assert len(report["per_worker"]) == 2
        for ws in report["per_worker"]:
            assert set(ws) == {"worker_index", "games", "positions", "sims",
                               "wall_time_s", "sims_per_sec"}
        assert {ws["worker_index"] for ws in report["per_worker"]} == {0, 1}
        assert sum(ws["games"] for ws in report["per_worker"]) == 2
        assert sum(ws["sims"] for ws in report["per_worker"]) == report["sims"]
        assert len(report["npz_files"]) == 2
        # per-worker slices are the strided distribution of the batch
        assert report["per_worker"][0]["worker_index"] == 0
        assert [int(r["seed"]) for r in records] == [0, 1]  # seed-sorted

    def test_workers1_report_unchanged_shape(self, net, cfg, tmp_path):
        """workers=1 (default) keeps the pre-P12 report keys only."""
        report, _ = sp.generate_games(
            net, cfg, games=1, data_dir=tmp_path / "data", keep=10, seed=0,
            **KWARGS)
        assert "workers" not in report
        assert "per_worker" not in report
        for key in ("games", "positions", "sims", "wall_time_s",
                    "sims_per_sec", "positions_per_sec", "games_per_hour",
                    "data_dir", "keep_games", "pruned",
                    "simulations_per_move", "npz_files"):
            assert key in report


# ---------------------------------------------------------------------------
# frame_callback with workers>1 (documented viz limitation)
# ---------------------------------------------------------------------------

class TestFrameCallbackWithWorkers:
    def test_workers1_forwards_callback(self, net, cfg, tmp_path):
        seen: list[int] = []

        def cb(board, move_number, color):
            seen.append(move_number)

        sp.generate_games(
            net, cfg, games=1, data_dir=tmp_path / "data", keep=10, seed=0,
            frame_callback=cb, **KWARGS)
        assert len(seen) > 0  # per-move callback works on the single-process path

    def test_workers2_ignores_callback_no_crash(self, net, cfg, tmp_path):
        """Per-move frames cannot be streamed from worker processes; the
        callback is simply not forwarded (documented limitation) and the batch
        still completes."""
        seen: list[int] = []

        def cb(board, move_number, color):
            seen.append(move_number)

        report, _ = sp.generate_games(
            net, cfg, games=2, data_dir=tmp_path / "data", keep=10, seed=0,
            workers=2, frame_callback=cb, **KWARGS)
        assert report["games"] == 2
        assert seen == []  # no per-move frames from worker processes
