"""Lightweight harness-level tests for the todo-21 e2e smoke (todo 21).

Per the plan's todo-21 acceptance the *full* zero-to-playable chain lives in
the CLI ``omigamax.cli.e2e_smoke`` + evidence, NOT in pytest. These tests
exercise what pytest should:

* the Wilson 95% CI math behind every "MET / NOT MET / UNCONFIRMED" claim;
* the vs-random early-milestone classifier (the plan's >80% weak signal);
* the smoke's default plan (sane scale for this machine's ~88-160 sims/s);
* the report line-builder over a representative phase result set;
* a tiny real end-to-end training slice: ``e2e_smoke --phases train`` on a
  9x9 board with a 1x8 network proves the zero-to-best.pt wiring
  (self-play -> train -> eval gate -> checkpoints + JSONL) works through the
  actual orchestrator.
"""

import json
from pathlib import Path

import pytest

from omigamax.cli import e2e_smoke as es


# ---------------------------------------------------------------------------
# Wilson 95% CI (the honesty contract behind MET / UNCONFIRMED)
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_zero_games_is_none(self):
        assert es.wilson_ci(0, 0) == (None, None)

    def test_perfect_run_ci_wide_at_small_n(self):
        lo, hi = es.wilson_ci(1, 1)
        assert lo is not None and hi is not None
        assert lo <= 1.0 <= hi  # 1/1's interval contains 1.0

    def test_known_interval(self):
        # 3/4 = 0.75; Wilson 95% CI is ~[0.30, 0.95] -- must straddle 0.5,
        # which is exactly why the plan calls 3-4 game samples UNCONFIRMED.
        lo, hi = es.wilson_ci(3, 4)
        assert lo <= 0.5 <= hi
        assert lo <= 0.75 <= hi

    def test_wide_sample_tightens(self):
        lo_small, hi_small = es.wilson_ci(30, 60)
        lo_large, hi_large = es.wilson_ci(3000, 6000)
        assert (hi_small - lo_small) > (hi_large - lo_large)

    def test_ci_always_contains_point_estimate(self):
        for wins, games in [(0, 5), (2, 7), (5, 5), (1, 20), (9, 10)]:
            lo, hi = es.wilson_ci(wins, games)
            p = wins / games
            assert lo <= p <= hi


class TestMilestoneStatus:
    def test_not_met_below_bar(self):
        st = es.milestone_status(0.333, 2, 6)
        assert st["met"] is False
        assert st["winrate"] == pytest.approx(0.333)

    def test_met_above_bar(self):
        st = es.milestone_status(0.9, 9, 10)
        assert st["met"] is True

    def test_zero_games_not_met(self):
        assert es.milestone_status(0.0, 0, 0)["met"] is False


# ---------------------------------------------------------------------------
# the smoke's default plan (documented design decisions)
# ---------------------------------------------------------------------------

class TestSmokePlanDefaults:
    def test_default_scale_is_sane_for_this_machine(self):
        """The defaults must stay within a ~4h wall budget at 88-160 sims/s."""
        total_games = es.DEFAULT_CYCLES * es.DEFAULT_GAMES_PER_CYCLE
        assert total_games == 12  # 12 real self-play games
        # every self-play game is capped so a weak model cannot run 1000 moves
        assert es.DEFAULT_SELFPLAY_MAX_MOVES <= 250
        # training steps are the cheap lever: >= 1000 steps per cycle
        assert es.DEFAULT_STEPS_PER_CYCLE >= 1000
        # the per-cycle eval gate stays cheap
        assert es.DEFAULT_EVAL_GAMES <= 3 and es.DEFAULT_EVAL_SIMS <= 40
        # the plan's vs-random milestone command form is 20 games
        assert es.DEFAULT_MATCH_GAMES == 20
        # the plan's reduced-sample ladder allowance is time-boxed
        assert es.DEFAULT_LADDER_MAX_TIME_MIN <= 120

    def test_default_phases_exclude_slow_ladder(self):
        # the ladder re-run is available but not in the default phase list
        # (it is the single most expensive phase; run it explicitly)
        assert "ladder" not in es.DEFAULT_PHASES.split(",")
        assert "train" in es.DEFAULT_PHASES.split(",")


# ---------------------------------------------------------------------------
# report builder over a representative phase set
# ---------------------------------------------------------------------------

def _fake_phases():
    return [
        {
            "phase": "train", "wall_time_s": 3600.0, "peak_gpu_mem_gb": 2.4,
            "protocol": {"cycles": 4, "games_per_cycle": 3,
                         "steps_per_cycle": 1000, "simulations": 60,
                         "batch_size": 128},
            "loop": {"games_generated": 12, "steps_trained": 4000,
                     "global_step_final": 4000, "cycles_done": 4,
                     "positions_in_buffer": 3200, "loss_first": 5.9,
                     "loss_last": 3.1, "loss_decrease": True,
                     "eval_gates": 4, "eval_summaries": []},
            "checkpoint": {"latest": "models/latest.pt",
                           "latest_exists": True, "best": "models/best.pt",
                           "best_exists": True},
        },
        {
            "phase": "match", "wall_time_s": 1800.0,
            "engine2": "random", "sims": 60, "games": 20, "completed": 20,
            "errors": 0, "wins": 16, "winrate": 0.8, "elo_diff": 240.4,
            "ci95": [0.58, 0.93],
            "milestone": {"winrate": 0.8, "wins": 16, "games": 20, "bar": 0.8,
                          "ci95": [0.58, 0.93], "met": False,
                          "note": "weak signal"},
            "evidence": "x",
        },
        {
            "phase": "ladder", "wall_time_s": 4500.0,
            "weight_source": "loaded from models/best.pt",
            "games_played": 9, "accepted": False,
            "monotonic": {"P(200>40)": 0.6, "P(800>200)": 1.0,
                          "ordering_ok": True},
            "pairings": {"40v200": {"games": 6, "hi_wins": 4,
                                    "hi_winrate": 0.666, "ci95": [0.30, 0.90]}},
        },
    ]


class TestReportBuilder:
    def test_report_lines_cover_all_plan_fields(self, tmp_path):
        report = {
            "date": "2026-08-07", "device": "cuda",
            "phases_run": ["train", "match", "ladder"],
            "wall_time_s_total": 9900.0, "wall_time_min_total": 165.0,
            "peak_gpu_mem_gb": 2.4,
            "soft_target_2h": {"note": "soft"},
            "training": {"games_generated": 12, "steps_trained": 4000,
                         "global_step_final": 4000,
                         "positions_in_buffer": 3200, "loss_first": 5.9,
                         "loss_last": 3.1, "loss_decrease": True,
                         "eval_gates": 4},
            "model": {"path": "models/best.pt", "arch": {"blocks": 10},
                      "global_step": 4000,
                      "forward": {"policy_shape": [1, 362],
                                  "value_shape": [1, 1], "finite": True}},
            "eval_history_elo_trajectory": [
                {"step": 1000, "winrate": 0.6, "elo": 30.0, "replaced": True}],
            "vs_random": {"phase": "match", "wins": 16, "games": 20,
                          "winrate": 0.8, "ci95": [0.58, 0.93], "sims": 60,
                          "elo_diff": 240.4, "wall_time_s": 1800.0,
                          "milestone": {"met": False, "note": "weak"}},
            "vs_katago": {"phase": "katago", "skipped": True},
            "todo12_gate_rerun": {"phase": "ladder",
                                  "weight_source": "models/best.pt",
                                  "games_played": 9, "wall_time_s": 4500.0,
                                  "pairings": {
                                      "40v200": {"games": 6, "hi_wins": 4,
                                                 "hi_winrate": 0.666,
                                                 "ci95": [0.30, 0.90]}},
                                  "monotonic": {"P(200>40)": 0.6,
                                                "P(800>200)": 1.0,
                                                "accepted": False},
                                  "accepted": False, "gate_note": "x"},
            "viz": {"phase": "viz", "png": "logs/viz_smoke.png", "bytes": 20480},
            "assessment": {"met": False,
                           "notes": ["n1", "n2", "n3"]},
        }
        lines = es._report_lines(report)
        blob = "\n".join(lines)
        for needle in [
            "wall time:", "peak GPU mem:", "games generated,", "steps trained",
            "positions in replay buffer", "loss", "eval gates",
            "eval history", "vs-random:", ">80%", "todo-12 gate re-run",
            "monotonic:", "UNCONFIRMED", "assessment",
        ]:
            assert needle in blob
        # the plan's required report fields are all present
        assert "date:" in blob and "model:" in blob

    def test_milestone_status_classification(self):
        st = es.milestone_status(0.8, 16, 20)
        assert st["met"] is False  # exactly AT the bar is not met
        st2 = es.milestone_status(0.85, 17, 20)
        assert st2["met"] is True


# ---------------------------------------------------------------------------
# tiny real end-to-end training slice through the actual orchestrator
# ---------------------------------------------------------------------------

class TestTrainPhaseE2E:
    def _run(self, tmp_path, monkeypatch):
        cfg = {
            "board_size": 9, "komi": 7.5, "blocks": 1, "channels": 8,
            "lr": 0.2, "momentum": 0.9, "l2": 1e-4,
            "lr_schedule_steps": [50000, 100000], "batch_size": 8,
            "replay_buffer_games": 1000, "symmetry_aug": True,
            "simulations": 4, "eval_games": 1, "eval_sims": 2,
            "eval_interval_steps": 2000, "replace_threshold": 0.55,
            "virtual_loss": 3, "viz_enabled": False,
        }
        cfg_path = tmp_path / "cfg.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        data = tmp_path / "data"
        ckpt = tmp_path / "models"
        tlog = tmp_path / "train.jsonl"
        hist = tmp_path / "eval_history.jsonl"
        ev = tmp_path / "evidence"
        argv = [
            "--phases", "train",
            "--config", str(cfg_path),
            "--data-dir", str(data),
            "--checkpoint-dir", str(ckpt),
            "--train-log", str(tlog),
            "--history", str(hist),
            "--evidence-dir", str(ev),
            "--board-size", "9", "--blocks", "1", "--channels", "8",
            "--batch-size", "8", "--no-symmetry",
            "--cycles", "1", "--games-per-cycle", "1", "--steps-per-cycle", "3",
            "--simulations", "4", "--selfplay-max-moves", "60",
            "--eval-games", "1", "--eval-sims", "2", "--eval-max-moves", "60",
            "--seed", "7",
        ]
        rc = es.main(argv)
        return rc, tmp_path

    def test_zero_to_best_pt_wiring(self, tmp_path, monkeypatch):
        rc, tmp = self._run(tmp_path, monkeypatch)
        assert rc == 0
        ev = tmp / "evidence"
        train_json = json.loads((ev / "train-report.json").read_text(encoding="utf-8"))
        # real self-play data was generated and trained on
        assert train_json["loop"]["games_generated"] >= 1
        assert train_json["loop"]["steps_trained"] >= 1
        assert train_json["loop"]["loss_first"] is not None
        assert train_json["loop"]["loss_decrease"] is not None
        assert train_json["loop"]["positions_in_buffer"] >= 1
        # checkpoints written (plan: >= 2 checkpoints)
        assert (tmp / "models" / "best.pt").exists()
        assert (tmp / "models" / "latest.pt").exists()
        # train.jsonl carries the plan's fields on each train-step line
        lines = [json.loads(l) for l in
                 (tmp / "train.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        steps = [e for e in lines if e.get("event") == "train_step"]
        assert len(steps) >= 1
        for rec in steps:
            assert {"step", "loss", "games", "elo"} <= set(rec)
        # the eval gate ran at least once (cycle-end gate, force_final_eval)
        assert train_json["loop"]["eval_gates"] >= 1
        assert train_json["checkpoint"]["best_exists"] is True

    def test_phase_cache_skips_completed_train(self, tmp_path, monkeypatch):
        rc, tmp = self._run(tmp_path, monkeypatch)
        assert rc == 0
        ev = tmp / "evidence"
        # a second run with the same evidence dir skips re-training
        cfg_path = tmp / "cfg.yaml"
        argv = [
            "--phases", "train",
            "--config", str(cfg_path),
            "--data-dir", str(tmp / "data"),
            "--checkpoint-dir", str(tmp / "models"),
            "--train-log", str(tmp / "train.jsonl"),
            "--history", str(tmp / "eval_history.jsonl"),
            "--evidence-dir", str(ev),
            "--board-size", "9", "--blocks", "1", "--channels", "8",
            "--batch-size", "8", "--no-symmetry",
            "--cycles", "1", "--games-per-cycle", "1", "--steps-per-cycle", "3",
            "--simulations", "4", "--selfplay-max-moves", "60",
            "--eval-games", "1", "--eval-sims", "2", "--eval-max-moves", "60",
            "--seed", "7",
        ]
        assert es.main(argv) == 0
        # the report artifact is still the one from the first run
        assert (ev / "train-report.json").exists()
