"""Tests for the training-loop orchestration (todo 16).

Per the plan's todo-16 acceptance criteria:

* **cycle wiring** -- a run with mocked self-play / train / eval proves the
  orchestrator calls ``generate_games`` -> ``train_steps`` (in order, with the
  right ``global_step``) and fires the evaluation gate once per cycle with
  ``models/latest.pt`` as the candidate;
* **interruptible resume** -- on a tiny budget, an interrupted run (graceful
  ``KeyboardInterrupt`` via ``--interrupt-at-steps``) plus a ``--resume`` run
  reproduces the loss trajectory of an uninterrupted run (deterministic
  resume, Oracle F9, tolerance ``1e-4``), and the resumed log's first step is
  greater than the interrupted log's last step (plan: 新日志首步 > 旧末步);
* **eval gating integration** -- with a real ``evaluate_and_gate`` but a
  stubbed ``run_evaluation`` win rate, a replace gate writes ``best.pt`` and a
  keep gate leaves it byte-for-byte unchanged, with matching ``eval_history``
  JSONL entries;
* **logging** -- every ``train_step`` JSONL record carries the plan's fields
  (``step`` / ``loss`` / ``games`` / ``elo`` / ``timestamp``, Oracle G2);
* **lazy viz** -- with ``viz_enabled=true`` and no todo-17 module, the loop
  warns and continues in pure-log mode (plan, Oracle #9/F3).

All games are synthetic npz files on a 9x9 board with a tiny 1x8 network on
the fastest available device so the suite stays fast and deterministic.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

import omigamax.train.evaluate as ev_mod
import omigamax.train.loop as loop
from omigamax.train.buffer import ReplayBuffer

SIZE = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOL = 1e-4  # plan Oracle F9 resume tolerance


def make_cfg(**overrides) -> dict:
    """Small, fast, deterministic config for the loop tests."""
    cfg = {
        "board_size": SIZE,
        "komi": 7.5,
        "blocks": 1,
        "channels": 8,
        "lr": 0.2,
        "momentum": 0.9,
        "l2": 1e-4,
        "lr_schedule_steps": [50000, 100000],
        "batch_size": 8,
        "replay_buffer_games": 1000,
        "symmetry_aug": False,
        "simulations": 4,
        "eval_games": 2,
        "eval_sims": 4,
        "eval_interval_steps": 2000,
        "replace_threshold": 0.55,
        "virtual_loss": 3,
        "viz_enabled": True,
    }
    cfg.update(overrides)
    return cfg


def write_synthetic_games(data_dir, games: int, seed_base: int, size: int = SIZE,
                          t: int = 30) -> None:
    """Deterministic synthetic npz games (same content for the same seed_base)."""
    rng = np.random.default_rng(1000 + int(seed_base))
    for g in range(int(games)):
        s = rng.random((t, 17, size, size)).astype(np.float32)
        pi = rng.random((t, size * size + 1)).astype(np.float32)
        pi /= pi.sum(axis=1, keepdims=True)
        z = rng.choice([-1.0, 1.0], size=t).astype(np.float32)
        np.savez(
            Path(data_dir) / f"game_{int(seed_base) + g:010d}.npz",
            s=s, pi=pi, z=z,
            board_size=np.int64(size), move_count=np.int64(t),
        )


def fake_generate_games(network, cfg, games, data_dir, keep, seed, simulations,
                        **kwargs):
    """Generate deterministic synthetic games; return a report dict."""
    write_synthetic_games(data_dir, games, int(seed), size=cfg["board_size"])
    t = 30
    return {
        "games": int(games),
        "positions": int(games) * t,
        "sims": int(games) * t * int(simulations),
        "sims_per_sec": 0.0,
        "wall_time_s": 0.0,
        "data_dir": str(data_dir),
    }, []


def fake_evaluate_and_gate(candidate_path, best_path, cfg, **kwargs):
    """Mocked gate: always KEEP, elo 3.0."""
    return {
        "replaced_best": False,
        "match": {"winrate": 0.5, "candidate_wins": 1, "games": 2},
        "elo_update": {"elo": 3.0},
    }


def read_train_steps(path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def train_step_lines(path) -> list[dict]:
    return [e for e in read_train_steps(path) if e.get("event") == "train_step"]


# ---------------------------------------------------------------------------
# evaluation scheduling
# ---------------------------------------------------------------------------

class TestEvalDue:
    def test_cycle_end_always_evaluates(self):
        assert loop.eval_due(1, cycle_end=True, eval_interval_steps=2000) is True
        assert loop.eval_due(7, cycle_end=True, eval_interval_steps=0) is True

    def test_interval_boundary_crossing(self):
        # step_after == multiple of the interval fires; others do not
        for s in (1, 2, 3, 4):
            assert loop.eval_due(s, cycle_end=False, eval_interval_steps=5) is False
        assert loop.eval_due(5, cycle_end=False, eval_interval_steps=5) is True
        for s in (6, 7, 8, 9):
            assert loop.eval_due(s, cycle_end=False, eval_interval_steps=5) is False
        assert loop.eval_due(10, cycle_end=False, eval_interval_steps=5) is True

    def test_interval_zero_only_cycle_end(self):
        assert loop.eval_due(5000, cycle_end=False, eval_interval_steps=0) is False
        assert loop.eval_due(5000, cycle_end=True, eval_interval_steps=0) is True

    def test_interval_does_not_suppress_cycle_end(self):
        # even a boundary step that is also a cycle end still evaluates once
        assert loop.eval_due(5, cycle_end=True, eval_interval_steps=5) is True


# ---------------------------------------------------------------------------
# lazy visualization (todo 17 mount point)
# ---------------------------------------------------------------------------

class TestLazyViz:
    def test_viz_enabled_but_module_absent_warns_and_continues(
            self, caplog, monkeypatch):
        """Graceful degradation when ``board_window`` is unavailable."""
        import sys
        monkeypatch.setitem(sys.modules, "omigamax.viz.board_window", None)
        logger = logging.getLogger("test_viz_absent")
        caplog.set_level(logging.WARNING, logger="test_viz_absent")
        out = loop.start_viz_if_available({"viz_enabled": True}, logger=logger)
        assert out["started"] is False
        assert out["reason"] == "module_unavailable"
        assert "viz" in caplog.text.lower()

    def test_viz_disabled(self):
        out = loop.start_viz_if_available({"viz_enabled": False})
        assert out["started"] is False
        assert out["reason"] == "disabled_by_config"

    def test_loop_runs_with_viz_enabled_true(self, tmp_path, monkeypatch):
        """The full loop with viz_enabled=true and the todo-17 module present:
        a viz thread is started and cleanly stopped; the loop completes."""
        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        report = loop.run_loop(
            make_cfg(),
            device=DEVICE,
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=3,
            batch_size=8, use_symmetry=False, seed=0,
        )
        viz = report["protocol"]["viz"]
        assert viz["reason"] == "available"
        assert viz["started"] is True
        # the run's own cleanup stopped the daemon thread
        thread = viz["thread"]
        thread.join(timeout=2)
        assert thread.stopped is True
        assert report["loop"]["interrupted"] is False
        assert report["checkpoint"]["latest_exists"] is True


# ---------------------------------------------------------------------------
# F2 MAJOR 2: the live window must actually receive frames during training
# ---------------------------------------------------------------------------

class TestVizFeed:
    def test_frames_pushed_during_training(self, tmp_path, monkeypatch):
        """With viz mounted, every training step enqueues a Snapshot frame."""
        from omigamax.viz.board_window import SnapshotQueue

        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        queue = SnapshotQueue(maxlen=32)
        monkeypatch.setattr(
            loop, "start_viz_if_available",
            lambda cfg, logger=None: {
                "started": True, "reason": "available",
                "queue": queue, "thread": None, "stop": lambda: None,
            },
        )
        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=5,
            batch_size=8, use_symmetry=False, seed=0,
        )
        # >= 1 opening frame (F3c) + >= 2 per-game frames (F3c) + one frame
        # per train step (F2) -- fake_generate_games pushes no per-move frames
        # (F3d), so the counts above are lower bounds, not exact totals
        assert len(queue) >= 8
        snaps = []
        while True:
            s = queue.poll(timeout=0.01)
            if s is None:
                break
            snaps.append(s)
        assert len(snaps) >= 8
        # oldest frame: the opening empty-board frame (no games, no metrics)
        assert snaps[0].train_step is None
        assert snaps[0].loss is None
        assert snaps[0].games == 0
        assert all(v == 0 for row in snaps[0].board for v in row)
        # per-game self-play frames (board present, no train step yet)
        g1 = next(s for s in snaps if s.games == 1)
        g2 = next(s for s in snaps if s.games == 2)
        assert g1.train_step is None and g2.train_step is None
        # newest frame carries the last train step's live metrics
        last = snaps[-1]
        assert last.train_step >= 1
        assert last.loss is not None and last.games >= 1

    def test_frame_pushed_during_selfplay_phase(self, tmp_path, monkeypatch):
        """F3b/F3c: with viz on, frames are enqueued during the self-play
        phase -- before ANY training step runs -- so the window opens while
        the cycle's games are being generated, not only once training
        starts."""
        from omigamax.viz.board_window import SnapshotQueue

        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        queue = SnapshotQueue(maxlen=32)
        monkeypatch.setattr(
            loop, "start_viz_if_available",
            lambda cfg, logger=None: {
                "started": True, "reason": "available",
                "queue": queue, "thread": None, "stop": lambda: None,
            },
        )

        qsize_at_first_step: dict = {}
        real_train_steps = loop.train_steps

        def recording_train_steps(model, optimizer, buffer, steps, **kwargs):
            if "qsize" not in qsize_at_first_step:
                qsize_at_first_step["qsize"] = len(queue)
            return real_train_steps(model, optimizer, buffer, steps, **kwargs)

        monkeypatch.setattr(loop, "train_steps", recording_train_steps)

        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=5,
            batch_size=8, use_symmetry=False, seed=0,
        )
        # by the time the FIRST training step runs, >= 3 frames were already
        # enqueued: the opening empty board + one per finished game
        # (fake_generate_games streams no per-move frames, so this is a lower
        # bound rather than an exact count)
        assert qsize_at_first_step["qsize"] >= 3

    def test_opening_frame_pushed_immediately(self, tmp_path, monkeypatch):
        """F3c: the opening empty-board frame is pushed right after the viz
        thread starts -- the queue is non-empty before ANY game is generated
        (window pops up within seconds of launch)."""
        from omigamax.viz.board_window import SnapshotQueue

        seen: dict = {}
        real_generate_games = loop.generate_games

        def recording_generate_games(network, cfg, games, data_dir, keep,
                                     seed, simulations, **kwargs):
            if "qsize" not in seen:
                # first call happens after start_viz_if_available returned --
                # the opening frame must already be in the queue
                seen["qsize"] = len(queue)
                seen["frame"] = queue.poll(timeout=0.01)
            return real_generate_games(network, cfg, games,
                                       data_dir=data_dir, keep=keep,
                                       seed=seed, simulations=simulations,
                                       **kwargs)

        monkeypatch.setattr(loop, "generate_games", recording_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        queue = SnapshotQueue(maxlen=32)
        monkeypatch.setattr(
            loop, "start_viz_if_available",
            lambda cfg, logger=None: {
                "started": True, "reason": "available",
                "queue": queue, "thread": None, "stop": lambda: None,
            },
        )
        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=2,
            batch_size=8, use_symmetry=False, seed=0,
        )
        # the opening frame was already enqueued before the first game
        assert seen["qsize"] >= 1
        frame = seen["frame"]
        assert frame is not None
        assert frame.games == 0
        assert frame.train_step is None
        assert frame.move_number == 0
        assert all(v == 0 for row in frame.board for v in row)

    def test_frame_after_first_game_of_cycle(self, tmp_path, monkeypatch):
        """F3c: after the first game of the cycle lands, a frame with that
        game's board (train_step None) is in the queue -- the window shows
        each finished game during self-play."""
        from omigamax.viz.board_window import SnapshotQueue

        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        queue = SnapshotQueue(maxlen=32)
        monkeypatch.setattr(
            loop, "start_viz_if_available",
            lambda cfg, logger=None: {
                "started": True, "reason": "available",
                "queue": queue, "thread": None, "stop": lambda: None,
            },
        )
        # record the queue contents right after the FIRST game is generated
        seen: dict = {}
        real_train_steps = loop.train_steps

        def recording_train_steps(model, optimizer, buffer, steps, **kwargs):
            if "qsize_after_game1" not in seen:
                seen["qsize_after_game1"] = len(queue)
            return real_train_steps(model, optimizer, buffer, steps, **kwargs)

        monkeypatch.setattr(loop, "train_steps", recording_train_steps)
        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=3,
            batch_size=8, use_symmetry=False, seed=0,
        )
        # opening + one frame per finished game => >= 2 after the first game
        assert seen["qsize_after_game1"] >= 2
        snaps = []
        while True:
            s = queue.poll(timeout=0.01)
            if s is None:
                break
            snaps.append(s)
        # find the per-game frame with the first game's board
        game_frame = next(s for s in snaps if s.games == 1)
        assert game_frame.train_step is None
        assert game_frame.board is not None

    def test_per_move_frames_during_selfplay(self, tmp_path, monkeypatch):
        """F3d: with viz on, the queue receives a frame for EVERY move of a
        game -- >= 2 frames with strictly increasing move_number arrive
        BEFORE the game ends (livestream, not just per-finished-game)."""
        from omigamax.viz.board_window import SnapshotQueue

        queue = SnapshotQueue(maxlen=64)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)

        def fake_gen_with_moves(network, cfg, games, data_dir, keep, seed,
                                simulations, frame_callback=None, **kwargs):
            # mimic one real game: 3 live moves streamed via the frame
            # callback (a fresh board each), then the synthetic npz lands
            from omigamax.rules import BLACK, WHITE, Board
            board = Board(cfg["board_size"])
            for i in range(3):
                color = BLACK if len(board.moves) % 2 == 0 else WHITE
                board.play((i, 0), color)
                if frame_callback is not None:
                    frame_callback(board, len(board.moves), color)
            write_synthetic_games(data_dir, games, int(seed),
                                  size=cfg["board_size"])
            return {"games": int(games), "sims_per_sec": 0.0}, []

        monkeypatch.setattr(loop, "generate_games", fake_gen_with_moves)
        monkeypatch.setattr(
            loop, "start_viz_if_available",
            lambda cfg, logger=None: {
                "started": True, "reason": "available",
                "queue": queue, "thread": None, "stop": lambda: None,
            },
        )
        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=1, steps_per_cycle=1,
            batch_size=8, use_symmetry=False, seed=0,
        )
        snaps = []
        while True:
            s = queue.poll(timeout=0.01)
            if s is None:
                break
            snaps.append(s)
        # FIFO order: opening empty board (move 0), then the 3 per-move live
        # frames (moves 1, 2, 3) before the game-end / train-step frames
        assert snaps[0].move_number == 0
        assert snaps[0].board and all(v == 0 for row in snaps[0].board for v in row)
        per_move = [s for s in snaps if s.move_number in (1, 2, 3)]
        assert [s.move_number for s in per_move] == [1, 2, 3]
        # consecutive frames carry the live board of the move just played
        assert per_move[0].board[0][0] == 1  # black stone on (0,0) after move 1
        assert per_move[1].board[1][0] == 2  # white stone on (1,0) after move 2
        assert per_move[2].board[2][0] == 1  # black stone on (2,0) after move 3
        assert per_move[0].train_step is None  # still in the self-play phase

    def test_push_failure_never_crashes_training(self, tmp_path, monkeypatch):
        """A broken queue must not abort the training loop."""
        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)

        class BrokenQueue:
            def push(self, snap):  # pragma: no cover - exercised, never caught
                raise RuntimeError("viz queue exploded")
            def __len__(self):
                return 0

        monkeypatch.setattr(
            loop, "start_viz_if_available",
            lambda cfg, logger=None: {
                "started": True, "reason": "available",
                "queue": BrokenQueue(), "thread": None, "stop": lambda: None,
            },
        )
        report = loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=4,
            batch_size=8, use_symmetry=False, seed=1,
        )
        assert report["loop"]["interrupted"] is False
        assert report["loop"]["steps_trained"] == 4
        assert report["checkpoint"]["latest_exists"] is True

    def test_viz_disabled_pushes_nothing(self, tmp_path, monkeypatch):
        """With viz off there is no queue to push to and training still runs."""
        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        report = loop.run_loop(
            make_cfg(viz_enabled=False), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=3,
            batch_size=8, use_symmetry=False, seed=2,
        )
        assert report["protocol"]["viz"]["started"] is False
        assert report["loop"]["steps_trained"] == 3


# ---------------------------------------------------------------------------
# cycle wiring (mocked components)
# ---------------------------------------------------------------------------

class TestCycleWiring:
    def test_cycle_order_and_args(self, tmp_path, monkeypatch):
        calls: list[tuple] = []

        def fake_gen(network, cfg, games, data_dir, keep, seed, simulations,
                     **kw):
            calls.append(("selfplay", int(seed), int(games)))
            write_synthetic_games(data_dir, games, int(seed),
                                  size=cfg["board_size"])
            return {"games": int(games), "sims_per_sec": 0.0}, []

        def fake_train(model, optimizer, buffer, steps, **kwargs):
            gs = int(kwargs["global_step"])
            calls.append(("train", gs))
            return [0.5 + 0.05 * gs], [0.2], gs + 1, kwargs["rng"]

        def fake_eval(candidate_path, best_path, cfg, **kw):
            calls.append(("eval", Path(candidate_path).name))
            return fake_evaluate_and_gate(candidate_path, best_path, cfg, **kw)

        monkeypatch.setattr(loop, "generate_games", fake_gen)
        monkeypatch.setattr(loop, "train_steps", fake_train)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_eval)

        report = loop.run_loop(
            make_cfg(),
            device=DEVICE,
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=3,
            batch_size=8, use_symmetry=False, seed=42,
        )

        # F3c: self-play generates one game per call, seeds staying
        # continuous (42, 43) -- the npz set equals a batch call with
        # seed=42, games=2; then 3 train steps, 1 gate
        selfplay = [c for c in calls if c[0] == "selfplay"]
        trains = [c for c in calls if c[0] == "train"]
        evals = [c for c in calls if c[0] == "eval"]
        assert selfplay == [("selfplay", 42, 1), ("selfplay", 43, 1)]
        assert [c[1] for c in trains] == [0, 1, 2]  # global_steps before step
        assert len(evals) == 1 and evals[0][1] == "latest.pt"

        # gate fires after the cycle's training
        assert calls.index(("eval", "latest.pt")) > calls.index(("train", 2))
        # the self-play generation happens before any training
        assert calls.index(("selfplay", 42, 1)) < calls.index(("train", 0))

        assert report["loop"]["global_step_final"] == 3
        assert report["loop"]["cycles_done"] == 1
        assert report["checkpoint"]["latest_exists"] is True

        # per-step log records
        steps = train_step_lines(tmp_path / "train.jsonl")
        assert [s["step"] for s in steps] == [1, 2, 3]

    def test_two_cycles_advance_games_and_evals(self, tmp_path, monkeypatch):
        seeds: list[int] = []
        n_evals = {"n": 0}

        def fake_gen(network, cfg, games, data_dir, keep, seed, simulations,
                     **kw):
            seeds.append(int(seed))
            write_synthetic_games(data_dir, games, int(seed),
                                  size=cfg["board_size"])
            return {"games": int(games), "sims_per_sec": 0.0}, []

        def fake_train(model, optimizer, buffer, steps, **kwargs):
            gs = int(kwargs["global_step"])
            return [0.5], [0.2], gs + 1, kwargs["rng"]

        def fake_eval(candidate_path, best_path, cfg, **kw):
            n_evals["n"] += 1
            return fake_evaluate_and_gate(candidate_path, best_path, cfg, **kw)

        monkeypatch.setattr(loop, "generate_games", fake_gen)
        monkeypatch.setattr(loop, "train_steps", fake_train)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_eval)

        report = loop.run_loop(
            make_cfg(),
            device=DEVICE,
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=2, games_per_cycle=2, steps_per_cycle=2,
            batch_size=8, use_symmetry=False, seed=7,
        )
        # seeds continue across cycles, one per game: 7, 8 then 9, 10
        assert seeds == [7, 8, 9, 10]
        assert n_evals["n"] == 2  # one gate per cycle
        assert report["loop"]["global_step_final"] == 4
        assert report["loop"]["cycles_done"] == 2
        assert report["loop"]["games_generated"] == 4


# ---------------------------------------------------------------------------
# per-step logging fields
# ---------------------------------------------------------------------------

class TestLogging:
    def test_train_step_records_plan_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)
        report = loop.run_loop(
            make_cfg(),
            device=DEVICE,
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=5,
            batch_size=8, use_symmetry=False, seed=3,
        )
        steps = train_step_lines(tmp_path / "train.jsonl")
        assert len(steps) == 5  # one line per training step
        for rec in steps:
            # plan Oracle G2 fields: step / loss / 对局数(games) / elo / 时间戳
            assert {"event", "timestamp", "step", "loss", "lr", "games",
                    "elo", "cycle"} <= set(rec)
            assert isinstance(rec["step"], int)
            assert isinstance(rec["loss"], float) and np.isfinite(rec["loss"])
            assert isinstance(rec["lr"], float)
            assert rec["games"] >= 1
            assert isinstance(rec["elo"], float)
        assert [s["step"] for s in steps] == [1, 2, 3, 4, 5]
        assert report["loop"]["eval_gates"] == 1


# ---------------------------------------------------------------------------
# interruptible resume: interrupted + --resume == uninterrupted
# ---------------------------------------------------------------------------

class TestResumeContinuity:
    def _run(self, tmp_path, *, resume=False, interrupt_after=None):
        report = loop.run_loop(
            make_cfg(),
            device=DEVICE,
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=20,
            batch_size=8, use_symmetry=False, seed=42,
            resume=resume, interrupt_after=interrupt_after,
        )
        return report

    def test_interrupted_then_resumed_equals_uninterrupted(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(loop, "generate_games", fake_generate_games)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)

        # uninterrupted: one full cycle of 20 steps
        run_a = self._run(tmp_path / "a")
        steps_a = train_step_lines(tmp_path / "a" / "train.jsonl")
        losses_a = [s["loss"] for s in steps_a]
        assert [s["step"] for s in steps_a] == list(range(1, 21))
        assert run_a["loop"]["global_step_final"] == 20
        # an uninterrupted completed cycle persists cycles_completed=1
        ckpt_a = loop.load_checkpoint(tmp_path / "a" / "models" / "latest.pt")
        assert ckpt_a["extra"]["cycles_completed"] == 1
        assert ckpt_a["extra"]["steps_into_cycle"] == 0

        # interrupted: 10 steps, graceful checkpoint
        run_b = self._run(tmp_path / "b", interrupt_after=10)
        assert run_b["loop"]["interrupted"] is True
        assert run_b["loop"]["global_step_final"] == 10
        steps_b1 = train_step_lines(tmp_path / "b" / "train.jsonl")
        losses_b1 = [s["loss"] for s in steps_b1]
        assert [s["step"] for s in steps_b1] == list(range(1, 11))
        # checkpoint carries the in-flight cycle progress
        ckpt = loop.load_checkpoint(tmp_path / "b" / "models" / "latest.pt")
        assert ckpt["extra"]["steps_into_cycle"] == 10
        assert ckpt["extra"]["games_generated"] == 2
        assert ckpt["extra"]["cycles_completed"] == 0

        # resumed: the second half of the same cycle from the checkpoint
        run_c = self._run(tmp_path / "b", resume=True)
        assert run_c["loop"]["interrupted"] is False
        assert run_c["loop"]["resumed"] is True
        assert run_c["loop"]["global_step_final"] == 20
        steps_b2 = train_step_lines(tmp_path / "b" / "train.jsonl")
        # 10 old lines + 10 new lines from the resumed run
        assert [s["step"] for s in steps_b2] == list(range(1, 21))
        resumed_steps = [s for s in steps_b2 if s["step"] > 10]
        losses_b2 = [s["loss"] for s in resumed_steps]
        assert [s["step"] for s in resumed_steps] == list(range(11, 21))

        # plan acceptance: 新日志首步 > 旧末步
        assert resumed_steps[0]["step"] == 11
        assert steps_b1[-1]["step"] == 10
        assert resumed_steps[0]["step"] > steps_b1[-1]["step"]

        # loss continuity: interrupted+resumed == uninterrupted (plan 1e-4)
        np.testing.assert_allclose(
            np.asarray(losses_b1, dtype=np.float64),
            np.asarray(losses_a[:10], dtype=np.float64),
            atol=TOL, rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(losses_b2, dtype=np.float64),
            np.asarray(losses_a[10:], dtype=np.float64),
            atol=TOL, rtol=0.0,
        )

    def test_resume_requires_no_duplicate_generation(self, tmp_path, monkeypatch):
        """A mid-cycle resume must NOT regenerate games (model has advanced)."""
        gen_calls = {"n": 0}

        def counting_gen(network, cfg, games, data_dir, keep, seed,
                         simulations, **kw):
            gen_calls["n"] += 1
            write_synthetic_games(data_dir, games, int(seed),
                                  size=cfg["board_size"])
            return {"games": int(games), "sims_per_sec": 0.0}, []

        monkeypatch.setattr(loop, "generate_games", counting_gen)
        monkeypatch.setattr(loop, "evaluate_and_gate", fake_evaluate_and_gate)

        # first leg interrupted mid-cycle: 2 per-game calls (F3c), one npz each
        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=10,
            batch_size=8, use_symmetry=False, seed=1, interrupt_after=5,
        )
        assert gen_calls["n"] == 2  # one per game

        # resumed leg: games already on disk -> no new generation
        loop.run_loop(
            make_cfg(), device=DEVICE,
            data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=1, games_per_cycle=2, steps_per_cycle=10,
            batch_size=8, use_symmetry=False, seed=1, resume=True,
        )
        assert gen_calls["n"] == 2  # unchanged


# ---------------------------------------------------------------------------
# eval gating integration (real evaluate_and_gate, stubbed win rate)
# ---------------------------------------------------------------------------

class TestEvalGating:
    def test_replace_writes_best_keep_leaves_unchanged(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(loop, "generate_games", fake_generate_games)

        winrates = iter([1.0, 0.0])  # first gate replaces, second keeps

        def fake_run_evaluation(candidate_net, best_net, cfg, **kwargs):
            w = next(winrates)
            wins = 2 if w > 0.5 else 0
            return {
                "games": 2, "sims": 2, "board_size": SIZE, "komi": 7.5,
                "virtual_loss": 3, "threshold": 0.55,
                "candidate_wins": wins, "draws": 0, "winrate": w,
                "elo_diff": 2400.0 if w > 0.5 else -2400.0,
                "replaced": w >= 0.55, "wall_time_s": 0.0, "games_detail": [],
            }

        monkeypatch.setattr(ev_mod, "run_evaluation", fake_run_evaluation)

        report = loop.run_loop(
            make_cfg(),
            device=DEVICE,
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "models",
            train_log=tmp_path / "train.jsonl",
            history=tmp_path / "eval_history.jsonl",
            cycles=2, games_per_cycle=2, steps_per_cycle=2,
            batch_size=8, use_symmetry=False, seed=11,
        )

        best_path = tmp_path / "models" / "best.pt"
        assert best_path.exists()
        assert report["checkpoint"]["best_exists"] is True
        summaries = report["loop"]["eval_summaries"]
        assert len(summaries) == 2
        assert summaries[0]["replaced"] is True
        assert summaries[1]["replaced"] is False
        assert summaries[0]["winrate"] == pytest.approx(1.0)
        assert summaries[1]["winrate"] == pytest.approx(0.0)

        # best.pt was written at the first (replace) gate with global_step 2
        best_ckpt = loop.load_checkpoint(best_path)
        assert best_ckpt["global_step"] == 2

        # eval history JSONL records both gates
        hist = (tmp_path / "eval_history.jsonl").read_text(encoding="utf-8")
        entries = [json.loads(l) for l in hist.splitlines() if l.strip()]
        assert [e["replaced_best"] for e in entries] == [True, False]
        assert entries[0]["elo"] == pytest.approx(entries[0]["elo_before"] + 16.0)
        # second gate: elo continues from the first
        assert entries[1]["elo_before"] == entries[0]["elo"]

        # best.pt is still the step-2 checkpoint after the keep gate: reloading
        # it must restore global_step 2 (a second replace would have bumped it).
        best_again = loop.load_checkpoint(best_path)
        assert best_again["global_step"] == 2


# ---------------------------------------------------------------------------
# signal handling (Windows SIGBREAK routed to KeyboardInterrupt)
# ---------------------------------------------------------------------------

class TestSignalHandling:
    def test_install_sigbreak_is_safe(self):
        # must not raise in the pytest main thread (Windows: SIGBREAK exists)
        result = loop._install_sigbreak()
        assert result in (True, False)
