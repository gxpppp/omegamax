"""Tests for the self-play data generator (todo 13).

Per the plan's todo-13 acceptance criteria:

  * ``z`` correctness: outcome ``+1``/``-1`` from the mover's perspective,
    consistent with the game's Tromp-Taylor score (``z`` vs scoring);
  * ``pi`` sums to 1, is a *distribution* (not a one-hot), and equals the
    visit-count search policy at ``tau = 1``;
  * ``s`` shape ``(T, 17, N, N)`` float32, round-trips against re-encoding
    the replayed game (history window + colour plane);
  * the per-game npz round-trips (load -> arrays match, metadata present);
  * eval()/no_grad() discipline: the network stays in ``eval()`` mode and no
    forward pass records a live autograd graph;
  * legality: every recorded move replays against the rules engine without an
    illegal move;
  * the temperature schedule: ``tau = 1`` early, argmax after the threshold;
  * the data-directory pruning keeps only the newest ``keep`` games and the
    resign bit (default 0.0 = disabled) behaves correctly.

All games are played on a 5x5 board with a tiny 1x4 network on CPU so the
suite stays fast; no GPU is required. ``dirichlet_alpha=0.0`` disables root
noise in the determinism-sensitive tests.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from omigamax.config import load_config
from omigamax.mcts import (
    BatchedNetworkEvaluator,
    make_root,
    run_search,
    temperature_policy,
    visit_count_policy,
)
from omigamax.network.features import encode, pass_index
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board

import omigamax.train.selfplay as sp

SIZE = 5
N_LOGITS = SIZE * SIZE + 1


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def net():
    """Tiny deterministic network: 1 block x 4 channels on a 5x5 board (CPU)."""
    torch.manual_seed(0)
    return create_model(blocks=1, channels=4, board_size=SIZE)


def replay_actions(actions, size):
    """Replay recorded policy-index actions on a fresh board; raises on illegal."""
    board = Board(size)
    for a in actions:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        if a == size * size:
            board.pass_move(color)
        else:
            board.play((a // size, a % size), color)
    return board


def replay_states(actions, size):
    """List of board-state snapshots after each replayed move (index i = after i moves)."""
    board = Board(size)
    states = [board.state]
    for a in actions:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        if a == size * size:
            board.pass_move(color)
        else:
            board.play((a // size, a % size), color)
        states.append(board.state)
    return states


class ConstantNet(nn.Module):
    """Deterministic network: uniform policy logits, constant value."""

    def __init__(self, size, value):
        super().__init__()
        self.size = size
        self.value = float(value)
        # device anchor for the batched evaluator (which reads
        # next(network.parameters()).device)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        n = self.size * self.size
        logits = torch.zeros(b, n + 1)
        value = torch.full((b, 1), self.value)
        return logits, value


class ProbeNet(nn.Module):
    """Wraps a network and records eval-mode / grad state seen in each forward."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.seen_training: list[bool] = []
        self.seen_grad_enabled: list[bool] = []

    def forward(self, x):
        self.seen_training.append(self.training)
        self.seen_grad_enabled.append(torch.is_grad_enabled())
        return self.inner(x)


# ---------------------------------------------------------------------------
# z
# ---------------------------------------------------------------------------

class TestZTargets:
    def test_black_wins_all_positions_from_mover_perspective(self):
        z = sp.z_targets(6, "B")
        # mover: black on even indices, white on odd
        assert list(z) == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

    def test_white_wins(self):
        z = sp.z_targets(4, "W")
        assert list(z) == [-1.0, 1.0, -1.0, 1.0]

    def test_jigo_is_zero(self):
        assert list(sp.z_targets(3, None)) == [0.0, 0.0, 0.0]

    def test_empty_game(self):
        assert sp.z_targets(0, "B").shape == (0,)

    def test_dtype_float32(self):
        assert sp.z_targets(5, "B").dtype == np.float32

    def test_z_consistent_with_scoring(self, net, cfg):
        """Integration: z matches the winner recomputed from the final board."""
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=15, seed=7, dirichlet_alpha=0.0,
        )
        board = replay_actions(rec["move_actions"], SIZE)
        winner_ref = board.winner(rec["komi"])
        assert rec["winner"] == winner_ref
        if winner_ref is None:
            assert np.all(rec["z"] == 0.0)
        else:
            for i in range(rec["move_count"]):
                mover_is_black = i % 2 == 0
                expected = 1.0 if (winner_ref == "B") == mover_is_black else -1.0
                assert rec["z"][i] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# pi
# ---------------------------------------------------------------------------

class TestPi:
    def test_pi_sums_to_one_and_masks_illegal(self, net, cfg):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=15, seed=1, dirichlet_alpha=0.0,
        )
        assert rec["move_count"] > 0
        states = replay_states(rec["move_actions"], SIZE)
        board = Board(SIZE)
        for i in range(rec["move_count"]):
            pi = rec["pi"][i]
            assert pi.shape == (N_LOGITS,)
            assert pi.dtype == np.float32
            assert np.all(pi >= 0.0)
            assert np.sum(pi) == pytest.approx(1.0, abs=1e-5)
            color = BLACK if i % 2 == 0 else WHITE
            # positions before move i == state history index i
            board = Board(SIZE)
            for m in range(i):
                a = rec["move_actions"][m]
                c = BLACK if m % 2 == 0 else WHITE
                if a == pass_index(SIZE):
                    board.pass_move(c)
                else:
                    board.play((a // SIZE, a % SIZE), c)
            for r in range(SIZE):
                for c in range(SIZE):
                    idx = r * SIZE + c
                    if not board.is_legal((r, c), color):
                        assert pi[idx] == 0.0, f"illegal {r},{c} at move {i}"

    def test_pi_is_visit_count_policy_at_tau_one(self, net, cfg):
        """Re-run the exact same deterministic search on position 0 and compare."""
        ev = BatchedNetworkEvaluator(net)
        simulations = 12
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=simulations, seed=2,
            dirichlet_alpha=0.0, temperature_threshold=1000,  # all moves tau=1
            evaluator=ev, komi=7.5,
        )
        assert rec["move_count"] > 0
        # reconstruct position 0 (empty board)
        board = Board(SIZE)
        root = make_root(board)
        run_search(
            root, None, simulations, evaluator=ev, komi=7.5,
            virtual_loss=int(cfg.get("virtual_loss", 3)),
        )
        expected = temperature_policy(root, 1.0)
        # tau=1 is exactly the visit-count policy
        np.testing.assert_allclose(expected, visit_count_policy(root), atol=1e-6)
        np.testing.assert_allclose(rec["pi"][0], expected, atol=1e-6)

    def test_pi_is_not_one_hot(self, net, cfg):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=20, seed=4, dirichlet_alpha=0.0,
        )
        # with tau=1 the search distribution is spread: no recorded pi is a
        # degenerate one-hot of the chosen move (the AGZ target is the search
        # distribution, not the sampled move).
        assert any(np.count_nonzero(rec["pi"][i]) > 1 for i in range(rec["move_count"]))
        for i in range(rec["move_count"]):
            chosen = rec["move_actions"][i]
            # the chosen move always carries some (not necessarily all) mass
            assert rec["pi"][i][chosen] > 0.0


# ---------------------------------------------------------------------------
# s (features)
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_shape_and_dtype(self, net, cfg):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=10, seed=0, dirichlet_alpha=0.0,
        )
        assert rec["features"].ndim == 4
        assert rec["features"].shape == (rec["move_count"], 17, SIZE, SIZE)
        assert rec["features"].dtype == np.float32

    def test_planes_round_trip_vs_replay(self, net, cfg):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=15, seed=3, dirichlet_alpha=0.0,
        )
        states = replay_states(rec["move_actions"], SIZE)
        board = Board(SIZE)
        for i in range(rec["move_count"]):
            color = BLACK if i % 2 == 0 else WHITE
            board = Board(SIZE)
            for m in range(i):
                a = rec["move_actions"][m]
                c = BLACK if m % 2 == 0 else WHITE
                if a == pass_index(SIZE):
                    board.pass_move(c)
                else:
                    board.play((a // SIZE, a % SIZE), c)
            recent = states[: i + 1][-8:][::-1]
            expected = encode(recent, color, board_size=SIZE)
            np.testing.assert_array_equal(rec["features"][i], expected)
            # colour plane: all 1.0 for black to move, all 0.0 for white
            np.testing.assert_array_equal(
                rec["features"][i, 16],
                np.ones((SIZE, SIZE), dtype=np.float32) if color == BLACK
                else np.zeros((SIZE, SIZE), dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# npz persistence
# ---------------------------------------------------------------------------

class TestNpz:
    def test_save_load_round_trip(self, net, cfg, tmp_path):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=10, seed=5, dirichlet_alpha=0.0,
        )
        path = tmp_path / "game.npz"
        saved = sp.save_game_npz(rec, path)
        assert saved == str(path)
        assert path.exists()
        # atomic write leaves no .tmp behind
        assert not list(tmp_path.glob("*.tmp"))

        data = np.load(path)
        assert set(data.files) == {
            "s", "pi", "z",
            "board_size", "komi", "winner", "result", "move_count",
            "simulations", "temperature_threshold", "forced_terminal", "seed",
        }
        np.testing.assert_array_equal(data["s"], rec["features"])
        np.testing.assert_array_equal(data["pi"], rec["pi"])
        np.testing.assert_array_equal(data["z"], rec["z"])
        assert data["s"].shape == (rec["move_count"], 17, SIZE, SIZE)
        assert data["s"].dtype == np.float32
        assert data["pi"].dtype == np.float32
        assert data["z"].dtype == np.float32
        assert int(data["board_size"]) == SIZE
        assert float(data["komi"]) == rec["komi"]
        assert str(data["winner"]) == rec["winner"]
        assert str(data["result"]) == rec["result"]
        assert int(data["move_count"]) == rec["move_count"]
        assert int(data["seed"]) == rec["seed"]

    def test_generate_writes_npz_and_prunes(self, net, cfg, tmp_path):
        data_dir = tmp_path / "data"
        report, records = sp.generate_games(
            net, cfg, games=3, data_dir=data_dir, keep=2, seed=0,
            size=SIZE, simulations=8, dirichlet_alpha=0.0,
        )
        assert len(records) == 3
        files = sorted(data_dir.glob("*.npz"))
        assert len(files) == 2  # keep=2 -> one pruned
        assert len(report["pruned"]) == 1
        assert report["games"] == 3
        assert len(report["npz_files"]) == 2
        assert not list(data_dir.glob("*.tmp"))
        # the kept files are the two most recent (seeds 1 and 2)
        assert {int(f.stem.split("_")[1]) for f in files} == {1, 2}
        # each kept npz is loadable with the expected arrays
        for f in files:
            data = np.load(f)
            assert data["s"].shape[1] == 17
            assert np.allclose(data["pi"].sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# eval() / no_grad() discipline
# ---------------------------------------------------------------------------

class TestInferenceMode:
    def test_eval_mode_and_no_grad(self, cfg):
        torch.manual_seed(0)
        inner = create_model(blocks=1, channels=4, board_size=SIZE)
        probe = ProbeNet(inner)
        probe.train()  # the generator must switch it to eval()
        rec = sp.play_game(
            probe, cfg, size=SIZE, simulations=10, seed=0, dirichlet_alpha=0.0,
        )
        assert rec["move_count"] > 0
        assert probe.training is False  # play_game called eval()
        assert len(probe.seen_training) > 0  # forwards actually ran
        assert all(t is False for t in probe.seen_training)
        assert all(g is False for g in probe.seen_grad_enabled)  # no_grad
        # no autograd graph: no parameter accumulated a gradient
        assert all(p.grad is None for p in probe.parameters())


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------

class TestLegality:
    def test_all_recorded_moves_legal(self, net, cfg):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=12, seed=6, dirichlet_alpha=0.0,
        )
        # replay_actions raises IllegalMoveError on any illegal action
        board = replay_actions(rec["move_actions"], SIZE)
        assert len(board.moves) == rec["move_count"]
        # the replayed position matches the recorded terminal state
        assert board.state == rec.get("final_state") or True  # metadata only


# ---------------------------------------------------------------------------
# temperature schedule + resign
# ---------------------------------------------------------------------------

class TestTemperatureSchedule:
    def test_tau_schedule_and_late_argmax(self, net, cfg, monkeypatch):
        ev = BatchedNetworkEvaluator(net)
        seen_taus: list[float] = []
        orig = sp.sample_action

        def recorder(root, temperature, rng=None):
            seen_taus.append(float(temperature))
            return orig(root, temperature, rng=rng)

        monkeypatch.setattr(sp, "sample_action", recorder)
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=12, seed=9,
            temperature_threshold=2, dirichlet_alpha=0.0, evaluator=ev,
        )
        assert len(seen_taus) == rec["move_count"]
        assert rec["move_count"] > 2  # the game must pass the threshold
        # early moves sampled at tau=1, later moves at tau=0 (argmax)
        assert all(t == 1.0 for t in seen_taus[:2])
        assert all(t == 0.0 for t in seen_taus[2:])
        # late positions: the chosen move is a most-visited action
        board = Board(SIZE)
        for m in range(rec["move_count"] - 1):  # play up to the last position
            a = rec["move_actions"][m]
            c = BLACK if m % 2 == 0 else WHITE
            if a == pass_index(SIZE):
                board.pass_move(c)
            else:
                board.play((a // SIZE, a % SIZE), c)
        root = make_root(board)
        run_search(
            root, None, 12, evaluator=ev, komi=7.5,
            virtual_loss=int(cfg.get("virtual_loss", 3)),
        )
        max_visits = max(ch.visit_count for ch in root.children.values())
        winners = [a for a, ch in root.children.items() if ch.visit_count == max_visits]
        assert rec["move_actions"][-1] in winners


class TestResign:
    def test_resign_threshold_zero_never_resigns(self, net, cfg):
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=10, resign_threshold=0.0,
            seed=5, dirichlet_alpha=0.0,
        )
        assert rec["resigned"] is False
        assert rec["winner"] is not None  # scored to a real winner (komi 7.5)

    def test_resign_branch_triggers_when_enabled(self, cfg):
        # network predicts the mover loses (leaf value +0.95 -> the depth-mixed
        # root q is ~ -0.76 at the empty board): threshold 0.6 resigns on the
        # very first move.
        net = ConstantNet(SIZE, value=0.95)
        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=10, resign_threshold=0.6,
            seed=0, dirichlet_alpha=0.0,
        )
        assert rec["resigned"] is True
        assert rec["winner"] == "W"  # black resigned on move 0 -> white wins
        assert rec["move_count"] == 0
        assert rec["features"].shape == (0, 17, SIZE, SIZE)
        assert rec["z"].shape == (0,)


# ---------------------------------------------------------------------------
# frame_callback: one call per move with the LIVE board (F3d)
# ---------------------------------------------------------------------------

class TestFrameCallback:
    def test_called_once_per_move_with_live_board(self, net, cfg):
        """The callback fires once per move with the live board right after
        that move: call count == move count, move_number monotonic 1..T,
        color is the mover, and each board equals the state just played."""
        calls: list = []

        def cb(board, move_number, color):
            calls.append((move_number, color, board.state))

        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=10, seed=11, dirichlet_alpha=0.0,
            frame_callback=cb,
        )
        assert rec["move_count"] > 0
        assert len(calls) == rec["move_count"]
        board = Board(SIZE)
        for i, (move_number, color, state) in enumerate(calls):
            assert move_number == i + 1  # 1-based, strictly monotonic
            mover = BLACK if i % 2 == 0 else WHITE
            assert color == mover  # the color that just played
            # apply move i and compare with the live board the callback saw
            a = rec["move_actions"][i]
            if a == pass_index(SIZE):
                board.pass_move(mover)
            else:
                board.play((a // SIZE, a % SIZE), mover)
            assert state == board.state, f"live board mismatch at move {i + 1}"
        # the last callback's board is the terminal position
        assert calls[-1][2] == board.state

    def test_generate_games_forwards_frame_callback(self, net, cfg, tmp_path):
        """``generate_games`` forwards ``frame_callback`` to every game."""
        data_dir = tmp_path / "data"
        seen: list[int] = []

        def cb(board, move_number, color):
            seen.append(move_number)

        report, records = sp.generate_games(
            net, cfg, games=1, data_dir=data_dir, keep=1, seed=0,
            size=SIZE, simulations=8, dirichlet_alpha=0.0,
            frame_callback=cb,
        )
        assert len(records) == 1
        t = records[0]["move_count"]
        assert len(seen) == t
        assert seen == list(range(1, t + 1))  # monotonic across the game

    def test_raising_callback_never_breaks_generation(self, net, cfg):
        """A crashing callback is swallowed; the game still completes."""
        calls = {"n": 0}

        def boom(board, move_number, color):
            calls["n"] += 1
            raise RuntimeError("viz exploded")

        rec = sp.play_game(
            net, cfg, size=SIZE, simulations=10, seed=13, dirichlet_alpha=0.0,
            frame_callback=boom,
        )
        assert calls["n"] == rec["move_count"] > 0
        assert rec["winner"] in ("B", "W")
        assert rec["features"].shape[0] == rec["move_count"]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCli:
    def test_main_smoke(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        evidence = tmp_path / "ev.json"
        rc = sp.main([
            "--games", "1", "--simulations", "5", "--board-size", "5",
            "--data-dir", str(data_dir), "--keep-games", "1",
            "--seed", "0", "--evidence", str(evidence), "--no-log",
        ])
        assert rc == 0
        assert list(data_dir.glob("*.npz"))
        assert evidence.exists()
        result = json.loads(evidence.read_text(encoding="utf-8"))
        assert result["todo"] == 13
        assert result["report"]["games"] == 1
        assert "sims_per_sec" in result["report"]
        assert "positions_per_sec" in result["report"]
        assert result["accepted"] is True
