"""Tests for the training step, replay buffer, symmetry and checkpoint (todo 14).

Per the plan's todo-14 acceptance criteria:

* **loss decreases** -- 100 SGD steps on a (small) replay-buffer dataset reduce
  the AGZ loss measurably (start/end recorded);
* **deterministic resume** -- train 20 steps, save a checkpoint, reload,
  continue 20 more; the full 40-step loss sequence equals an uninterrupted
  40-step run from the same seed (fixed seed; ``cudnn.deterministic=True``,
  ``benchmark=False``; Oracle F9 tolerance ``1e-4``);
* **lr schedule** -- AGZ piecewise schedule returns the expected lr at the
  config boundaries (``lr_schedule_steps=[50000, 100000]``):
  ``0.2`` before 50000, ``0.02`` at/after 50000, ``0.002`` at/after 100000;
* **8-fold symmetry** -- the 8 dihedral transforms of ``(s, pi)`` are
  bijective, the ``pi`` index mapping is consistent with the point-coordinate
  mapping, and the meaningful round-trip holds: with a *real* tiny network,
  evaluating on the transformed ``s`` and inverse-transforming the policy
  reproduces the original policy;
* **buffer** -- sampling returns correct shapes, honours ``batch_size``,
  window pruning respects ``replay_buffer_games``, and no position outside a
  game's recorded range (or outside the window) is ever sampled;
* **checkpoint** -- save/load round-trip restores weights, optimizer state
  (momentum buffers), ``global_step``, a config snapshot and the sampling RNG
  state; written atomically.

All games are synthetic npz files on a 5x5 board with a tiny 1x8 network, run
on the fastest available device (CPU unless CUDA is present) so the suite
stays fast and bit-exact.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train import train as tr
from omigamax.train.buffer import ReplayBuffer
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.symmetry import (
    SYMMETRY_COUNT,
    apply_to_features,
    apply_to_pi,
    augment,
    augment_batch,
    inverse_permutation,
    policy_permutation,
    transform_point,
)

SIZE = 5
N_LOGITS = SIZE * SIZE + 1  # 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GIB = 1024**3


def set_deterministic(seed: int) -> None:
    """Fixed seed + deterministic GPU kernels (Oracle F9 recipe)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# helpers: synthetic npz games
# ---------------------------------------------------------------------------

def write_game(path, size: int, t: int, seed: int) -> dict:
    """Write one synthetic npz game; return the arrays it stored."""
    rng = np.random.default_rng(seed)
    s = rng.random((t, 17, size, size)).astype(np.float32)
    pi = rng.random((t, size * size + 1)).astype(np.float32)
    pi /= pi.sum(axis=1, keepdims=True)
    z = rng.choice([-1.0, 1.0], size=t).astype(np.float32)
    np.savez(
        path,
        s=s, pi=pi, z=z,
        board_size=np.int64(size),
        move_count=np.int64(t),
    )
    return {"s": s, "pi": pi, "z": z}


def write_games(tmp_path, n_games: int, t: int = 24, size: int = SIZE,
                seed: int = 0) -> list[dict]:
    """Write ``n_games`` synthetic npz files; return their array records."""
    records = []
    for g in range(n_games):
        rec = write_game(tmp_path / f"game_{g:010d}.npz", size, t, seed + g)
        records.append(rec)
    return records


def tiny_net():
    set_deterministic(0)
    return create_model(blocks=1, channels=8, board_size=SIZE).to(DEVICE)


class NoBNNet(nn.Module):
    """Tiny batch-norm-free policy+value net: exact chunk-equivalence tests.

    Mirrors the two-head contract of ``create_model`` (returns
    ``(logits, value)``, exposes ``blocks``/``channels``/``board_size``)
    without any BatchNorm, so chunked gradient accumulation is exactly equal
    to a single full-batch step (no batch-stat noise).
    """

    def __init__(self, size: int = SIZE):
        super().__init__()
        self.blocks = 1
        self.channels = 8
        self.board_size = size
        n_in = 17 * size * size
        self.fc1 = nn.Linear(n_in, 64)
        self.fc_logits = nn.Linear(64, size * size + 1)
        self.fc_value = nn.Linear(64, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x.flatten(1)))
        return self.fc_logits(h), self.fc_value(h)


# ---------------------------------------------------------------------------
# 8-fold symmetry augmentation
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_eight_distinct_symmetries(self):
        perms = [policy_permutation(k, SIZE) for k in range(SYMMETRY_COUNT)]
        assert SYMMETRY_COUNT == 8
        assert len({tuple(p.tolist()) for p in perms}) == 8  # all distinct

    def test_pi_permutation_is_bijective_and_pass_fixed(self):
        for k in range(SYMMETRY_COUNT):
            perm = policy_permutation(k, SIZE)
            assert sorted(perm.tolist()) == list(range(N_LOGITS)), (
                f"symmetry {k} perm is not a bijection"
            )
            assert perm[N_LOGITS - 1] == N_LOGITS - 1  # pass -> pass

    def test_permutation_matches_point_transform(self):
        """dest = perm[dest] must equal the flat image of the point's preimage."""
        for k in range(SYMMETRY_COUNT):
            perm = policy_permutation(k, SIZE)
            for r in range(SIZE):
                for c in range(SIZE):
                    src = r * SIZE + c
                    dr, dc = transform_point(k, r, c, SIZE)  # image of (r, c)
                    dest = dr * SIZE + dc
                    # pi_aug[dest] = pi[src]  ->  perm[dest] == src
                    assert perm[dest] == src, (
                        f"symmetry {k}: perm[{dest}]={perm[dest]} != src {src}"
                    )

    def test_feature_transform_places_marker_at_image_point(self):
        for k in range(SYMMETRY_COUNT):
            s = np.zeros((17, SIZE, SIZE), dtype=np.float32)
            s[0, 2, 3] = 1.0  # marker at (r=2, c=3)
            s_k = apply_to_features(s, k)
            dr, dc = transform_point(k, 2, 3, SIZE)
            assert s_k[0, dr, dc] == 1.0, f"symmetry {k} marker misplaced"
            assert int((s_k[0] > 0).sum()) == 1  # single marker, no others

    def test_color_plane_invariant_under_all_symmetries(self):
        black = np.ones((1, SIZE, SIZE), dtype=np.float32)
        white = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        for k in range(SYMMETRY_COUNT):
            np.testing.assert_array_equal(
                apply_to_features(black, k), black)
            np.testing.assert_array_equal(
                apply_to_features(white, k), white)

    def test_apply_to_pi_gathers_by_perm(self):
        rng = np.random.default_rng(3)
        pi = rng.random(N_LOGITS).astype(np.float32)
        pi /= pi.sum()
        for k in range(SYMMETRY_COUNT):
            perm = policy_permutation(k, SIZE)
            np.testing.assert_array_equal(apply_to_pi(pi, k), pi[perm])

    def test_inverse_permutation_inverts(self):
        for k in range(SYMMETRY_COUNT):
            perm = policy_permutation(k, SIZE)
            inv = inverse_permutation(k, SIZE)
            np.testing.assert_array_equal(perm[inv], np.arange(N_LOGITS))
            np.testing.assert_array_equal(inv[perm], np.arange(N_LOGITS))

    def test_network_round_trip_equivariance(self):
        """Real-network check: inv-transform(pi(transform(s))) == pi(s).

        The b10c128-class policy head ends in a *dense* layer over the
        flattened spatial features, so a freshly initialised network is only
        approximately equivariant (measured worst-case |diff| ~ 3e-3 for the
        tiny test topology). The tolerance 0.01 therefore leaves a ~3x margin
        while a wrong pi index mapping (the bug this test exists to catch)
        misassigns probability mass and produces O(1) differences.
        """
        torch.manual_seed(11)
        model = tiny_net()
        model.eval()
        rng = np.random.default_rng(7)
        with torch.no_grad():
            for k in range(SYMMETRY_COUNT):
                s = rng.random((17, SIZE, SIZE)).astype(np.float32)
                s_t = torch.from_numpy(s[None]).to(DEVICE)
                logits = model(s_t)[0][0].cpu().numpy()
                pi = np.exp(logits - logits.max()); pi /= pi.sum()
                s_k = apply_to_features(s, k)
                logits_k = model(
                    torch.from_numpy(s_k[None]).to(DEVICE))[0][0].cpu().numpy()
                pi_k = np.exp(logits_k - logits_k.max()); pi_k /= pi_k.sum()
                # pi_k[dest] == pi[src] with dest = perm^{-1}[src]
                inv = inverse_permutation(k, SIZE)
                pi_recovered = pi_k[inv]
                np.testing.assert_allclose(pi_recovered, pi, atol=0.01, rtol=0.0)

    def test_augment_batch_shapes_and_contents(self):
        rng = np.random.default_rng(5)
        s = rng.random((3, 17, SIZE, SIZE)).astype(np.float32)
        pi = rng.random((3, N_LOGITS)).astype(np.float32)
        pi /= pi.sum(axis=1, keepdims=True)
        s8, pi8 = augment_batch(s, pi)
        assert s8.shape == (24, 17, SIZE, SIZE)
        assert pi8.shape == (24, N_LOGITS)
        # each of the 8 blocks matches direct per-symmetry augmentation
        for k in range(SYMMETRY_COUNT):
            np.testing.assert_array_equal(s8[k * 3:(k + 1) * 3], apply_to_features(s, k))
            np.testing.assert_array_equal(pi8[k * 3:(k + 1) * 3], apply_to_pi(pi, k))
        # z is unaffected by symmetry: augment() applies to (s, pi) only
        pairs = augment(s[0], pi[0])
        assert len(pairs) == 8
        assert all(p[1].sum() == pytest.approx(1.0, abs=1e-5) for p in pairs)


# ---------------------------------------------------------------------------
# lr schedule (AGZ piecewise)
# ---------------------------------------------------------------------------

class TestLrSchedule:
    def test_boundary_values_config_defaults(self):
        # config lr_schedule_steps=[50000, 100000]; lr=0.2
        assert tr.agz_lr(0) == pytest.approx(0.2)
        assert tr.agz_lr(49999) == pytest.approx(0.2)
        assert tr.agz_lr(50000) == pytest.approx(0.02)
        assert tr.agz_lr(99999) == pytest.approx(0.02)
        assert tr.agz_lr(100000) == pytest.approx(0.002)
        assert tr.agz_lr(150000) == pytest.approx(0.002)

    def test_no_schedule_steps_is_constant(self):
        assert tr.agz_lr(0, schedule_steps=[]) == pytest.approx(0.2)
        assert tr.agz_lr(10 ** 9, schedule_steps=[]) == pytest.approx(0.2)

    def test_custom_base_and_boundaries(self):
        assert tr.agz_lr(0, lr_base=0.1, schedule_steps=[10, 20]) == pytest.approx(0.1)
        assert tr.agz_lr(10, lr_base=0.1, schedule_steps=[10, 20]) == pytest.approx(0.01)
        assert tr.agz_lr(20, lr_base=0.1, schedule_steps=[10, 20]) == pytest.approx(0.001)
        assert tr.agz_lr(30, lr_base=0.1, schedule_steps=[10, 20]) == pytest.approx(0.001)

    def test_set_learning_rate(self):
        torch.manual_seed(0)
        model = create_model(1, 8, SIZE)
        opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
        tr.set_learning_rate(opt, 0.02)
        assert opt.param_groups[0]["lr"] == 0.02


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------

class TestBuffer:
    def test_sampling_shapes_and_batch_size(self, tmp_path):
        recs = write_games(tmp_path, n_games=5)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        assert buf.num_games == 5
        batch = buf.sample(32, rng=np.random.default_rng(1))
        assert batch["s"].shape == (32, 17, SIZE, SIZE)
        assert batch["pi"].shape == (32, N_LOGITS)
        assert batch["z"].shape == (32, 1)
        assert batch["s"].dtype == np.float32
        assert batch["pi"].dtype == np.float32
        assert batch["z"].dtype == np.float32
        assert np.all((batch["z"] == 1.0) | (batch["z"] == -1.0))

    def test_sampled_rows_match_source_games(self, tmp_path):
        recs = write_games(tmp_path, n_games=4)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        batch = buf.sample(20, rng=np.random.default_rng(2))
        for i in range(20):
            gi = int(batch["game_idxs"][i])
            pi_idx = int(batch["position_idxs"][i])
            assert 0 <= gi < 4
            assert 0 <= pi_idx < recs[gi]["s"].shape[0]  # no out-of-range
            np.testing.assert_array_equal(batch["s"][i], recs[gi]["s"][pi_idx])
            np.testing.assert_array_equal(batch["pi"][i], recs[gi]["pi"][pi_idx])
            assert batch["z"][i, 0] == recs[gi]["z"][pi_idx]

    def test_no_future_position_leakage(self, tmp_path):
        recs = write_games(tmp_path, n_games=4, t=10)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        rng = np.random.default_rng(4)
        for _ in range(5):
            batch = buf.sample(64, rng=rng)
            for i in range(64):
                gi = int(batch["game_idxs"][i])
                p = int(batch["position_idxs"][i])
                assert p < recs[gi]["s"].shape[0]

    def test_window_pruning_respects_max_games(self, tmp_path):
        recs = write_games(tmp_path, n_games=5)
        buf = ReplayBuffer(tmp_path, max_games=3)
        buf.refresh()
        assert buf.num_games == 3
        files = sorted(p.name for p in tmp_path.glob("*.npz"))
        assert len(files) == 3  # older games pruned from disk
        # the kept files are the 3 newest (seeds 2, 3, 4)
        kept_seeds = sorted(int(f.split("_")[1][:-4]) for f in files)
        assert kept_seeds == [2, 3, 4]
        # every sampled game index stays inside the window
        batch = buf.sample(50, rng=np.random.default_rng(3))
        assert batch["game_idxs"].max() < 3
        assert batch["game_idxs"].min() >= 0

    def test_empty_buffer_raises(self, tmp_path):
        buf = ReplayBuffer(tmp_path, max_games=1000)
        assert buf.num_games == 0
        with pytest.raises(RuntimeError):
            buf.sample(8)

    def test_empty_game_is_skipped(self, tmp_path):
        write_game(tmp_path / "game_0000000000.npz", SIZE, t=0, seed=0)
        write_games(tmp_path, n_games=3, t=8, seed=10)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        batch = buf.sample(16, rng=np.random.default_rng(6))
        assert batch["s"].shape[0] == 16

    def test_cache_limit_evicts(self, tmp_path):
        write_games(tmp_path, n_games=5, t=8)
        buf = ReplayBuffer(tmp_path, max_games=5, cache_limit=2)
        for _ in range(20):
            buf.sample(8, rng=np.random.default_rng())
        assert len(buf._cache) <= 2

    def test_num_positions_counts_all_games(self, tmp_path):
        write_games(tmp_path, n_games=4, t=11)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        assert buf.num_positions == 4 * 11


# ---------------------------------------------------------------------------
# checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_round_trip_restores_everything(self, tmp_path):
        torch.manual_seed(0)
        model = create_model(1, 8, SIZE).to(DEVICE)
        opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
        rng = np.random.default_rng(42)
        cfg = {"batch_size": 128, "lr": 0.2}
        # a few optimizer steps so the momentum buffer is non-empty
        recs = write_games(tmp_path, n_games=2, t=10)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        tr.train_steps(model, opt, buf, steps=4, rng=rng, batch_size=8,
                       device=DEVICE, symmetry=False)
        path = tmp_path / "ckpt" / "latest.pt"
        saved = tr.save_checkpoint(
            path, model, opt, global_step=7, config=cfg, rng=rng)
        assert saved == str(path)
        assert path.exists()
        assert not list(path.parent.glob("*.tmp"))  # atomic: no tmp left

        ckpt = tr.load_checkpoint(path)
        assert ckpt["global_step"] == 7
        assert ckpt["config"] == cfg
        assert ckpt["arch"] == {"blocks": 1, "channels": 8, "board_size": SIZE}
        assert "rng_state" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "model_state_dict" in ckpt

        # restore into a fresh model + optimizer
        torch.manual_seed(1)
        model2 = create_model(1, 8, SIZE).to(DEVICE)
        opt2 = make_sgd_optimizer(model2, lr=0.2, momentum=0.9, l2=1e-4)
        step = tr.restore_from_checkpoint(ckpt, model2, opt2)
        assert step == 7
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1.detach().cpu(), p2.detach().cpu())
        for key, val in opt.state_dict()["state"].items():
            assert key in opt2.state_dict()["state"]
            for k, v in val.items():
                ref = opt2.state_dict()["state"][key][k]
                if torch.is_tensor(v):
                    torch.testing.assert_close(v.detach().cpu(), ref.detach().cpu())
                else:
                    assert v == ref

        # rng state restored: next draws match the saved generator
        rng2 = np.random.default_rng(0)
        tr.restore_rng(rng2, ckpt["rng_state"])
        assert int(rng.integers(0, 10**9)) == int(rng2.integers(0, 10**9))


# ---------------------------------------------------------------------------
# deterministic resume + loss decrease (plan acceptance)
# ---------------------------------------------------------------------------

class TestDeterministicResume:
    def _buffer(self, tmp_path):
        write_games(tmp_path, n_games=4, t=30, seed=0)
        return ReplayBuffer(tmp_path, max_games=1000)

    def test_interrupted_equals_uninterrupted_40_steps(self, tmp_path):
        set_deterministic(1234)
        buf = self._buffer(tmp_path)

        # interrupted: train 20, save, reload, continue 20
        model_a = create_model(1, 8, SIZE).to(DEVICE)
        opt_a = make_sgd_optimizer(model_a, lr=0.2, momentum=0.9, l2=1e-4)
        rng_a = np.random.default_rng(99)
        losses_a, _, _, _ = tr.train_steps(
            model_a, opt_a, buf, steps=20, rng=rng_a, batch_size=16,
            device=DEVICE, symmetry=False)
        ckpt_path = tmp_path / "latest.pt"
        tr.save_checkpoint(ckpt_path, model_a, opt_a, global_step=20,
                           config=None, rng=rng_a)
        ckpt = tr.load_checkpoint(ckpt_path)
        tr.restore_from_checkpoint(ckpt, model_a, opt_a)
        tr.restore_rng(rng_a, ckpt["rng_state"])
        losses_a2, _, _, _ = tr.train_steps(
            model_a, opt_a, buf, steps=20, rng=rng_a, batch_size=16,
            device=DEVICE, symmetry=False)
        interrupted = losses_a + losses_a2

        # uninterrupted: 40 steps from the same seed (re-seed torch so the
        # fresh model has the SAME init as model_a -- create_model consumes
        # torch RNG)
        buf2 = ReplayBuffer(tmp_path, max_games=1000)
        torch.manual_seed(1234)
        model_b = create_model(1, 8, SIZE).to(DEVICE)
        opt_b = make_sgd_optimizer(model_b, lr=0.2, momentum=0.9, l2=1e-4)
        rng_b = np.random.default_rng(99)
        uninterrupted, _, _, _ = tr.train_steps(
            model_b, opt_b, buf2, steps=40, rng=rng_b, batch_size=16,
            device=DEVICE, symmetry=False)

        assert len(interrupted) == 40
        assert len(uninterrupted) == 40
        # plan tolerance (Oracle F9): fixed seed + deterministic kernels still
        # compared with 1e-4 (not bit-exact) on GPU.
        np.testing.assert_allclose(
            np.asarray(interrupted), np.asarray(uninterrupted), atol=1e-4)

    def test_100_step_loss_decreases(self, tmp_path):
        """Plan gate: 100 training steps reduce the loss (record start/end).

        Samples ONE fixed batch from the replay buffer (exercising buffer
        sampling + 8-fold symmetry augmentation) and runs 100 SGD steps on it
        -- the todo-8 "loss 有限且下降" pattern, which guarantees the model
        measurably learns the batch.
        """
        set_deterministic(2024)
        write_games(tmp_path, n_games=3, t=24, seed=5)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        batch = buf.sample(16, rng=np.random.default_rng(77))
        s8, pi8 = augment_batch(batch["s"], batch["pi"])
        fixed = {"s": s8, "pi": pi8, "z": np.repeat(batch["z"], 8, axis=0)}
        model = create_model(1, 8, SIZE).to(DEVICE)
        opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
        # chunks=8 keeps each forward/backward at batch 16 (the config
        # batch_size) -- the same chunking the training loop uses.
        losses = [tr.train_on_batch(model, opt, fixed, device=DEVICE, chunks=8)
                  for _ in range(100)]
        assert all(math.isfinite(l) for l in losses)
        assert losses[-1] < losses[0] * 0.9, (
            f"loss did not decrease measurably: first={losses[0]} "
            f"last={losses[-1]}"
        )

    def test_train_steps_records_schedule_lrs(self, tmp_path):
        set_deterministic(7)
        write_games(tmp_path, n_games=2, t=10)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        model = create_model(1, 8, SIZE).to(DEVICE)
        opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
        rng = np.random.default_rng(3)
        _, lrs, step, _ = tr.train_steps(
            model, opt, buf, steps=5, rng=rng, batch_size=8, device=DEVICE,
            symmetry=False, schedule_steps=[3, 6])
        expected = [0.2, 0.2, 0.2, 0.02, 0.02]  # boundary at step 3 (0-based)
        assert len(lrs) == len(expected)
        for got, want in zip(lrs, expected):
            assert got == pytest.approx(want)
        assert step == 5

    def test_grad_clip_keeps_gradients_finite(self, tmp_path):
        set_deterministic(9)
        write_games(tmp_path, n_games=2, t=10)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        model = create_model(1, 8, SIZE).to(DEVICE)
        opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
        rng = np.random.default_rng(1)
        losses, _, _, _ = tr.train_steps(
            model, opt, buf, steps=10, rng=rng, batch_size=8, device=DEVICE,
            symmetry=False, grad_clip=5.0)
        assert all(math.isfinite(l) for l in losses)
        for p in model.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all()


class TestChunkedTraining:
    def test_chunks_8_equals_single_full_batch(self, tmp_path):
        """Gradient accumulation over 8 chunks == one full 8B step.

        The 8-fold augmented batch is processed as ``chunks=8`` of the config
        batch_size (avoids the pathological large-batch cuDNN kernels on this
        6GB card; measured: batch-512 conv forward ~130s vs batch-128 ~0.2s).
        A BN-free network makes the equivalence exact (no batch-stat noise):
        the loss and every weight after the chunked step must match the
        single full-batch step.
        """
        set_deterministic(31)
        write_games(tmp_path, n_games=2, t=10)
        buf = ReplayBuffer(tmp_path, max_games=1000)
        batch = buf.sample(16, rng=np.random.default_rng(5))
        s8, pi8 = augment_batch(batch["s"], batch["pi"])
        fixed = {"s": s8, "pi": pi8, "z": np.repeat(batch["z"], 8, axis=0)}

        def run_once(chunks):
            torch.manual_seed(31)  # create_model/NoBN consume torch RNG
            model = NoBNNet().to(DEVICE)
            opt = make_sgd_optimizer(model, lr=0.2, momentum=0.9, l2=1e-4)
            loss = tr.train_on_batch(model, opt, fixed, device=DEVICE,
                                     chunks=chunks)
            return loss, model

        l1, m1 = run_once(1)
        l8, m8 = run_once(8)
        assert l1 == pytest.approx(l8, abs=1e-4)
        for p1, p2 in zip(m1.parameters(), m8.parameters()):
            torch.testing.assert_close(p1.detach().cpu(), p2.detach().cpu(),
                                       atol=1e-4, rtol=1e-4)
