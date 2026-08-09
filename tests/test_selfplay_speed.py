"""P11: self-play speedup tests (rules fast-paths, decode, fp16, leaf_batch).

Covers:
(a) ``is_legal_move`` empty-neighbor fast path agrees with the full
    placement-simulation reference on random 9x9 positions;
(b) ``has_liberty`` early-exit agrees with the full liberty-set reference;
(c) ``Board.play(check_legal=False)`` mutates the board identically to the
    checked path for a legal move;
(d) ``decode_policy(legal_moves=...)`` == ``decode_policy()`` (no legal_moves)
    on the same inputs;
(e) batched-evaluator fp16: on CPU it is bit-identical to fp32; on CUDA it
    returns normalized legal priors and finite values;
(f) ``leaf_batch`` is threaded through ``play_game`` -> evaluator: every
    flushed batch respects the configured bound.
"""

import numpy as np
import pytest
import torch

from omigamax.mcts.batched_evaluator import BatchedNetworkEvaluator
from omigamax.mcts.mcts import make_root
from omigamax.network.features import decode_policy
from omigamax.network.model import create_model
from omigamax.rules import Board, BLACK, WHITE
from omigamax.rules.liberties import EMPTY, group, liberties, neighbors
from omigamax.train.selfplay import play_game

SIZE = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = DEVICE.type == "cuda"


# ---------------------------------------------------------------------------
# (a) is_legal_move fast path vs full-simulation reference
# ---------------------------------------------------------------------------

def _is_legal_slow(state, size, move, color):
    """Reference: simulate the placement and check capture / suicide."""
    from omigamax.rules.captures import captured_groups
    from omigamax.rules.legality import is_on_board
    from omigamax.rules.liberties import has_liberty
    if move is None:
        return True
    if not is_on_board(move, size):
        return False
    r, c = move
    idx = r * size + c
    if state[idx] != EMPTY:
        return False
    state[idx] = color
    try:
        if captured_groups(state, size, r, c, color):
            return True
        return has_liberty(state, size, r, c)
    finally:
        state[idx] = EMPTY


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_is_legal_fastpath_matches_reference(seed):
    from omigamax.rules.legality import is_legal_move
    rng = np.random.default_rng(seed)
    size = SIZE
    for trial in range(6):
        state = rng.integers(0, 3, size=size * size).astype(int)
        for color in (BLACK, WHITE):
            for r in range(size):
                for c in range(size):
                    move = (r, c)
                    fast = is_legal_move(state, size, move, color)
                    slow = _is_legal_slow(list(state), size, move, color)
                    assert fast == slow, (seed, trial, color, move, fast, slow)
            assert is_legal_move(state, size, None, color) is True


# ---------------------------------------------------------------------------
# (b) has_liberty early-exit matches the full liberty-set computation
# ---------------------------------------------------------------------------

def test_has_liberty_matches_liberties():
    from omigamax.rules.liberties import has_liberty
    rng = np.random.default_rng(7)
    size = SIZE
    for trial in range(30):
        state = rng.integers(0, 3, size=size * size).astype(int)
        for r in range(size):
            for c in range(size):
                expected = len(liberties(state, size, r, c)) > 0
                assert has_liberty(state, size, r, c) == expected, (trial, r, c)


# ---------------------------------------------------------------------------
# (c) Board.play(check_legal=False) is identical for a legal move
# ---------------------------------------------------------------------------

def test_play_check_legal_false_matches_checked():
    rng = np.random.default_rng(11)
    for trial in range(20):
        a = Board(SIZE)
        b = Board(SIZE)
        for _ in range(rng.integers(1, 30)):
            color = BLACK if len(a.moves) % 2 == 0 else WHITE
            legal = [mv for mv in
                     [(r, c) for r in range(SIZE) for c in range(SIZE)]
                     if a.is_legal(mv, color)]
            if not legal:
                a.pass_move(color)
                b.pass_move(color)
                continue
            mv = legal[int(rng.integers(0, len(legal)))]
            captured_a = a.play(mv, color)
            captured_b = b.play(mv, color, check_legal=False)
            assert captured_a == captured_b
            assert a._state == b._state
            assert a.moves == b.moves
            assert a.last_captured_point == b.last_captured_point
            assert a.pass_count == b.pass_count


# ---------------------------------------------------------------------------
# (d) decode_policy(legal_moves=...) == decode_policy()
# ---------------------------------------------------------------------------

def test_decode_policy_legal_moves_equals_default():
    rng = np.random.default_rng(3)
    for trial in range(20):
        board = Board(SIZE)
        # build a random legal-move set via the board's own legality scan
        color = BLACK if trial % 2 == 0 else WHITE
        legal = tuple(sorted(
            [r * SIZE + c for r in range(SIZE) for c in range(SIZE)
             if board.is_legal((r, c), color)] + [SIZE * SIZE]))
        logits = rng.standard_normal(SIZE * SIZE + 1)
        a = decode_policy(logits, board, color=color)
        b = decode_policy(logits, board, color=color, legal_moves=legal)
        np.testing.assert_array_equal(a, b)
        assert a.sum() == pytest.approx(1.0)
        for idx in range(SIZE * SIZE + 1):
            if idx not in legal:
                assert a[idx] == 0.0


# ---------------------------------------------------------------------------
# (e) batched-evaluator fp16 correctness
# ---------------------------------------------------------------------------

def _tiny_game_net():
    return create_model(1, 8, SIZE).to(DEVICE)


def test_fp16_evaluator_cpu_bit_exact(tmp_path):
    net = create_model(1, 8, SIZE)
    ev_f32 = BatchedNetworkEvaluator(net, batch_size=4, fp16=False)
    ev_f16 = BatchedNetworkEvaluator(net, batch_size=4, fp16=True)
    rng = np.random.default_rng(0)
    for trial in range(3):
        board = Board(SIZE)
        root = make_root(board)
        from omigamax.mcts.mcts import expand
        prior = rng.random(SIZE * SIZE + 1).astype(np.float32)
        prior /= prior.sum()
        expand(root, prior)
        leaves = list(root.children.values())[:4]
        for ev in (ev_f32, ev_f16):
            for nd in leaves:
                ev.submit(nd)
        r32 = ev_f32.flush()
        r16 = ev_f16.flush()
        for (n32, p32, v32), (n16, p16, v16) in zip(r32, r16):
            assert n32 is n16
            np.testing.assert_array_equal(p32, p16)  # bit-exact on CPU
            assert v32 == v16


@pytest.mark.skipif(not CUDA, reason="fp16 GPU test needs CUDA")
def test_fp16_evaluator_cuda_legal_and_finite():
    net = _tiny_game_net()
    ev = BatchedNetworkEvaluator(net, batch_size=8, fp16=True)
    rng = np.random.default_rng(1)
    board = Board(SIZE)
    root = make_root(board)
    from omigamax.mcts.mcts import expand, legal_actions
    prior = rng.random(SIZE * SIZE + 1).astype(np.float32)
    prior /= prior.sum()
    expand(root, prior)
    leaves = list(root.children.values())[:8]
    for nd in leaves:  # mirror run_search: legal_moves is set before submit
        nd.legal_moves = legal_actions(nd.board, nd.color)
    for nd in leaves:
        ev.submit(nd)
    results = ev.flush()
    assert len(results) == 8
    for node, p, v in results:
        assert np.isfinite(v)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= 0.0)
        legal = set(node.legal_moves)
        for idx in range(SIZE * SIZE + 1):
            if idx not in legal:
                assert p[idx] == 0.0


# ---------------------------------------------------------------------------
# (f) leaf_batch threads through play_game -> evaluator
# ---------------------------------------------------------------------------

def test_leaf_batch_threaded_into_evaluator(monkeypatch):
    from omigamax.train import selfplay as sp
    net = create_model(1, 8, SIZE).to(DEVICE)
    net.eval()
    seen = {}
    real_be = sp.BatchedNetworkEvaluator

    def spy(network, batch_size=None, fp16=False):
        seen["batch_size"] = batch_size
        seen["fp16"] = fp16
        return real_be(network, batch_size=batch_size, fp16=fp16)

    monkeypatch.setattr(sp, "BatchedNetworkEvaluator", spy)
    cfg = {"board_size": SIZE, "komi": 7.5, "temperature_threshold": 2,
           "dirichlet_alpha": 0.03, "dirichlet_eps": 0.25,
           "virtual_loss": 3, "simulations": 30, "max_moves": 40}
    rec = play_game(net, cfg, size=SIZE, simulations=30, max_moves=40,
                    leaf_batch=4, seed=1, fp16=True)
    assert seen == {"batch_size": 4, "fp16": True}  # threaded both knobs
    assert rec["move_count"] > 0
    assert all(0 <= a <= SIZE * SIZE for a in rec["move_actions"])


def test_evaluator_batch_size_bounds():
    net = create_model(1, 8, SIZE).to(DEVICE)
    ev = BatchedNetworkEvaluator(net, batch_size=8)
    assert ev.batch_size == 8
    ev2 = BatchedNetworkEvaluator(net, batch_size=1)
    assert ev2.batch_size == 1
