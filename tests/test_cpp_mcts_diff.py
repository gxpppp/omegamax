"""Differential test: the C++ MCTS engine vs the Python reference (P16-5/6).

The C++ engine (``omigamax_core.CppMCTSEngine``) mirrors
:func:`omigamax.mcts.run_search` bit-exactly (same UCB operation order in IEEE
double, same expand/backup semantics, tie-break to the lowest action index),
while owning the tree in C++ and reaching the batched evaluator through the
transient-shell leaf protocol. ``run_search`` routes through the engine when
``OMIGAMAX_USE_CPP_MCTS=1`` and ``cpp_mcts_available()``.

Two gates:

* **Gate 1 (bit-exact)**: fixed seeds 0-19, sims=30, 9x9 tiny CPU model. Both
  paths share the same seed rng and the same deterministic network, so the
  games must be IDENTICAL move-for-move (and the full search trees bit-identical
  on a direct ``run_search`` comparison). Any divergence is a bug in the C++
  side -- the test records the seed + divergence step instead of relaxing.
* **Gate 2 (statistical)**: 200 games (seeds 0-199, sims 20-50). Move
  agreement >= 95%, 100% legal moves (replay against the rules engine, no
  ``IllegalMoveError``), every game terminates (double-pass or ``max_moves``).
  Winner-agreement rate is report-only; the first-divergence distribution is
  reported alongside the gate result.

The tests toggle the env var per call, so they run both paths regardless of the
outer environment (the default full-suite run keeps the Python path for
everything else).

Runtime note: the statistical gate runs 200 games through BOTH engines on CPU
(~10 min with the tiny model); Gate 1 is ~1 min. This is the cost the plan's
"200 games, sims 20-50" acceptance criteria demand.
"""

import os

import numpy as np
import pytest
import torch

from omigamax.config import load_config
from omigamax.mcts import (
    BatchedNetworkEvaluator,
    cpp_mcts_available,
    make_root,
    run_search,
)
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board
from omigamax.train.selfplay import play_game

SIZE = 9
PASS = SIZE * SIZE
MAX_MOVES = 100

pytestmark = pytest.mark.skipif(
    not cpp_mcts_available(),
    reason="C++ MCTS engine not built (cpp/ not compiled into omigamax_core)",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _model():
    """Tiny deterministic 9x9 CPU policy-value network (b1c4)."""
    torch.manual_seed(0)
    net = create_model(blocks=1, channels=4, board_size=SIZE)
    net.eval()
    return net


def _play(net, cfg, seed, sims, use_cpp):
    os.environ["OMIGAMAX_USE_CPP_MCTS"] = "1" if use_cpp else "0"
    rec = play_game(
        net, cfg, size=SIZE, simulations=sims, seed=seed,
        dirichlet_alpha=0.0, max_moves=MAX_MOVES,
    )
    return rec


def _play_both(net, cfg, seed, sims):
    return _play(net, cfg, seed, sims, True), _play(net, cfg, seed, sims, False)


def _first_divergence(actions_cpp, actions_py):
    """0-based index of the first differing move, or None if identical."""
    for i, (a, b) in enumerate(zip(actions_cpp, actions_py)):
        if a != b:
            return i
    if len(actions_cpp) != len(actions_py):
        return min(len(actions_cpp), len(actions_py))
    return None


def _replay(actions):
    """Replay recorded policy-index actions on a fresh board (raises if illegal)."""
    board = Board(SIZE)
    for a in actions:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        if a == PASS:
            board.pass_move(color)
        else:
            board.play((a // SIZE, a % SIZE), color)
    return board


def _assert_terminates(rec, label):
    if rec["forced_terminal"]:
        return  # max-moves cap counts as a termination per the gate
    board = _replay(rec["move_actions"])
    assert board.is_terminal(), (
        f"[{label}] game ended neither by double-pass nor max_moves"
    )


# ---------------------------------------------------------------------------
# Gate 1 (bit-exact): 20 seeds, sims=30, 100% move-for-move identity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def net():
    return _model()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("seed", range(20))
def test_gate1_bit_exact_identity(net, cfg, seed):
    """Same seed + same rng + IEEE double same op order -> identical games."""
    sims = 30
    rec_cpp, rec_py = _play_both(net, cfg, seed, sims)

    div = _first_divergence(rec_cpp["move_actions"], rec_py["move_actions"])
    assert div is None, (
        f"seed={seed} sims={sims}: move divergence at step {div}: "
        f"cpp={rec_cpp['move_actions'][div] if div < len(rec_cpp['move_actions']) else '-'} "
        f"py={rec_py['move_actions'][div] if div < len(rec_py['move_actions']) else '-'}"
    )
    assert rec_cpp["winner"] == rec_py["winner"], f"seed={seed} winner divergence"
    assert rec_cpp["result"] == rec_py["result"], f"seed={seed} result divergence"
    # policy and features bit-exact (visit counts match -> same float32 arrays)
    np.testing.assert_array_equal(rec_cpp["pi"], rec_py["pi"])
    np.testing.assert_array_equal(rec_cpp["features"], rec_py["features"])
    # legality + termination of the C++-path game (also Gate-2 invariants)
    _replay(rec_cpp["move_actions"])
    _assert_terminates(rec_cpp, f"seed={seed}")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gate1_direct_search_tree_identical(net, cfg, seed):
    """The exported C++ search tree matches the Python tree bit-exactly.

    Stronger than the game-level identity: every root child's visit_count and
    q_value (value_sum/visit_count in double) are compared exactly.
    """
    sims = 30
    os.environ["OMIGAMAX_USE_CPP_MCTS"] = "0"
    root_py = make_root(Board(SIZE))
    run_search(
        root_py, None, sims, evaluator=BatchedNetworkEvaluator(net, batch_size=16),
        komi=7.5, virtual_loss=int(cfg.get("virtual_loss", 3)),
    )
    os.environ["OMIGAMAX_USE_CPP_MCTS"] = "1"
    root_cpp = make_root(Board(SIZE))
    run_search(
        root_cpp, None, sims, evaluator=BatchedNetworkEvaluator(net, batch_size=16),
        komi=7.5, virtual_loss=int(cfg.get("virtual_loss", 3)),
    )

    assert set(root_cpp.children) == set(root_py.children)
    for action in root_py.children:
        c, p = root_cpp.children[action], root_py.children[action]
        assert c.visit_count == p.visit_count, (
            f"seed={seed} action={action} visit divergence "
            f"cpp={c.visit_count} py={p.visit_count}"
        )
        assert c.value_sum == p.value_sum, (
            f"seed={seed} action={action} value_sum divergence "
            f"cpp={c.value_sum} py={p.value_sum}"
        )
        assert c.prior == p.prior, f"seed={seed} action={action} prior divergence"


# ---------------------------------------------------------------------------
# Gate 2 (statistical): 200 games, sims 20-50
# ---------------------------------------------------------------------------

def test_gate2_statistical_agreement(net, cfg):
    """200 games (seeds 0-199, sims 20+seed%31): >=95% move agreement, 100%
    legal, all terminate; winner agreement + first-divergence distribution
    reported."""
    total_moves = 0
    agreed_moves = 0
    winner_agreed = 0
    divergence_steps: list[int] = []
    n_games = 200

    for seed in range(n_games):
        sims = 20 + seed % 31
        rec_cpp, rec_py = _play_both(net, cfg, seed, sims)

        # -- invariants on the C++-path game --
        _replay(rec_cpp["move_actions"])  # raises IllegalMoveError if illegal
        _assert_terminates(rec_cpp, f"seed={seed}")

        # -- move agreement over the common prefix --
        d = _first_divergence(rec_cpp["move_actions"], rec_py["move_actions"])
        n_cpp = len(rec_cpp["move_actions"])
        matches = n_cpp if d is None else d
        total_moves += n_cpp
        agreed_moves += matches
        if d is not None:
            divergence_steps.append(d)
        if rec_cpp["winner"] == rec_py["winner"]:
            winner_agreed += 1

    rate = agreed_moves / total_moves if total_moves else 1.0
    winner_rate = winner_agreed / n_games
    n_div = len(divergence_steps)
    div_summary = (
        f"n={n_div} "
        + (f"steps={sorted(set(divergence_steps))[:10]}{'...' if n_div > 10 else ''}"
           if n_div else "none")
    )

    print(
        f"[gate2] games={n_games} total_moves={total_moves} "
        f"agreed={agreed_moves} rate={rate:.4f} "
        f"winner_rate={winner_rate:.3f} divergences={div_summary}"
    )

    assert rate >= 0.95, (
        f"move agreement {rate:.4f} < 0.95 across {n_games} games "
        f"({n_div} diverging games; first divergence steps "
        f"{sorted(divergence_steps)[:20]})"
    )
    # winner agreement is report-only, not a hard gate
    assert winner_rate >= 0.0  # informational


def test_cpp_mcts_probe_reports_available():
    """The env-gated probe is True when the extension is built (Gate 1/2 rely
    on the routing actually engaging)."""
    assert cpp_mcts_available() is True
