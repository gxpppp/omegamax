"""Tests for the MCTS tree-search core (todo 9).

Covers, per the plan's todo-9 acceptance criteria:

  * UCB selection on a hand-built tree -- the literal AlphaGo-Zero formula
    ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`` with ``c_puct = 2.5``:
    higher Q and higher P both increase selection (exact numbers asserted);
  * simulation visit counts accumulate at the root;
  * ``backup`` propagates ``-value`` to ancestors along the search path;
  * ``policy`` = visit counts normalized over legal actions, pass included;
  * illegal moves never become children / never get policy mass (masking);
  * the pass branch exists and is selectable;
  * one search from the empty board: root visits == simulations and child
    priors match the network (plan acceptance);
  * determinism under a fixed network;
  * tree reuse across consecutive actions (root becomes the chosen child);
  * edge case: a root where only pass is legal;
  * edge case: a terminal (two-pass) leaf is never expanded -- the terminal
    value is propagated directly.

All search tests run on small boards (3x3 / 5x5) with tiny networks or mock
evaluators, so the suite stays fast. No GPU is required.
"""

import copy
import math

import numpy as np
import pytest
import torch

from omigamax.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_KOMI,
    MCTS,
    Node,
    NetworkEvaluator,
    descend,
    expand,
    legal_actions,
    make_root,
    most_visited_action,
    run_search,
    select_child,
    terminal_value,
    visit_count_policy,
)
from omigamax.network.features import decode_policy, encode, pass_index, point_to_index
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_node(board=None, prior=0.0):
    """A bare node over a fresh board (hand-built trees for UCB tests)."""
    board = board if board is not None else Board(5)
    return Node(board=board, prior=prior)


def hand_child(prior, visits, value_sum):
    """A child node with manually fixed visit/value statistics."""
    node = build_node()
    node.prior = prior
    node.visit_count = visits
    node.value_sum = value_sum
    return node


def hand_root(children):
    """A root with the given ``{action: node}`` children and 10 visits."""
    root = build_node()
    root.visit_count = 10
    root.legal_moves = tuple(sorted(children))
    root.children = dict(sorted(children.items()))
    return root


def expected_ucb(parent_visits, child_prior, child_visits, q, c_puct=DEFAULT_C_PUCT):
    """The plan's literal UCB formula, mirrored in the test."""
    return q + c_puct * child_prior * math.sqrt(parent_visits) / (1.0 + child_visits)


class UniformEvaluator:
    """Mock leaf evaluator: uniform prior over legal moves, fixed value."""

    def __init__(self, value=0.0):
        self.value = value
        self.calls = 0

    def __call__(self, node):
        self.calls += 1
        size = node.board.size
        prior = np.zeros(size * size + 1, dtype=np.float32)
        for action in node.legal_moves:
            prior[action] = 1.0
        prior /= prior.sum()
        return prior, self.value


# ---------------------------------------------------------------------------
# UCB selection (hand-built tree, exact numbers with c_puct = 2.5)
# ---------------------------------------------------------------------------

def test_ucb_high_q_wins_with_same_prior_and_visits():
    root = hand_root(
        {
            0: hand_child(prior=0.5, visits=5, value_sum=2.5),   # Q = +0.5
            1: hand_child(prior=0.5, visits=5, value_sum=-2.5),  # Q = -0.5
        }
    )
    action, child = select_child(root, DEFAULT_C_PUCT)
    assert action == 0
    assert child.q_value == 0.5
    assert expected_ucb(10, 0.5, 5, 0.5) > expected_ucb(10, 0.5, 5, -0.5)


def test_ucb_high_prior_wins_with_same_q_and_visits():
    root = hand_root(
        {
            0: hand_child(prior=0.2, visits=3, value_sum=0.0),
            1: hand_child(prior=0.8, visits=3, value_sum=0.0),
        }
    )
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 1
    assert expected_ucb(10, 0.8, 3, 0.0) > expected_ucb(10, 0.2, 3, 0.0)


def test_ucb_formula_exact_numbers():
    """Literal formula Q + c_puct*P*sqrt(N_parent)/(1+N_child), c_puct=2.5.

    Root visit count = 10, so sqrt(10) is shared by all three siblings; the
    numbers below are the closed-form evaluations of the plan's formula.
    """
    root = hand_root(
        {
            0: hand_child(prior=0.3, visits=4, value_sum=2.0),   # Q = 0.5
            1: hand_child(prior=0.3, visits=6, value_sum=-1.0),  # Q = -1/6
            2: hand_child(prior=0.1, visits=0, value_sum=0.0),   # Q = 0
        }
    )
    sq = math.sqrt(10.0)
    u0 = 0.5 + 2.5 * 0.3 * sq / 5.0
    u1 = -1.0 / 6.0 + 2.5 * 0.3 * sq / 7.0
    u2 = 0.0 + 2.5 * 0.1 * sq / 1.0
    assert u0 > u2 > u1

    action, _ = select_child(root, 2.5)
    assert action == 0  # the argmax of the literal formula
    scores = {
        a: c.q_value + 2.5 * c.prior * sq / (1.0 + c.visit_count)
        for a, c in root.children.items()
    }
    assert action == max(scores, key=lambda a: (scores[a], -a))


# ---------------------------------------------------------------------------
# search: selection / expansion / backup
# ---------------------------------------------------------------------------

def test_one_search_accumulates_visits_and_children_sum_to_root():
    # Pre-expand the root so every simulation passes through exactly one
    # direct child (on a fresh root the first sim expands the root itself).
    root = make_root(Board(3))
    prior = np.ones(10, dtype=np.float32) / 10.0
    expand(root, prior)
    run_search(root, None, simulations=10, evaluator=UniformEvaluator(value=0.0))
    assert root.visit_count == 10
    assert sum(c.visit_count for c in root.children.values()) == 10
    # every child is a legal action of the empty 3x3 board (9 points + pass)
    assert set(root.children) == set(legal_actions(root.board))


def test_backup_propagates_negated_values_to_ancestors():
    root = make_root(Board(3))
    prior = np.ones(10, dtype=np.float32) / 10.0
    expand(root, prior)  # pre-expand so a child is the next leaf
    run_search(root, None, simulations=1, evaluator=UniformEvaluator(value=0.8))
    assert root.visit_count == 1
    # The lowest-index child (action 0 = (0,0)) is the leaf: with an empty
    # subtree every UCB is equal and ties break to the lowest index.
    child = root.children[point_to_index(0, 0, 3)]
    assert child.visit_count == 1
    assert math.isclose(child.value_sum, 0.8, abs_tol=1e-9)  # leaf's perspective
    assert math.isclose(root.value_sum, -0.8, abs_tol=1e-9)  # negated for parent
    assert math.isclose(root.q_value, -0.8, abs_tol=1e-9)


def test_expansion_uses_network_prior_and_root_visits_equal_sims():
    """Plan acceptance: 1 search from the empty board -> root visits == sims
    and every child's prior matches the network's masked distribution."""
    torch.manual_seed(0)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    root = make_root(Board(5))
    run_search(root, model, simulations=1)

    assert root.visit_count == 1
    assert root.is_expanded
    assert set(root.children) == set(root.legal_moves)

    x = torch.from_numpy(encode(root.history, root.color, board_size=5)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
    expected = decode_policy(logits, root.board)
    assert expected.shape == (26,)
    assert math.isclose(float(expected.sum()), 1.0, abs_tol=1e-5)
    for action in root.legal_moves:
        assert math.isclose(
            root.children[action].prior, float(expected[action]), rel_tol=1e-6
        )


# ---------------------------------------------------------------------------
# policy / selection output
# ---------------------------------------------------------------------------

def test_policy_is_normalized_visit_counts_including_pass():
    root = make_root(Board(3))
    run_search(root, None, simulations=30, evaluator=UniformEvaluator(value=0.0))
    pi = visit_count_policy(root)
    assert pi.shape == (10,)
    assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-6)
    total = sum(c.visit_count for c in root.children.values())
    for action in root.legal_moves:
        expected = root.children[action].visit_count / total
        assert math.isclose(float(pi[action]), expected, rel_tol=1e-6, abs_tol=1e-9)
    for action in range(10):
        if action not in root.legal_moves:
            assert pi[action] == 0.0
    # pass is a legal action and holds a child after expansion
    assert pass_index(3) in root.children


def test_select_action_returns_most_visited_child():
    root = make_root(Board(3))
    run_search(root, None, simulations=20, evaluator=UniformEvaluator(value=0.0))
    action = most_visited_action(root)
    assert action in root.legal_moves
    visits = {a: c.visit_count for a, c in root.children.items()}
    assert visits[action] == max(visits.values())


def test_zero_simulations_produces_all_zero_policy():
    root = make_root(Board(5))
    run_search(root, None, simulations=0, evaluator=UniformEvaluator(value=0.0))
    assert root.visit_count == 0
    pi = visit_count_policy(root)
    assert pi.shape == (26,)
    assert (pi == 0.0).all()


# ---------------------------------------------------------------------------
# legality masking
# ---------------------------------------------------------------------------

def test_illegal_actions_never_become_children():
    board = Board(3)
    board.play((0, 0), BLACK)  # black played -> white to move, (0,0) occupied
    root = make_root(board)
    assert point_to_index(0, 0, 3) not in root.legal_moves

    class EvilEvaluator:
        """Gives a huge prior to the occupied action 0 -- must be masked."""

        def __call__(self, node):
            size = node.board.size
            prior = np.full(size * size + 1, 0.01, dtype=np.float32)
            prior[0] = 10.0  # occupied by black -> illegal for white
            prior[pass_index(size)] = 10.0
            prior /= prior.sum()
            return prior, 0.0

    run_search(root, None, simulations=5, evaluator=EvilEvaluator())
    assert 0 not in root.children
    for action in root.children:
        assert action in root.legal_moves
    pi = visit_count_policy(root)
    assert pi[0] == 0.0


# ---------------------------------------------------------------------------
# pass handling
# ---------------------------------------------------------------------------

def test_pass_child_exists_and_is_selectable():
    root = make_root(Board(3))
    assert pass_index(3) in root.legal_moves
    # Pre-expand with a prior that favours pass at the root itself, so UCB
    # selection sends every simulation down the pass branch.
    prior = np.zeros(10, dtype=np.float32)
    prior[pass_index(3)] = 1.0
    expand(root, prior)

    class PassEvaluator:
        """Also favours pass for any deeper leaf."""

        def __call__(self, node):
            size = node.board.size
            prior = np.zeros(size * size + 1, dtype=np.float32)
            prior[pass_index(size)] = 1.0
            return prior, 0.0

    run_search(root, None, simulations=8, evaluator=PassEvaluator())
    assert pass_index(3) in root.children
    # The first simulation from a fresh root (N_root=0) is a blind pick of the
    # lowest-index child because sqrt(N_root) zeroes every exploration term --
    # standard AGZ behaviour -- so the pass branch receives the remaining 7.
    assert root.children[pass_index(3)].visit_count == 7
    assert root.visit_count == 8
    assert most_visited_action(root) == pass_index(3)
    pi = visit_count_policy(root)
    assert math.isclose(float(pi[pass_index(3)]), 7.0 / 8.0, rel_tol=1e-6)


def test_only_pass_legal_root():
    """Edge case: a root whose only legal action is pass."""
    board = Board(3)
    root = Node(board=board, legal_moves=(pass_index(3),))
    prior = np.zeros(10, dtype=np.float32)
    prior[pass_index(3)] = 1.0
    expand(root, prior)  # pre-expand -> every simulation visits the pass child
    run_search(root, None, simulations=6, evaluator=UniformEvaluator(value=0.0))
    assert root.visit_count == 6
    assert set(root.children) == {pass_index(3)}
    assert root.children[pass_index(3)].visit_count == 6
    assert most_visited_action(root) == pass_index(3)
    pi = visit_count_policy(root)
    assert math.isclose(float(pi[pass_index(3)]), 1.0, abs_tol=1e-9)
    assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# terminal handling
# ---------------------------------------------------------------------------

def test_terminal_leaf_is_never_expanded_value_propagated_directly():
    board = Board(3)
    board.pass_move(BLACK)
    board.pass_move(WHITE)
    assert board.is_terminal()

    root = Node(
        board=copy.deepcopy(board),
        legal_moves=legal_actions(board),
    )

    class BoomEvaluator:
        def __call__(self, node):  # pragma: no cover - must never be called
            raise AssertionError("evaluator must not run on a terminal leaf")

    run_search(root, None, simulations=5, evaluator=BoomEvaluator())
    assert root.visit_count == 5
    assert not root.children  # no expansion on a terminal position
    # empty board, komi 7.5 -> W wins; black is to move (2 moves) -> value -1
    assert math.isclose(terminal_value(board, DEFAULT_KOMI), -1.0, abs_tol=1e-9)
    assert math.isclose(root.value_sum, -5.0, abs_tol=1e-9)
    assert math.isclose(root.q_value, -1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# determinism / tree reuse
# ---------------------------------------------------------------------------

def test_deterministic_with_fixed_network():
    torch.manual_seed(123)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    root_a = make_root(Board(5))
    root_b = make_root(Board(5))
    run_search(root_a, model, simulations=15)
    run_search(root_b, model, simulations=15)

    np.testing.assert_array_equal(visit_count_policy(root_a), visit_count_policy(root_b))
    for action in root_a.children:
        assert root_a.children[action].visit_count == root_b.children[action].visit_count


def test_tree_reuse_root_becomes_child_across_actions():
    torch.manual_seed(7)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    mcts = MCTS(network=model, c_puct=DEFAULT_C_PUCT, komi=DEFAULT_KOMI)
    root = mcts.new_root(Board(5))
    mcts.run(simulations=10)
    action = mcts.select_action()
    assert action in root.legal_moves

    mcts.apply_action(action)
    assert mcts.root is root.children[action]

    before = mcts.root.visit_count  # the child already carried visits from search 1
    mcts.run(simulations=10)
    assert mcts.root.visit_count == before + 10
    pi = mcts.policy()
    assert pi.shape == (26,)
    assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-6)


def test_descend_helper_returns_chosen_child():
    root = make_root(Board(3))
    run_search(root, None, simulations=4, evaluator=UniformEvaluator(value=0.0))
    action = most_visited_action(root)
    assert descend(root, action) is root.children[action]
