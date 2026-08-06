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
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPS,
    DEFAULT_KOMI,
    DEFAULT_VIRTUAL_LOSS,
    MCTS,
    Node,
    NetworkEvaluator,
    TAU_ARGMAX_THRESHOLD,
    apply_dirichlet_noise,
    clear_root_noise,
    descend,
    expand,
    legal_actions,
    make_root,
    most_visited_action,
    run_search,
    sample_action,
    select_child,
    temperature_policy,
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


# ---------------------------------------------------------------------------
# AGZ details (todo 10): Dirichlet root noise
# ---------------------------------------------------------------------------

def _expanded_root(size=3, prior_value=1.0):
    """A pre-expanded root whose children carry a uniform prior and the given
    visit/value statistics (fresh, then overwritten by the caller as needed)."""
    root = make_root(Board(size))
    prior = np.ones(size * size + 1, dtype=np.float32) / (size * size + 1)
    expand(root, prior)
    return root


def test_dirichlet_noise_blend_formula_exact():
    """P'(a) = (1-eps)*P(a) + eps*eta(a), eta ~ Dir(alpha) -- reproduced by
    replaying the seeded draw with numpy's own dirichlet."""
    seed = 12345
    rng = np.random.default_rng(seed)
    root = _expanded_root(size=3)
    actions = list(root.children)
    noisy = apply_dirichlet_noise(root, DEFAULT_DIRICHLET_ALPHA, DEFAULT_DIRICHLET_EPS, rng=rng)

    replay = np.random.default_rng(seed)
    eta = replay.dirichlet(np.full(len(actions), DEFAULT_DIRICHLET_ALPHA))
    for i, action in enumerate(actions):
        expected = (1.0 - DEFAULT_DIRICHLET_EPS) * root.children[action].prior + DEFAULT_DIRICHLET_EPS * float(eta[i])
        assert math.isclose(noisy[action], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_dirichlet_noise_changes_distribution_and_sums_to_one():
    root = _expanded_root(size=3)
    original = {a: c.prior for a, c in root.children.items()}
    noisy = apply_dirichlet_noise(root, 0.03, 0.25, rng=np.random.default_rng(7))
    # distribution actually changed (some child got a boost/shift)
    assert any(not math.isclose(noisy[a], original[a], abs_tol=1e-9) for a in noisy)
    total = sum(noisy.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6)  # (1-eps)*sum(P) + eps*sum(eta) = 1


def test_dirichlet_noise_priors_stay_in_unit_interval():
    root = _expanded_root(size=3)
    noisy = apply_dirichlet_noise(root, 0.03, 0.25, rng=np.random.default_rng(11))
    assert len(noisy) == len(root.children)
    for value in noisy.values():
        assert 0.0 <= value <= 1.0


def test_dirichlet_noise_deterministic_with_seeded_rng():
    root_a = _expanded_root(size=3)
    root_b = _expanded_root(size=3)
    noisy_a = apply_dirichlet_noise(root_a, rng=np.random.default_rng(42))
    noisy_b = apply_dirichlet_noise(root_b, rng=np.random.default_rng(42))
    assert noisy_a == noisy_b


def test_dirichlet_noise_only_affects_root_stored_priors_untouched():
    """The blend lives in root.noisy_prior; stored child.prior (and deeper
    nodes) are never mutated -- repeated application cannot compound noise."""
    root = _expanded_root(size=3)
    # give one child an expanded subtree with a known prior
    child = next(iter(root.children.values()))
    child_prior = np.ones(10, dtype=np.float32) / 10.0
    expand(child, child_prior)
    stored_child_priors = {a: c.prior for a, c in root.children.items()}
    grandchild_priors = {a: c.prior for a, c in child.children.items()}

    apply_dirichlet_noise(root, rng=np.random.default_rng(3))
    assert root.noisy_prior is not None
    # stored priors unchanged everywhere
    for action, prior in stored_child_priors.items():
        assert root.children[action].prior == prior
    for action, prior in grandchild_priors.items():
        assert child.children[action].prior == prior
    # non-root nodes never carry a noise override
    for c in root.children.values():
        assert c.noisy_prior is None


def test_select_child_uses_noisy_prior_override():
    """The override flips selection; reverting the override restores it."""
    root = hand_root(
        {
            0: hand_child(prior=0.1, visits=5, value_sum=0.0),
            1: hand_child(prior=0.9, visits=5, value_sum=0.0),
        }
    )
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 1  # higher prior wins without noise
    root.noisy_prior = {0: 0.99, 1: 0.01}
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 0  # noise override flips the selection
    assert root.children[0].prior == 0.1  # stored prior untouched
    assert root.children[1].prior == 0.9
    clear_root_noise(root)
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 1  # original selection restored


def test_dirichlet_noise_unexpanded_root_is_noop():
    root = make_root(Board(3))  # no children yet
    noisy = apply_dirichlet_noise(root, rng=np.random.default_rng(1))
    assert noisy == {}
    assert root.noisy_prior is None


def test_run_search_noise_cleared_when_not_requested():
    """A noisy run sets the override; the next no-noise run clears stale
    noise so stored priors drive selection again."""
    root = _expanded_root(size=3)
    run_search(root, None, simulations=5, evaluator=UniformEvaluator(value=0.0),
               dirichlet_alpha=0.03, dirichlet_eps=0.25, dirichlet_rng=np.random.default_rng(9))
    assert root.noisy_prior is not None
    run_search(root, None, simulations=5, evaluator=UniformEvaluator(value=0.0))
    assert root.noisy_prior is None


# ---------------------------------------------------------------------------
# AGZ details (todo 10): temperature selection
# ---------------------------------------------------------------------------

def test_temperature_tau_one_matches_visit_policy():
    root = make_root(Board(3))
    run_search(root, None, simulations=40, evaluator=UniformEvaluator(value=0.0))
    pi_tau1 = temperature_policy(root, 1.0)
    assert np.allclose(pi_tau1, visit_count_policy(root), atol=1e-6)


def test_temperature_policy_shape_and_normalization():
    root = make_root(Board(3))
    run_search(root, None, simulations=20, evaluator=UniformEvaluator(value=0.0))
    for tau in (1.0, 0.5, 0.1, 0.0, TAU_ARGMAX_THRESHOLD / 2):
        pi = temperature_policy(root, tau)
        assert pi.shape == (10,)
        assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-6)
        for action in range(10):
            if action not in root.legal_moves:
                assert pi[action] == 0.0


def test_temperature_tau_zero_concentrates_on_argmax():
    root = make_root(Board(3))
    run_search(root, None, simulations=30, evaluator=UniformEvaluator(value=0.0))
    pi = temperature_policy(root, 0.0)
    max_visits = max(c.visit_count for c in root.children.values())
    winners = [a for a, c in root.children.items() if c.visit_count == max_visits]
    assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-6)
    assert math.isclose(float(pi[winners].sum()), 1.0, abs_tol=1e-6)
    assert all(pi[a] == 0.0 for a in root.children if a not in winners)


def test_sample_action_tau_zero_always_picks_most_visited():
    root = make_root(Board(3))
    run_search(root, None, simulations=40, evaluator=UniformEvaluator(value=0.0))
    max_visits = max(c.visit_count for c in root.children.values())
    rng = np.random.default_rng(1)
    for _ in range(50):
        action = sample_action(root, 0.0, rng=rng)
        assert root.children[action].visit_count == max_visits


def test_sample_action_tau_one_covers_multiple_children():
    """With spread visit counts, tau=1 sampling (one shared rng) reaches
    several children."""
    root = hand_root(
        {
            0: hand_child(prior=0.25, visits=10, value_sum=0.0),
            1: hand_child(prior=0.25, visits=5, value_sum=0.0),
            2: hand_child(prior=0.25, visits=2, value_sum=0.0),
            3: hand_child(prior=0.25, visits=1, value_sum=0.0),
        }
    )
    rng = np.random.default_rng(5)
    seen = {sample_action(root, 1.0, rng=rng) for _ in range(200)}
    assert len(seen) >= 3  # all but the rarest almost surely


def test_temperature_tau_zero_tie_break_reproducible_with_seed():
    """tau->0 on tied argmax children samples uniformly among them, and a
    seeded rng reproduces the exact same sequence."""
    root = hand_root(
        {
            0: hand_child(prior=0.5, visits=7, value_sum=0.0),
            1: hand_child(prior=0.5, visits=7, value_sum=0.0),
        }
    )
    rng_a = np.random.default_rng(77)
    rng_b = np.random.default_rng(77)
    seq_a = [sample_action(root, 0.0, rng=rng_a) for _ in range(40)]
    seq_b = [sample_action(root, 0.0, rng=rng_b) for _ in range(40)]
    assert seq_a == seq_b
    assert set(seq_a) == {0, 1}  # both tied argmax children get picked


def test_temperature_guard_no_division_by_zero_on_unsearched_root():
    root = _expanded_root(size=3)  # every child exists with 0 visits
    for tau in (0.0, 1e-9, TAU_ARGMAX_THRESHOLD / 2):
        pi = temperature_policy(root, tau)  # must not raise / overflow
        assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-6)


def test_temperature_policy_terminal_root_all_zero():
    root = Node(board=Board(3))  # no children at all
    pi = temperature_policy(root, 1.0)
    assert (pi == 0.0).all()
    assert sample_action(root, 1.0) == pass_index(3)  # fallback to argmax helper


# ---------------------------------------------------------------------------
# AGZ details (todo 10): virtual loss
# ---------------------------------------------------------------------------

def test_virtual_loss_depresses_ucb_and_flips_selection():
    """UCB divides by 1 + N + virtual_loss: a claimed child scores
    Q + c_puct*P*sqrt(Np)/(1+N+vl), and once the claim is reverted the
    original UCB / selection is restored."""
    root = hand_root(
        {
            0: hand_child(prior=0.6, visits=0, value_sum=0.0),
            1: hand_child(prior=0.4, visits=0, value_sum=0.0),
        }
    )
    # no virtual loss: higher prior (action 0) wins
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 0

    # claim virtual loss 3 on the favourite -> depressed below its sibling
    root.children[0].virtual_loss = DEFAULT_VIRTUAL_LOSS
    sq = math.sqrt(root.visit_count)  # hand_root sets root visits = 10
    ucb_claimed = 0.0 + DEFAULT_C_PUCT * 0.6 * sq / (1.0 + 0 + DEFAULT_VIRTUAL_LOSS)
    ucb_sibling = 0.0 + DEFAULT_C_PUCT * 0.4 * sq / (1.0 + 0 + 0)
    assert ucb_claimed < ucb_sibling
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 1

    # revert -> original UCB restored, favourite wins again
    root.children[0].virtual_loss = 0
    action, _ = select_child(root, DEFAULT_C_PUCT)
    assert action == 0


def test_run_search_claims_virtual_loss_on_leaf_during_eval():
    """During leaf evaluation node.virtual_loss == config value (3); after the
    search every node's virtual_loss is back to 0."""

    class RecordingEvaluator:
        def __init__(self):
            self.seen = []

        def __call__(self, node):
            self.seen.append(node.virtual_loss)
            size = node.board.size
            prior = np.ones(size * size + 1, dtype=np.float32)
            prior /= prior.sum()
            return prior, 0.0

    rec = RecordingEvaluator()
    root = make_root(Board(3))
    run_search(root, None, simulations=6, evaluator=rec, virtual_loss=DEFAULT_VIRTUAL_LOSS)
    assert rec.seen, "evaluator must have run"
    assert set(rec.seen) == {DEFAULT_VIRTUAL_LOSS}
    nodes = [root] + list(_walk_tree(root))
    assert all(n.virtual_loss == 0 for n in nodes)


def _walk_tree(root):
    """BFS over every node in the tree (excluding the root)."""
    queue = list(root.children.values())
    while queue:
        node = queue.pop(0)
        yield node
        queue.extend(node.children.values())


def test_virtual_loss_does_not_leak_into_visits_or_policy():
    """Single-threaded search: the claim/release is transparent, so visit
    counts and the final policy are identical to a run without virtual loss,
    and no virtual_loss residue remains."""
    root_plain = make_root(Board(3))
    run_search(root_plain, None, simulations=30, evaluator=UniformEvaluator(value=0.0),
               virtual_loss=0)
    root_vl = make_root(Board(3))
    run_search(root_vl, None, simulations=30, evaluator=UniformEvaluator(value=0.0),
               virtual_loss=DEFAULT_VIRTUAL_LOSS)

    for action in root_plain.legal_moves:
        assert root_plain.children[action].visit_count == root_vl.children[action].visit_count
    np.testing.assert_array_equal(
        visit_count_policy(root_plain), visit_count_policy(root_vl)
    )
    for node in [root_vl] + list(_walk_tree(root_vl)):
        assert node.virtual_loss == 0


def test_virtual_loss_zero_disabled():
    """virtual_loss=0 never claims anything on the leaf."""

    class RecordingEvaluator:
        def __init__(self):
            self.seen = []

        def __call__(self, node):
            self.seen.append(node.virtual_loss)
            size = node.board.size
            prior = np.ones(size * size + 1, dtype=np.float32)
            prior /= prior.sum()
            return prior, 0.0

    rec = RecordingEvaluator()
    root = make_root(Board(3))
    run_search(root, None, simulations=4, evaluator=rec, virtual_loss=0)
    assert rec.seen and set(rec.seen) == {0}
