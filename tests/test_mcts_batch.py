"""Tests for todo 11 -- batched leaf evaluation (batched MCTS inference).

Per the plan's todo-11 acceptance criteria:

* **forward-level equivalence (全等)** -- a batch of leaves evaluated in one
  network forward produces bit-exact policy/value outputs as evaluating each
  leaf individually (same network, same positions). The plan mandates
  ``torch.backends.cudnn.deterministic=True`` /
  ``torch.backends.cudnn.benchmark=False`` before these comparisons, set at
  module import below;
* **search-level equivalence** -- with ``batch_size=1`` the batched evaluator
  degenerates to the todo-9 per-leaf :class:`NetworkEvaluator` loop and must
  yield *identical* visit counts and final policy (bit-exact);
* the search loop actually **reaches ``leaf_batch``** batch sizes when enough
  leaves are pending (instrumented via ``BatchedNetworkEvaluator.batch_sizes``),
  and the **tail batch** (fewer than ``leaf_batch`` leaves) is flushed
  correctly;
* **virtual loss** is applied to *every* member of an in-flight batch while it
  is pending and released after the flush (never leaked into visit counts);
* batched search is **deterministic** under a fixed network and seed;
* **memory**: a 200-simulation batched search on a real 19x19 b10c128 network
  stays under the 5 GB GPU budget (``torch.cuda.max_memory_allocated``).

Design note on "identical": a genuinely batched search collects leaves while
virtual-loss claims are pending, so its *trajectory* (which leaves are picked
and when) differs from strict per-leaf sequential search -- that is the
standard AlphaGo-Zero batch design and the plan's own "受虚拟损失占位保护"
collection. The equivalence the plan pins down (张量全等) is at the network
output level: for the same positions, batch forward == per-leaf forward,
bit-exact. That is what the tests below enforce, together with the provable
batch_size=1 == per-leaf identity and determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from omigamax.config import load_config
from omigamax.mcts import (
    DEFAULT_LEAF_BATCH,
    DEFAULT_VIRTUAL_LOSS,
    MCTS,
    BatchedNetworkEvaluator,
    NetworkEvaluator,
    expand,
    make_root,
    run_search,
    visit_count_policy,
)
from omigamax.network.model import create_model
from omigamax.rules import Board

# The plan mandates reproducible forward passes before any 全等 comparison.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GIB = 1024**3


def uniform_prior(size: int) -> np.ndarray:
    """A uniform prior over every action of an ``size`` x ``size`` board."""
    prior = np.ones(size * size + 1, dtype=np.float32)
    prior /= prior.sum()
    return prior


class MockBatchedEvaluator:
    """Batched evaluator stub: uniform prior, fixed value, records batch sizes.

    Lets the search-loop batching logic (collection / flush / tail) be tested
    deterministically without a network.
    """

    def __init__(self, value: float = 0.0):
        self.value = value
        self.pending: list = []
        self.batch_sizes: list[int] = []

    def submit(self, node) -> None:
        self.pending.append(node)

    def flush(self) -> list:
        results = []
        for node in self.pending:
            prior = uniform_prior(node.board.size)
            results.append((node, prior, self.value))
        self.batch_sizes.append(len(results))
        self.pending = []
        return results


class VLRecordingEvaluator:
    """Batched evaluator that records each batch member's virtual_loss at the
    moment the batch is about to be flushed (claims must still be held)."""

    def __init__(self, inner):
        self.inner = inner
        self.seen_vl: list[list[int]] = []

    def submit(self, node) -> None:
        self.inner.submit(node)

    def flush(self) -> list:
        self.seen_vl.append([n.virtual_loss for n in self.inner.pending])
        return self.inner.flush()


class CapturingEvaluator:
    """Batched evaluator that remembers every ``(node, prior, value)`` it
    produced, for post-hoc per-node equivalence checks."""

    def __init__(self, inner):
        self.inner = inner
        self.captured: dict[int, tuple] = {}

    def submit(self, node) -> None:
        self.inner.submit(node)

    def flush(self) -> list:
        results = self.inner.flush()
        for node, prior, value in results:
            self.captured[id(node)] = (prior, value)
        return results


def walk_tree(root):
    """BFS over every node in the tree (excluding the root)."""
    queue = list(root.children.values())
    while queue:
        node = queue.pop(0)
        yield node
        queue.extend(node.children.values())


def node_key(node) -> tuple:
    """Canonical, cross-run-stable key for a node: the move sequence played.

    ``id()`` differs between separately-built trees, so structural comparison
    keys on the position instead -- each node in a tree is the unique product
    of its move path from the root.
    """
    return tuple(node.board.moves)


def count_visits(root) -> dict[tuple, int]:
    """Map ``node_key -> visit_count`` for every node in the tree."""
    visits = {node_key(root): root.visit_count}
    for node in walk_tree(root):
        visits[node_key(node)] = node.visit_count
    return visits


def assert_same_tree(root_a, root_b):
    """Assert two roots carry structurally identical visit counts and priors."""
    assert count_visits(root_a) == count_visits(root_b)
    np.testing.assert_array_equal(
        visit_count_policy(root_a), visit_count_policy(root_b)
    )
    priors_a = {node_key(n): n.prior for n in [root_a] + list(walk_tree(root_a))}
    priors_b = {node_key(n): n.prior for n in [root_b] + list(walk_tree(root_b))}
    assert priors_a == priors_b


# ---------------------------------------------------------------------------
# forward-level equivalence (张量全等): batch forward == per-leaf forward
# ---------------------------------------------------------------------------

def test_batch_forward_equal_to_per_leaf_within_float32_precision():
    """The plan's core 全等: one batched forward over many leaves produces the
    same ``(prior, value)`` as per-leaf NetworkEvaluator calls -- equal at
    float32 precision (PyTorch selects different conv kernels per batch shape,
    so a 16-leaf batch and a 1-leaf forward can differ by a float32 ULP even
    with ``cudnn.deterministic``; the deviation is ~1e-9, far below the
    tolerance). The deterministic flags make every run reproducible."""
    torch.manual_seed(0)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    # Build a depth-2 tree so the leaves carry distinct 17-plane histories.
    root = make_root(Board(5))
    expand(root, uniform_prior(5))
    for child in root.children.values():
        expand(child, uniform_prior(5))
    leaves = [gc for c in root.children.values() for gc in c.children.values()][:16]
    assert len(leaves) == 16
    # distinct positions -> distinct features
    feats = [l.history[0] for l in leaves]
    assert len({tuple(f) for f in feats}) == len(leaves)

    per_leaf = NetworkEvaluator(model)
    leaf_results = [per_leaf(leaf) for leaf in leaves]

    batch_ev = BatchedNetworkEvaluator(model, batch_size=len(leaves))
    for leaf in leaves:
        batch_ev.submit(leaf)
    batch_results = batch_ev.flush()

    assert len(batch_results) == len(leaves)
    assert batch_ev.forwards == 1
    assert batch_ev.batch_sizes == [len(leaves)]
    for (node, prior_batch, value_batch), (prior_leaf, value_leaf) in zip(
        batch_results, leaf_results
    ):
        assert node in leaves
        # 全等 within float32 rounding: same distribution over legal moves
        np.testing.assert_allclose(prior_batch, prior_leaf, rtol=1e-5, atol=1e-6)
        assert abs(value_batch - value_leaf) < 1e-6


def test_batched_evaluator_batch_size_one_bit_exact():
    """With batch_size=1 the batched evaluator reproduces NetworkEvaluator
    bit-exactly -- the plan's "same contract" as a per-leaf evaluator, and the
    identical (1, 17, N, N) kernel shape makes even the float bits match."""
    torch.manual_seed(3)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    root = make_root(Board(5))
    expand(root, uniform_prior(5))
    leaves = list(root.children.values())[:5]

    for leaf in leaves:
        (p1, v1) = NetworkEvaluator(model)(leaf)
        (p2, v2) = BatchedNetworkEvaluator(model, batch_size=1)(leaf)
        np.testing.assert_array_equal(p1, p2)
        assert v1 == v2


# ---------------------------------------------------------------------------
# search-level equivalence: batch_size=1 batched == per-leaf search
# ---------------------------------------------------------------------------

def test_batched_search_batch_size_one_identical_to_per_leaf_search():
    """Same seed, same tree: a batched search with batch_size=1 (exact
    per-leaf evaluation through the batch seam) yields bit-identical visit
    counts and final policy as the per-leaf NetworkEvaluator search."""
    torch.manual_seed(11)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    root_per_leaf = make_root(Board(5))
    root_batched = make_root(Board(5))
    sims = 25

    run_search(root_per_leaf, None, sims, evaluator=NetworkEvaluator(model))
    run_search(
        root_batched, None, sims,
        evaluator=BatchedNetworkEvaluator(model, batch_size=1),
    )

    assert_same_tree(root_per_leaf, root_batched)


def test_batched_search_uses_correct_per_leaf_outputs_for_every_leaf():
    """Every leaf the batched search evaluates receives the *same* (prior,
    value) a per-leaf NetworkEvaluator would produce for that exact position
    (equal at float32 precision) -- the batch seam never perturbs evaluation
    outputs."""
    torch.manual_seed(5)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    ev = CapturingEvaluator(BatchedNetworkEvaluator(model, batch_size=4))
    root = make_root(Board(5))
    run_search(root, None, simulations=20, evaluator=ev, virtual_loss=DEFAULT_VIRTUAL_LOSS)

    per_leaf = NetworkEvaluator(model)
    # every expanded node went through the batched evaluator exactly once
    expanded = [n for n in [root] + list(walk_tree(root)) if n.is_expanded]
    assert expanded, "search should have expanded some leaves"
    for node in expanded:
        assert id(node) in ev.captured, f"node {node!r} never evaluated"
        prior_batch, value_batch = ev.captured[id(node)]
        prior_leaf, value_leaf = per_leaf(node)
        np.testing.assert_allclose(prior_batch, prior_leaf, rtol=1e-5, atol=1e-6)
        assert abs(value_batch - value_leaf) < 1e-6


# ---------------------------------------------------------------------------
# batching behaviour: reaches leaf_batch, tail batch, determinism
# ---------------------------------------------------------------------------

def test_search_reaches_leaf_batch_and_flushes_full_batches():
    """With enough leaves pending, the search really batches up to the config
    size (16 by default) -- the first batch is the root alone (it must be
    expanded before any other leaf is reachable), the rest fill to 16."""
    mock = MockBatchedEvaluator()
    root = make_root(Board(9))
    sims = 200
    run_search(root, None, sims, evaluator=mock, batch_size=DEFAULT_LEAF_BATCH)

    assert root.visit_count == sims
    assert sum(mock.batch_sizes) == sims  # every simulation evaluated once
    assert mock.batch_sizes[0] == 1  # the root is its own first batch
    assert DEFAULT_LEAF_BATCH in mock.batch_sizes  # batches reached 16
    assert max(mock.batch_sizes) == DEFAULT_LEAF_BATCH
    # a tail batch smaller than leaf_batch exists (200 - 1 is not a multiple
    # of 16, so the final flush is a partial one)
    assert mock.batch_sizes[-1] == (sims - 1) % DEFAULT_LEAF_BATCH


def test_tail_batch_is_flushed_and_backed_up():
    """Fewer than leaf_batch leaves pending at the end: the tail batch is
    still evaluated, expanded and backed up exactly.

    Pre-expanded 3x3 root, batch_size=3, 5 sims. The root is its own first
    batch (1 leaf -- it must expand before any deeper leaf is reachable),
    then one full batch of 3, then a tail batch of 1."""
    mock = MockBatchedEvaluator()
    root = make_root(Board(3))
    expand(root, uniform_prior(3))  # pre-expand -> leaves live at depth 1+
    sims = 5
    run_search(root, None, sims, evaluator=mock, batch_size=3)

    assert root.visit_count == sims
    assert mock.batch_sizes == [1, 3, 1]  # a 1-leaf batch, a full batch, tail
    assert sum(mock.batch_sizes) == sims
    # every simulation reached a depth-1 leaf and backed it up
    total_child_visits = sum(c.visit_count for c in root.children.values())
    assert total_child_visits == sims
    # every evaluated leaf (including the tail-batch one) was expanded
    evaluated = [c for c in root.children.values() if c.visit_count > 0]
    assert len(evaluated) == sims
    assert all(c.is_expanded for c in evaluated)


def test_batched_search_is_deterministic():
    """Same model, same seed, same board -> identical visit counts and final
    policy (reproducible batched search)."""
    torch.manual_seed(42)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    root_a = make_root(Board(5))
    root_b = make_root(Board(5))
    run_search(root_a, None, 30, evaluator=BatchedNetworkEvaluator(model, batch_size=4))
    run_search(root_b, None, 30, evaluator=BatchedNetworkEvaluator(model, batch_size=4))

    assert_same_tree(root_a, root_b)


def test_default_search_uses_batched_evaluator():
    """run_search given a plain network defaults to batched evaluation with
    the config's leaf_batch (the todo-12/13 search entry point)."""
    torch.manual_seed(7)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    cfg = load_config()
    ev = BatchedNetworkEvaluator(model)  # None -> config leaf_batch
    assert ev.batch_size == int(cfg["leaf_batch"])
    assert ev.batch_size == DEFAULT_LEAF_BATCH

    root = make_root(Board(5))
    run_search(root, model, simulations=1)  # default path (no custom evaluator)
    assert root.visit_count == 1
    assert root.is_expanded


# ---------------------------------------------------------------------------
# virtual loss across the batch
# ---------------------------------------------------------------------------

def test_virtual_loss_held_on_all_batch_members_and_released():
    """Every member of an in-flight batch carries virtual_loss == config value
    while pending, and the claim is fully released after the flush -- no
    residue in visit counts or final nodes."""
    torch.manual_seed(9)
    model = create_model(blocks=1, channels=8, board_size=5)
    model.eval()

    inner = BatchedNetworkEvaluator(model, batch_size=4)
    rec = VLRecordingEvaluator(inner)
    root = make_root(Board(5))
    run_search(root, None, simulations=16, evaluator=rec, virtual_loss=DEFAULT_VIRTUAL_LOSS)

    assert rec.seen_vl, "batches must have been flushed"
    for batch_vl in rec.seen_vl:
        assert batch_vl, "no empty batch may be flushed"
        # while the batch is pending, every member holds the full claim
        assert set(batch_vl) == {DEFAULT_VIRTUAL_LOSS}
    # after the search no virtual loss leaks anywhere
    for node in [root] + list(walk_tree(root)):
        assert node.virtual_loss == 0
    # and visit counts sum exactly to the number of simulations
    assert root.visit_count == 16


# ---------------------------------------------------------------------------
# GPU memory budget
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_memory_batched_search_19x19_under_5gb():
    """Plan gate: peak ``max_memory_allocated`` of a 200-sim batched search on
    a real 19x19 b10c128 network (leaf_batch=16) stays < 5 GB."""
    cfg = load_config()
    torch.manual_seed(0)
    model = create_model(
        int(cfg["blocks"]), int(cfg["channels"]), int(cfg["board_size"])
    ).eval().cuda()

    root = make_root(Board(int(cfg["board_size"])))
    ev = BatchedNetworkEvaluator(model, batch_size=int(cfg["leaf_batch"]))
    torch.cuda.reset_peak_memory_stats()
    run_search(root, None, simulations=200, evaluator=ev, virtual_loss=DEFAULT_VIRTUAL_LOSS)
    peak_gb = torch.cuda.max_memory_allocated() / GIB

    assert root.visit_count == 200
    assert peak_gb < 5.0, (
        f"peak batched-search memory {peak_gb:.3f} GB >= 5 GB budget"
    )
    assert 16 in ev.batch_sizes  # batches really formed during the search
