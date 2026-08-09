"""MCTS tree-search core (todo 9) + AGZ search details (todo 10).

AlphaGo Zero style Monte Carlo tree search over the :class:`~omigamax.rules.board.Board`
API. Each simulation has three phases (AGZ Nature 2017 Methods):

1. **selection** -- from the root, repeatedly pick the child maximising
   ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`` (the plan's exact
   formula, ``c_puct = 2.5`` by default -- configurable via ``config/default.yaml``),
   where ``Q = value_sum / visit_count`` and ``P`` is the edge's prior
   probability from the network, until an unexpanded leaf is reached;
2. **expansion** -- the leaf's position is evaluated once by the network
   (per-leaf synchronous here; a pluggable ``evaluator`` is the seam that
   todo 11 turns into batched inference), the raw policy is masked to legal
   moves via :func:`omigamax.network.features.decode_policy`, and a child
   node is created for every legal action -- including pass (the last index
   ``board_size**2``), which is a normal legal branch;
3. **backup** -- the leaf value (negated at every level, because the
   perspective flips with the side to move) is accumulated into ``value_sum``
   and ``visit_count`` along the path back to the root.

Terminal positions (two consecutive passes, :meth:`Board.is_terminal`) are
handled without expansion: the game outcome is computed by the rules engine
and propagated directly -- the leaf is never evaluated by the network.

Pass handling: the pass move is index ``board_size ** 2`` (see
:mod:`omigamax.network.features`); it is always legal, so the root's legal
actions always include it and the expanded children always contain a pass
node.

Todo 10 adds the AGZ search details on top of the todo-9 core:

* **Dirichlet root noise** (:func:`apply_dirichlet_noise`) -- the root's
  legal-child priors are blended ``P'(a) = (1 - eps) * P(a) + eps * eta(a)``
  with ``eta ~ Dir(alpha)`` (``alpha = dirichlet_alpha = 0.03``,
  ``eps = dirichlet_eps = 0.25``). The blend is stored in a *transient*
  override on the root (``Node.noisy_prior``), never in ``child.prior``, so
  stored network priors stay pristine and re-running a search cannot
  compound stale noise. It is applied only when ``run_search`` is asked for
  it (``dirichlet_alpha is not None``) and only at the root.
* **Temperature selection** (:func:`temperature_policy` /
  :func:`sample_action`) -- AGZ move selection: ``pi(a) propto N(a)^(1/tau)``.
  ``tau = 1.0`` (proportional to visit counts) for the first
  ``temperature_threshold = 30`` moves, then ``tau -> 0`` (argmax). The
  ``tau < 1e-6`` branch short-circuits to the argmax (ties share the mass
  uniformly -- ``max(0, ..)`` protected, never a division by zero).
* **Virtual loss** -- the ``virtual_loss = 3`` counter is incremented on the
  leaf node *while it is being evaluated* and reverted immediately after
  (``try/finally``). During the claim the node's effective visit count in
  UCB is ``N_child + virtual_loss`` (``select_child`` divides by
  ``1 + N_child + virtual_loss``), so a claimed leaf looks worse to any
  concurrent selection -- the seam todo 11's batched leaf evaluation slots
  into. Reverting keeps visit counts and the final policy exact.

Design seams for later todos:

* the ``evaluator`` interface ``callable(node) -> (prior_probs, value)`` --
  todo 11 swaps the per-leaf synchronous
  :class:`NetworkEvaluator` for the batched :class:`BatchedNetworkEvaluator`
  (``submit``/``flush`` protocol; the default when ``run_search``/``MCTS`` is
  given a plain ``network``), behind the same ``(node) -> (prior, value)``
  contract for batch-size-1 use;
* ``Node.virtual_loss`` and ``Node.noisy_prior`` -- consumed only by
  ``select_child``; everything else in this module treats them as opaque.

Todo 11 (batched leaf evaluation) works through the virtual-loss seam: the
search loop collects up to ``leaf_batch`` leaves -- each claimed with
``virtual_loss`` while in flight, which depresses them in UCB and spreads
selection across distinct branches -- then evaluates the whole batch in a
single forward and expands/backs-up every leaf. See :func:`run_search`.

Structure of the module:

* :class:`Node` -- the tree node (``visit_count``, ``prior``, ``value_sum``,
  ``children``, ``legal_moves``, plus the position snapshot the node owns);
* :func:`select_child` -- the UCB selection step (noisy-prior override +
  virtual-loss aware);
* :func:`expand` -- create one child per legal move from the network prior;
* :func:`apply_dirichlet_noise` -- AGZ root-prior noise (todo 10);
* :func:`temperature_policy` / :func:`sample_action` -- AGZ move selection
  with temperature (todo 10);
* :func:`run_search` -- ``simulations`` full selection/expansion/backup
  passes from a root (also exported on the :class:`MCTS` facade);
* :func:`visit_count_policy` / :func:`most_visited_action` -- the search
  output: a distribution over legal moves proportional to visit counts and
  the most-visited action.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts.batched_evaluator import DEFAULT_LEAF_BATCH, BatchedNetworkEvaluator
from omigamax.network.features import (
    HISTORY_STEPS,
    decode_policy,
    encode,
    index_to_point,
    pass_index,
    point_to_index,
)
from omigamax.rules import BLACK, WHITE, Board

# Defaults mirroring config/default.yaml; overridable per call / per MCTS.
DEFAULT_C_PUCT = 2.5
DEFAULT_KOMI = 7.5

# Type of a leaf evaluator: (node) -> (prior_probs (N**2+1,), value float).
# Todo 11 extends this contract to a batched evaluator.
Evaluator = Callable[["Node"], tuple[np.ndarray, float]]


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class Node:
    """A node of the MCTS tree, owned by the caller.

    Attributes:
        board: the Go position this node represents (a private snapshot --
            never mutated after the node is created).
        prior: prior probability ``P(s_parent, a)`` of the edge from the
            parent into this node (the root's prior is unused).
        visit_count: number of simulations that passed through this node.
        value_sum: accumulated value from this node's own perspective
            (the player to move at ``board``).
        children: ``{action index: Node}`` for every legal action; populated
            by :func:`expand` (empty for unexpanded / terminal leaves).
        legal_moves: tuple of legal action indices for the player to move
            (``None`` until computed; always includes the pass index).
        parent: the parent node (``None`` for the root).
        virtual_loss: counter claimed on a leaf while it is being evaluated
            (todo 10/11 virtual loss); ``select_child`` divides UCB by
            ``1 + N_child + virtual_loss`` while it is non-zero.
        noisy_prior: transient ``{action: noisy prior}`` override for the
            *root's* children set by :func:`apply_dirichlet_noise` (``None``
            when no noise is active). Selection uses the override at the root
            only; the stored ``child.prior`` values are never mutated.
        _color: explicitly threaded side to move (handicap positions, see
            :func:`make_root`); ``None`` falls back to parity / the parent
            chain via :attr:`color`.
    """

    def __init__(
        self,
        board: Board,
        prior: float = 0.0,
        legal_moves: tuple[int, ...] | None = None,
        parent: "Node | None" = None,
        color: "int | None" = None,
    ) -> None:
        self.board = board
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.legal_moves = legal_moves
        self.parent = parent
        self._color = color
        # --- todo-10/11 seams: virtual loss + root Dirichlet-noise override ---
        self.virtual_loss = 0
        self.noisy_prior: dict[int, float] | None = None

    # -- derived statistics ----------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean action value ``value_sum / visit_count`` (0 for a new node)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def is_expanded(self) -> bool:
        """True once :func:`expand` has created this node's children."""
        return bool(self.children)

    @property
    def color(self) -> int:
        """Side to move at this node.

        An explicitly threaded ``color`` (handicap positions, see
        :func:`make_root`) wins; otherwise the mover flips every ply up the
        parent chain (each child is the position after its parent's move, so
        the opponent is to play). A parentless node without a threaded colour
        falls back to move-count parity -- black opens, so an even move count
        means black to play. The parent-chain path is bit-identical to parity
        for every ordinary game (where black made move 0), and the only
        difference in a handicap position is exactly the correction we want:
        the true mover is threaded at the root and propagates down.
        """
        if self._color is not None:
            return self._color
        if self.parent is not None:
            return 3 - self.parent.color
        return BLACK if len(self.board.moves) % 2 == 0 else WHITE

    @property
    def history(self) -> tuple:
        """AGZ feature history: position snapshots, most recent first, <= 8.

        Reconstructed on demand by walking up to ``HISTORY_STEPS`` parents, so
        nodes stay lean (the feature encoding is only needed at leaf
        evaluation). Snapshot 0 is this node's own position.
        """
        snapshots = []
        node = self
        while node is not None and len(snapshots) < HISTORY_STEPS:
            snapshots.append(node.board.state)
            node = node.parent
        return tuple(snapshots)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Node(visits={self.visit_count}, prior={self.prior:.4f}, "
            f"q={self.q_value:.4f}, children={len(self.children)})"
        )


# ---------------------------------------------------------------------------
# tree construction helpers
# ---------------------------------------------------------------------------

def legal_actions(board: Board, color: "int | None" = None) -> tuple[int, ...]:
    """Ascending tuple of legal action indices for the player to move.

    ``color`` optionally forces the side to move (the true mover in handicap
    positions); ``None`` derives it from move-count parity -- black opens, so
    an even move count means black to play.

    Points come first in ``(row, col)`` order (index ``row*size + col``), then
    the pass index ``size**2`` (always legal). Ties in UCB selection break to
    the lowest index, so ascending order keeps selection deterministic.
    """
    size = board.size
    if color is None:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
    actions = [
        point_to_index(r, c, size)
        for r in range(size)
        for c in range(size)
        if board.is_legal((r, c), color)
    ]
    actions.append(pass_index(size))
    return tuple(actions)


def _copy_board(board: Board) -> Board:
    """Fast copy of a :class:`Board` for tree nodes.

    ``copy.deepcopy`` would work but is ~10x slower and this path (one child
    board per legal move per expansion) dominates the search cost. Reads the
    rules module's own (private) attributes; a plain ``list``/``tuple`` copy
    keeps the child fully independent of the parent's mutable history.
    """
    new_board = Board(board.size)
    new_board._state = list(board._state)
    new_board.moves = list(board.moves)
    new_board.pass_count = board.pass_count
    new_board.last_captured_point = board.last_captured_point
    return new_board


def make_root(board: Board, color: "int | None" = None) -> Node:
    """Build a root node for ``board`` (a copy is taken).

    ``color`` optionally forces the side to move -- the *true* mover in
    handicap positions, where move-count parity disagrees (handicap stones are
    BLACK moves but WHITE is to play, so with an even handicap the parity
    answer ``BLACK`` is wrong). ``None`` (the default) keeps the parity
    behaviour, correct for every ordinary non-handicap caller.

    The root's legal moves are computed eagerly so :func:`expand` can create
    the full child set on the first visit.
    """
    board = _copy_board(board)
    return Node(board=board, color=color, legal_moves=legal_actions(board, color))


def descend(root: Node, action: int) -> Node:
    """Return the child reached by ``action`` -- for cross-move tree reuse."""
    return root.children[action]


# ---------------------------------------------------------------------------
# UCB selection
# ---------------------------------------------------------------------------

def select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    """Pick the child maximising ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)``.

    ``Q`` is the child's mean value (0 for unvisited children); ``P`` is the
    child's prior -- the network prior, or the Dirichlet-noised override from
    ``node.noisy_prior`` when one is active (todo 10; only ever set on the
    root, so non-root selection always uses the true stored priors). ``N`` in
    the denominator is the child's *effective* visit count
    ``visit_count + virtual_loss`` (todo 10): a leaf claimed for evaluation
    (virtual loss 3) is depressed as if it had been visited ``virtual_loss``
    more times, and once the claim is reverted its UCB returns to the true
    value. Ties break to the lowest action index (deterministic, because
    ``children`` is inserted in ascending ``legal_moves`` order).

    The ``sqrt(N_parent) / (1 + N_child)`` form is division-safe for brand
    new children (``N_child == 0``) and for a root that has never been
    visited (``sqrt(0) == 0``) -- never change the formula shape.
    """
    sqrt_parent = math.sqrt(node.visit_count)
    noisy = node.noisy_prior if node.noisy_prior is not None else {}
    best_action: int | None = None
    best_score = -math.inf
    for action, child in node.children.items():
        prior = noisy.get(action, child.prior)
        effective_visits = child.visit_count + child.virtual_loss
        ucb = child.q_value + c_puct * prior * sqrt_parent / (1.0 + effective_visits)
        if ucb > best_score:
            best_score = ucb
            best_action = action
    return best_action, node.children[best_action]


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------

def expand(node: Node, prior_probs: np.ndarray) -> None:
    """Create one child per legal move of ``node`` from the network prior.

    ``prior_probs`` is the full ``(board_size**2 + 1,)`` distribution over
    legal moves (already masked by :func:`decode_policy` -- illegal actions
    carry zero mass and are skipped because they are not in
    ``node.legal_moves``). Each child owns a deep-copied board advanced by
    that action, the corresponding AGZ history, and the edge prior.
    """
    if node.legal_moves is None:
        node.legal_moves = legal_actions(node.board, node.color)
    size = node.board.size
    color = node.color
    for action in node.legal_moves:
        child_board = _copy_board(node.board)
        if action == pass_index(size):
            child_board.pass_move(color)
        else:
            row, col = index_to_point(action, size)
            # The action came from node.legal_moves (vetted by legal_actions,
            # which includes simple-ko), so the child build skips the legality
            # re-check -- the hot-path cost of doubling is_legal+is_ko per
            # child (P11 self-play speedup).
            child_board.play((row, col), color, check_legal=False)
        node.children[action] = Node(
            board=child_board,
            prior=float(prior_probs[action]),
            parent=node,
        )


# ---------------------------------------------------------------------------
# terminal value
# ---------------------------------------------------------------------------

def terminal_value(
    board: Board,
    komi: float = DEFAULT_KOMI,
    color: "int | None" = None,
) -> float:
    """Game outcome from the perspective of the player to move at ``board``.

    The position must be terminal (two consecutive passes). Returns +1 if the
    side to move won, -1 if it lost, 0 on jigo (impossible with komi 7.5).
    ``color`` optionally forces the side to move (handicap positions, where
    move-count parity disagrees); ``None`` derives it from parity.
    """
    winner = board.winner(komi)
    if winner is None:
        return 0.0
    if color is None:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
    current_is_black = color == BLACK
    if (winner == "B") == current_is_black:
        return 1.0
    return -1.0


# ---------------------------------------------------------------------------
# leaf evaluators (seam for todo 11)
# ---------------------------------------------------------------------------

class NetworkEvaluator:
    """Per-leaf synchronous network evaluator (todo 9).

    Encodes the leaf's AGZ 17-plane history, runs one network forward pass in
    ``eval()``/``no_grad()``, and returns ``(masked_prior, value)``. The
    caller is responsible for putting the network in ``eval()`` mode.

    Todo 11 replaces this with a batched evaluator behind the same contract;
    nothing else in this module depends on the per-leaf implementation.
    """

    def __init__(self, network: torch.nn.Module) -> None:
        self.network = network

    def __call__(self, node: Node) -> tuple[np.ndarray, float]:
        device = next(self.network.parameters()).device
        features = encode(node.history, node.color, board_size=node.board.size)
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = self.network(x)
        prior = decode_policy(logits, node.board, color=node.color,
                              legal_moves=node.legal_moves)
        return prior, float(value.reshape(-1)[0].item())


# ---------------------------------------------------------------------------
# search driver
# ---------------------------------------------------------------------------

def run_search(
    root: Node,
    network: torch.nn.Module | None,
    simulations: int,
    c_puct: float | None = None,
    komi: float | None = None,
    evaluator: Evaluator | None = None,
    dirichlet_alpha: float | None = None,
    dirichlet_eps: float | None = None,
    dirichlet_rng: np.random.Generator | None = None,
    virtual_loss: int | None = None,
    batch_size: int | None = None,
) -> Node:
    """Run ``simulations`` full MCTS passes from ``root`` (in place).

    Args:
        root: the search root (see :func:`make_root`).
        network: the policy-value network used for leaf evaluation. Ignored
            when ``evaluator`` is given (the todo-11 batch seam); required
            otherwise. When a plain ``network`` is used the leaves are
            evaluated by a batched :class:`BatchedNetworkEvaluator` (todo 11)
            with the config's ``leaf_batch``.
        simulations: number of selection/expansion/backup passes.
        c_puct: UCB exploration constant (default ``config c_puct`` = 2.5).
        komi: komi used for terminal scoring (default ``config komi`` = 7.5).
        evaluator: pluggable leaf evaluator. A per-leaf callable
            ``(node) -> (prior, value)`` runs the todo-9 synchronous path. A
            batched evaluator (has ``submit``/``flush``, e.g.
            :class:`BatchedNetworkEvaluator`) runs the todo-11 collect-then-
            flush path -- the default when ``evaluator`` is None.
        dirichlet_alpha: AGZ root-noise concentration. When ``None`` (the
            default) no noise is applied and any stale override is cleared;
            otherwise ``apply_dirichlet_noise(root, dirichlet_alpha,
            dirichlet_eps, rng=dirichlet_rng)`` blends the root's legal-child
            priors once, before the first simulation (AGZ re-samples per
            position -- callers apply it to a fresh root each move).
        dirichlet_eps: noise blend weight (default ``config dirichlet_eps``
            = 0.25, used only when ``dirichlet_alpha`` is given).
        dirichlet_rng: numpy generator for the noise draw.
        virtual_loss: virtual-loss value claimed on a leaf while it is being
            evaluated (default ``config virtual_loss`` = 3; 0 disables). With
            a batched evaluator every leaf of the in-flight batch carries the
            claim while the batch is pending (spreading selection across
            distinct branches) and the claim is released after the batch's
            flush (``try/finally``), so the final visit counts and policy are
            exact.
        batch_size: leaves collected before one network forward (todo 11).
            ``None`` uses ``config leaf_batch`` = 16 for a batched evaluator
            (and the synchronous path when the evaluator is a per-leaf
            callable). ``1`` degenerates to exact per-leaf evaluation.

    Returns:
        ``root``, with updated visit statistics.

    Batched evaluation (todo 11): instead of evaluating one leaf per forward
    pass, the loop collects up to ``batch_size`` leaves -- each virtual-loss
    claimed on submission -- then flushes them through a single forward
    (``evaluator.flush()``), expands every leaf with its batch prior, releases
    every claim and backs up every path. Selection never descends through an
    unexpanded leaf, so an already-pending leaf can only be re-selected when
    every reachable leaf is pending: the batch is flushed then and the
    selection is retried (no simulation is consumed). The final batch -- the
    tail, with fewer than ``batch_size`` leaves -- is flushed after the loop.
    """
    cfg = load_config()
    if c_puct is None:
        c_puct = float(cfg.get("c_puct", DEFAULT_C_PUCT))
    if komi is None:
        komi = float(cfg.get("komi", DEFAULT_KOMI))
    if virtual_loss is None:
        virtual_loss = int(cfg.get("virtual_loss", DEFAULT_VIRTUAL_LOSS))
    if evaluator is None:
        if network is None:
            raise ValueError("either `network` or `evaluator` must be provided")
        evaluator = NetworkEvaluator(network)

    # -- todo-11 batch wiring: detect a batched evaluator, pick batch size --
    batched = hasattr(evaluator, "submit") and hasattr(evaluator, "flush")
    if batch_size is None:
        batch_size = (
            int(cfg.get("leaf_batch", DEFAULT_LEAF_BATCH)) if batched else 1
        )
    batch_size = max(1, int(batch_size))

    # -- AGZ Dirichlet root noise: applied once, before the simulations --
    if dirichlet_alpha is not None:
        alpha = float(dirichlet_alpha)
        eps = (
            float(dirichlet_eps)
            if dirichlet_eps is not None
            else float(cfg.get("dirichlet_eps", DEFAULT_DIRICHLET_EPS))
        )
        apply_dirichlet_noise(root, alpha, eps, rng=dirichlet_rng)
    else:
        root.noisy_prior = None  # never let stale noise leak into a fresh run

    # Pending (in-flight) batch: (leaf, path from root to leaf). The batch is
    # flushed whenever it reaches batch_size, on re-selecting an already
    # pending leaf (nothing new to explore below it until it expands), and at
    # the very end (tail batch).
    pending: list[tuple[Node, list[Node]]] = []

    def flush_batch() -> None:
        """Run one batched forward over ``pending``, expand + backup each leaf.

        Virtual-loss claims on every batch member are released in ``finally``
        so an evaluator failure can never leak a claim.
        """
        if not pending:
            return
        try:
            results = evaluator.flush()
            if len(results) != len(pending):
                raise RuntimeError(
                    f"batched evaluator returned {len(results)} results for "
                    f"{len(pending)} submitted leaves"
                )
            for (node, path), (fnode, prior, value) in zip(pending, results):
                if fnode is not node:
                    raise RuntimeError(
                        "batched evaluator returned a leaf in a different "
                        "order than submitted"
                    )
                expand(node, prior)
                # -- backup: negate the perspective at every level --
                v = float(value)
                for visited in reversed(path):
                    visited.visit_count += 1
                    visited.value_sum += v
                    v = -v
        finally:
            for node, _ in pending:
                node.virtual_loss -= virtual_loss
            pending.clear()

    sims_done = 0
    while sims_done < int(simulations):
        # -- selection: descend until an unexpanded leaf --
        node = root
        path = [root]
        while node.is_expanded:
            _, node = select_child(node, c_puct)
            path.append(node)

        # -- terminal leaves never go through the evaluator --
        if node.legal_moves is None:
            node.legal_moves = legal_actions(node.board, node.color)
        if node.board.is_terminal():
            value = terminal_value(node.board, komi, color=node.color)
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value
            sims_done += 1
            if batched and len(pending) >= batch_size:
                flush_batch()
            continue

        if not batched:
            # -- per-leaf synchronous path (todo 9/10): claim, evaluate,
            #    expand, release -- bit-identical to the original loop. --
            node.virtual_loss += virtual_loss
            try:
                prior, value = evaluator(node)
                expand(node, prior)
            finally:
                node.virtual_loss -= virtual_loss
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value
            sims_done += 1
            continue

        # -- re-selecting an already-pending leaf: flush to free it --
        if any(pn is node for pn, _ in pending):
            if pending:
                flush_batch()
            continue  # no simulation consumed; retry selection on the new tree

        # -- expandable leaf: claim virtual loss and submit to the batch --
        node.virtual_loss += virtual_loss
        evaluator.submit(node)
        pending.append((node, path))
        sims_done += 1
        if len(pending) >= batch_size:
            flush_batch()

    # -- tail batch: fewer than batch_size leaves pending at the end --
    if batched and pending:
        flush_batch()

    return root


# ---------------------------------------------------------------------------
# search output
# ---------------------------------------------------------------------------

def visit_count_policy(root: Node) -> np.ndarray:
    """The search policy: visit counts normalized over the root's children.

    Per AGZ (Nature 2017 Methods), the policy is proportional to the *child*
    visit counts: ``pi(a) = N(root, a) / sum_b N(root, b)``. Normalizing over
    the children (not the root's own visit count) keeps the distribution
    exactly normalised even on the very first simulation, when the root is
    itself the expanded leaf and carries one visit with no child behind it.

    Returns a ``(board_size**2 + 1,)`` float32 array with one entry per
    action (points then pass); illegal actions are exactly 0. Sums to 1 once
    at least one simulation has run; a never-searched or terminal root (no
    children) yields all zeros -- no move should be drawn from a finished
    position.
    """
    size = root.board.size
    pi = np.zeros(size * size + 1, dtype=np.float32)
    if not root.children:
        return pi
    total = 0.0
    for action, child in root.children.items():
        pi[action] = child.visit_count
        total += child.visit_count
    if total > 0.0:
        pi /= total
    return pi


def most_visited_action(root: Node) -> int:
    """The action with the most visits under ``root`` (ties -> lowest index).

    Returns the pass index for a terminal root with no children.
    """
    if not root.children:
        return pass_index(root.board.size)
    action, _ = max(root.children.items(), key=lambda kv: (kv[1].visit_count, -kv[0]))
    return action


# ---------------------------------------------------------------------------
# AGZ search details (todo 10): Dirichlet root noise + temperature selection
# ---------------------------------------------------------------------------

# AGZ defaults mirroring config/default.yaml (overridable per call).
DEFAULT_DIRICHLET_ALPHA = 0.03
DEFAULT_DIRICHLET_EPS = 0.25
DEFAULT_TEMPERATURE_THRESHOLD = 30
DEFAULT_VIRTUAL_LOSS = 3
# Selection temperatures below this are treated as tau -> 0 (argmax), which
# avoids the division ``1 / tau`` blowing up (plan: tau < 1e-6 guard).
TAU_ARGMAX_THRESHOLD = 1e-6


def apply_dirichlet_noise(
    root: Node,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    epsilon: float = DEFAULT_DIRICHLET_EPS,
    rng: np.random.Generator | None = None,
) -> dict[int, float]:
    """Blend the root's legal-child priors with AGZ Dirichlet noise.

    The exact AGZ blend (Nature 2017 Methods) is

    ``P'(a) = (1 - epsilon) * P(a) + epsilon * eta(a)``,  ``eta ~ Dir(alpha)``,

    applied to every *legal child* of the root (``root.children`` -- exactly
    the legal moves, pass included). With ``alpha = 0.03`` and
    ``epsilon = 0.25`` (the config defaults) the draw is sparse -- most
    children get ``(1 - eps) * P`` and a handful get a sizeable boost.

    The result is *not* written into ``child.prior``: it is stored in the
    transient ``root.noisy_prior`` override (``select_child`` reads it only
    at the root), so the stored network priors are never mutated, re-running
    a search cannot compound stale noise, and non-root selection is
    unaffected. Callers that want noise must re-apply it to a fresh root
    every move (AGZ re-samples per position).

    Reproducible: pass a seeded ``rng`` (``np.random.default_rng(seed)``).

    Args:
        root: the search root whose *children's* priors are noised.
        alpha: Dirichlet concentration (``dirichlet_alpha``, default 0.03).
        epsilon: blend weight (``dirichlet_eps``, default 0.25).
        rng: numpy generator (defaults to a fresh ``default_rng()``).

    Returns:
        The ``{action: noisy_prior}`` override now stored on ``root`` (empty
        for an unexpanded root or when ``alpha <= 0`` or ``epsilon <= 0``).
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy: dict[int, float] = {}
    children = list(root.children.items())
    if children and alpha > 0.0 and epsilon > 0.0:
        eta = rng.dirichlet(np.full(len(children), float(alpha), dtype=np.float64))
        for (action, child), eta_a in zip(children, eta):
            noisy[action] = (1.0 - epsilon) * child.prior + epsilon * float(eta_a)
    root.noisy_prior = noisy if noisy else None
    return noisy


def clear_root_noise(root: Node) -> None:
    """Drop any Dirichlet-noise override from ``root`` (restore stored priors)."""
    root.noisy_prior = None


def temperature_policy(root: Node, temperature: float) -> np.ndarray:
    """The AGZ temperature-softened search policy over the root's children.

    ``pi(a) propto N(root, a) ** (1 / temperature)`` (visit counts), the
    search distribution used for move selection and, in todo 13, as the
    self-play training target. ``temperature = 1.0`` reproduces
    :func:`visit_count_policy` exactly; ``temperature -> 0`` (any value
    ``< TAU_ARGMAX_THRESHOLD``) concentrates all mass on the most-visited
    children -- ties share the mass uniformly, so sampling resolves them
    uniformly at random (the plan's "平局随机"). The ``tau < 1e-6`` guard
    short-circuits before ``1 / temperature`` could overflow.

    Computed in log-space (``(1/tau) * log N``, shifted by the max) so very
    small temperatures never overflow ``N ** (1/tau)``. A root with no
    visits (or no children) yields a uniform distribution over its children
    -- sampling still returns a legal move -- and all-zero for a root with
    no children at all (terminal position).

    Returns a ``(board_size**2 + 1,)`` float32 array, zero outside the
    root's children, summing to 1 whenever the root has children.
    """
    size = root.board.size
    pi = np.zeros(size * size + 1, dtype=np.float32)
    if not root.children:
        return pi

    if temperature < TAU_ARGMAX_THRESHOLD:
        # tau -> 0: argmax. Ties share the mass -> random tie-break on sampling.
        max_visits = max(c.visit_count for c in root.children.values())
        winners = [a for a, c in root.children.items() if c.visit_count == max_visits]
        share = 1.0 / len(winners)
        for action in winners:
            pi[action] = share
        return pi

    # tau > 0: pi propto N^(1/tau), in log space to avoid overflow.
    log_weights = {}
    for action, child in root.children.items():
        log_weights[action] = (
            math.log(child.visit_count) / temperature if child.visit_count > 0 else -math.inf
        )
    max_log = max(log_weights.values())
    if max_log == -math.inf:  # every child unvisited -> uniform over children
        share = 1.0 / len(root.children)
        for action in root.children:
            pi[action] = share
        return pi
    for action, log_w in log_weights.items():
        if log_w > -math.inf:
            pi[action] = math.exp(log_w - max_log)
    total = float(pi.sum())
    if total > 0.0:
        pi /= total
    return pi


def sample_action(
    root: Node,
    temperature: float,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one move from :func:`temperature_policy` at ``temperature``.

    ``temperature = 1.0`` samples proportional to visit counts (AGZ first
    ``temperature_threshold = 30`` moves); ``temperature -> 0`` always
    returns a most-visited action (ties resolved uniformly at random --
    reproducible with a seeded ``rng``). Falls back to
    :func:`most_visited_action` for a terminal root (no children).

    Args:
        root: the search root.
        temperature: the selection temperature (``tau``).
        rng: numpy generator for the sample (defaults to a fresh one).
    """
    if rng is None:
        rng = np.random.default_rng()
    pi = temperature_policy(root, temperature)
    actions = list(root.children)
    if not actions:
        return most_visited_action(root)
    probs = np.array([pi[a] for a in actions], dtype=np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        return most_visited_action(root)
    probs /= total
    idx = int(rng.choice(len(actions), p=probs))
    return actions[idx]


# ---------------------------------------------------------------------------
# facade
# ---------------------------------------------------------------------------

class MCTS:
    """High-level MCTS facade: owns the root across consecutive actions.

    ``network`` (or a custom ``evaluator``) evaluates leaves; ``c_puct`` /
    ``komi`` / AGZ search details default from ``config/default.yaml``. Usage::

        mcts = MCTS(network=model)
        root = mcts.new_root(board)   # reset the search tree
        mcts.run(simulations=200)     # search from the root
        pi = mcts.policy()            # visit-count distribution
        action = mcts.select_action() # most-visited move
        mcts.apply_action(action)     # reuse the tree: root becomes child

    Todo-10 knobs: ``dirichlet_alpha`` (``None`` = no root noise, otherwise
    the AGZ concentration applied to a fresh root on every ``run``),
    ``dirichlet_eps``, ``dirichlet_rng`` (seeded for reproducibility) and
    ``virtual_loss``. ``sample_action(tau, rng)`` implements AGZ temperature
    move selection (``tau = 1`` early, ``tau -> 0`` = argmax later).
    """

    def __init__(
        self,
        network: torch.nn.Module | None = None,
        c_puct: float | None = None,
        komi: float | None = None,
        evaluator: Evaluator | None = None,
        dirichlet_alpha: float | None = None,
        dirichlet_eps: float | None = None,
        dirichlet_rng: np.random.Generator | None = None,
        virtual_loss: int | None = None,
        leaf_batch: "int | None" = None,
    ) -> None:
        cfg = load_config()
        self.network = network
        self.c_puct = float(c_puct) if c_puct is not None else float(cfg.get("c_puct", DEFAULT_C_PUCT))
        self.komi = float(komi) if komi is not None else float(cfg.get("komi", DEFAULT_KOMI))
        self.evaluator = evaluator
        self.leaf_batch = int(leaf_batch) if leaf_batch is not None else None
        self.dirichlet_alpha = (
            float(dirichlet_alpha) if dirichlet_alpha is not None else None
        )
        self.dirichlet_eps = (
            float(dirichlet_eps)
            if dirichlet_eps is not None
            else float(cfg.get("dirichlet_eps", DEFAULT_DIRICHLET_EPS))
        )
        self.dirichlet_rng = dirichlet_rng
        self.virtual_loss = (
            int(virtual_loss)
            if virtual_loss is not None
            else int(cfg.get("virtual_loss", DEFAULT_VIRTUAL_LOSS))
        )
        self.root: Node | None = None

    def new_root(self, board: Board, color: "int | None" = None) -> Node:
        """Reset the search tree to a fresh root over ``board``.

        ``color`` optionally forces the side to move (the true mover in
        handicap positions); ``None`` keeps the parity default.
        """
        self.root = make_root(board, color=color)
        return self.root

    def run(self, simulations: int) -> Node:
        """Run ``simulations`` MCTS passes from the current root."""
        if self.root is None:
            raise RuntimeError("call new_root(board) before run()")
        return run_search(
            self.root,
            self.network,
            simulations,
            c_puct=self.c_puct,
            komi=self.komi,
            evaluator=self.evaluator,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_eps=self.dirichlet_eps,
            dirichlet_rng=self.dirichlet_rng,
            virtual_loss=self.virtual_loss,
            batch_size=self.leaf_batch,
        )

    def policy(self) -> np.ndarray:
        """Visit-count distribution over the root's legal moves."""
        if self.root is None:
            raise RuntimeError("call new_root(board) before policy()")
        return visit_count_policy(self.root)

    def select_action(self) -> int:
        """The most-visited action at the root."""
        if self.root is None:
            raise RuntimeError("call new_root(board) before select_action()")
        return most_visited_action(self.root)

    def sample_action(self, temperature: float, rng: np.random.Generator | None = None) -> int:
        """Sample a move with AGZ temperature selection (todo 10).

        ``tau = 1.0`` samples proportional to visit counts (the first
        ``temperature_threshold`` moves); ``tau < 1e-6`` resolves to the
        argmax (ties uniform-random, reproducible with a seeded ``rng``).
        """
        if self.root is None:
            raise RuntimeError("call new_root(board) before sample_action()")
        return sample_action(self.root, temperature, rng=rng)

    def apply_action(self, action: int) -> Node:
        """Advance the tree: the chosen child becomes the new root."""
        if self.root is None:
            raise RuntimeError("call new_root(board) before apply_action()")
        self.root = descend(self.root, action)
        return self.root
