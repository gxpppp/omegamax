"""MCTS tree-search core (todo 9): selection, expansion, backup (UCB).

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

Design seams for later todos (deliberately *not* implemented here):

* ``Node.virtual_loss`` (default 0) -- the counter that todo 10's virtual
  loss and todo 11's batched leaf evaluation will use as a placeholder;
* the ``evaluator`` interface ``callable(node) -> (prior_probs, value)`` --
  todo 11 swaps the per-leaf synchronous
  :class:`NetworkEvaluator` for a batched one with the same contract.

Structure of the module:

* :class:`Node` -- the tree node (``visit_count``, ``prior``, ``value_sum``,
  ``children``, ``legal_moves``, plus the position snapshot the node owns);
* :func:`select_child` -- the UCB selection step;
* :func:`expand` -- create one child per legal move from the network prior;
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
        virtual_loss: placeholder counter reserved for todo 10/11 (virtual
            loss + batched leaf evaluation); unused in todo 9.
    """

    def __init__(
        self,
        board: Board,
        prior: float = 0.0,
        legal_moves: tuple[int, ...] | None = None,
        parent: "Node | None" = None,
    ) -> None:
        self.board = board
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.legal_moves = legal_moves
        self.parent = parent
        # --- todo-10/11 seam: virtual loss + batched leaf evaluation ---
        self.virtual_loss = 0

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
        """Side to move at this node (black opens, so even move count -> black)."""
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

def legal_actions(board: Board) -> tuple[int, ...]:
    """Ascending tuple of legal action indices for the player to move.

    Points come first in ``(row, col)`` order (index ``row*size + col``), then
    the pass index ``size**2`` (always legal). Ties in UCB selection break to
    the lowest index, so ascending order keeps selection deterministic.
    """
    size = board.size
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


def make_root(board: Board) -> Node:
    """Build a root node for ``board`` (a copy is taken).

    The root's legal moves are computed eagerly so :func:`expand` can create
    the full child set on the first visit.
    """
    board = _copy_board(board)
    return Node(board=board, legal_moves=legal_actions(board))


def descend(root: Node, action: int) -> Node:
    """Return the child reached by ``action`` -- for cross-move tree reuse."""
    return root.children[action]


# ---------------------------------------------------------------------------
# UCB selection
# ---------------------------------------------------------------------------

def select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    """Pick the child maximising ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)``.

    ``Q`` is the child's mean value (0 for unvisited children); ``P`` is the
    child's prior. Ties break to the lowest action index (deterministic,
    because ``children`` is inserted in ascending ``legal_moves`` order).

    The ``sqrt(N_parent) / (1 + N_child)`` form is division-safe for brand
    new children (``N_child == 0``) and for a root that has never been
    visited (``sqrt(0) == 0``) -- never change the formula shape.
    """
    sqrt_parent = math.sqrt(node.visit_count)
    best_action: int | None = None
    best_score = -math.inf
    for action, child in node.children.items():
        ucb = child.q_value + c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count)
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
        node.legal_moves = legal_actions(node.board)
    size = node.board.size
    color = node.color
    for action in node.legal_moves:
        child_board = _copy_board(node.board)
        if action == pass_index(size):
            child_board.pass_move(color)
        else:
            row, col = index_to_point(action, size)
            child_board.play((row, col), color)
        node.children[action] = Node(
            board=child_board,
            prior=float(prior_probs[action]),
            parent=node,
        )


# ---------------------------------------------------------------------------
# terminal value
# ---------------------------------------------------------------------------

def terminal_value(board: Board, komi: float = DEFAULT_KOMI) -> float:
    """Game outcome from the perspective of the player to move at ``board``.

    The position must be terminal (two consecutive passes). Returns +1 if the
    side to move won, -1 if it lost, 0 on jigo (impossible with komi 7.5).
    """
    winner = board.winner(komi)
    if winner is None:
        return 0.0
    current_is_black = BLACK if len(board.moves) % 2 == 0 else WHITE
    if (winner == "B") == (current_is_black == BLACK):
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
        prior = decode_policy(logits, node.board)
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
) -> Node:
    """Run ``simulations`` full MCTS passes from ``root`` (in place).

    Args:
        root: the search root (see :func:`make_root`).
        network: the policy-value network used for leaf evaluation. Ignored
            when ``evaluator`` is given (the todo-11 batch seam); required
            otherwise.
        simulations: number of selection/expansion/backup passes.
        c_puct: UCB exploration constant (default ``config c_puct`` = 2.5).
        komi: komi used for terminal scoring (default ``config komi`` = 7.5).
        evaluator: pluggable leaf evaluator ``(node) -> (prior, value)``;
            defaults to a per-leaf :class:`NetworkEvaluator`.

    Returns:
        ``root``, with updated visit statistics.
    """
    cfg = load_config()
    if c_puct is None:
        c_puct = float(cfg.get("c_puct", DEFAULT_C_PUCT))
    if komi is None:
        komi = float(cfg.get("komi", DEFAULT_KOMI))
    if evaluator is None:
        if network is None:
            raise ValueError("either `network` or `evaluator` must be provided")
        evaluator = NetworkEvaluator(network)

    for _ in range(int(simulations)):
        # -- selection: descend until an unexpanded leaf --
        node = root
        path = [root]
        while node.is_expanded:
            _, node = select_child(node, c_puct)
            path.append(node)

        # -- expansion (skipped for terminal positions) --
        if node.legal_moves is None:
            node.legal_moves = legal_actions(node.board)
        if node.board.is_terminal():
            value = terminal_value(node.board, komi)
        else:
            prior, value = evaluator(node)
            expand(node, prior)

        # -- backup: accumulate along the path, negating the perspective --
        for visited in reversed(path):
            visited.visit_count += 1
            visited.value_sum += value
            value = -value

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
# facade
# ---------------------------------------------------------------------------

class MCTS:
    """High-level MCTS facade: owns the root across consecutive actions.

    ``network`` (or a custom ``evaluator``) evaluates leaves; ``c_puct`` /
    ``komi`` default from ``config/default.yaml``. Usage::

        mcts = MCTS(network=model)
        root = mcts.new_root(board)   # reset the search tree
        mcts.run(simulations=200)     # search from the root
        pi = mcts.policy()            # visit-count distribution
        action = mcts.select_action() # most-visited move
        mcts.apply_action(action)     # reuse the tree: root becomes child
    """

    def __init__(
        self,
        network: torch.nn.Module | None = None,
        c_puct: float | None = None,
        komi: float | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        cfg = load_config()
        self.network = network
        self.c_puct = float(c_puct) if c_puct is not None else float(cfg.get("c_puct", DEFAULT_C_PUCT))
        self.komi = float(komi) if komi is not None else float(cfg.get("komi", DEFAULT_KOMI))
        self.evaluator = evaluator
        self.root: Node | None = None

    def new_root(self, board: Board) -> Node:
        """Reset the search tree to a fresh root over ``board``."""
        self.root = make_root(board)
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

    def apply_action(self, action: int) -> Node:
        """Advance the tree: the chosen child becomes the new root."""
        if self.root is None:
            raise RuntimeError("call new_root(board) before apply_action()")
        self.root = descend(self.root, action)
        return self.root
