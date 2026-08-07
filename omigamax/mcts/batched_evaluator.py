"""Batched leaf evaluation for MCTS (todo 11).

AlphaGo Zero evaluates leaves in batches of up to ``leaf_batch`` positions in a
single network forward (the paper's 64-TPU batch; the config default here is
``leaf_batch = 16``). This module is the network side of that seam:

* :class:`BatchedNetworkEvaluator` accumulates leaves via :meth:`submit` while
  the search loop holds virtual-loss claims on them, then runs *one* forward
  over the whole batch via :meth:`flush` and hands the ``(prior, value)`` pair
  back for each leaf (in submission order). The caller (:func:`omigamax.mcts.mcts.run_search`)
  owns the tree semantics -- expansion, virtual-loss release and backup -- so
  this class stays a pure "encode + forward + decode" unit.

Guarantee: for the *same positions* the batch forward is bit-exact with the
per-leaf :class:`~omigamax.mcts.mcts.NetworkEvaluator` forward (each sample in
a batch is evaluated independently, and the AGZ 17-plane encoding per leaf is
identical), so the plan's "批推理结果与逐叶子推理输出全等（同网络同局面）"
holds. The module never mutates the tree: expansion / backup / virtual-loss
release all happen in ``mcts.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.network.features import decode_policy, encode_batch

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids a circular import)
    from omigamax.mcts.mcts import Node

DEFAULT_LEAF_BATCH = 16


class BatchedNetworkEvaluator:
    """Network leaf evaluator that batches up to ``batch_size`` leaves.

    Usage (the todo-11 search loop protocol)::

        ev = BatchedNetworkEvaluator(model, batch_size=16)
        ev.submit(leaf_a)          # accumulate a virtual-loss-claimed leaf
        ev.submit(leaf_b)
        results = ev.flush()       # one forward over {a, b} -> [(a, p, v), (b, p, v)]

    ``flush()`` returns results in submission order. A ``batch_size`` of 1
    degenerates to exact per-leaf evaluation (used by the equivalence tests).

    The class also exposes the todo-9 ``Evaluator`` contract ``__call__(node)
    -> (prior, value)`` for one-off / batch-size-1 use.

    Attributes (instrumentation the tests assert on):
        batch_sizes: sizes of every flushed batch (recorded after each flush).
        forwards: number of network forward passes run.
        leaves_evaluated: total leaves sent through ``flush()``.
    """

    def __init__(self, network: torch.nn.Module, batch_size: int | None = None) -> None:
        self.network = network
        if batch_size is None:
            batch_size = int(load_config().get("leaf_batch", DEFAULT_LEAF_BATCH))
        self.batch_size = max(1, int(batch_size))
        self.pending: list["Node"] = []
        # -- instrumentation (asserted on by tests) --
        self.batch_sizes: list[int] = []
        self.forwards = 0
        self.leaves_evaluated = 0

    # -- batch protocol ---------------------------------------------------

    def submit(self, node: "Node") -> None:
        """Accumulate ``node`` for the next batch (virtual loss already claimed).

        The search loop claims ``virtual_loss`` on the leaf *before* calling
        this, so every member of an in-flight batch is depressed in UCB while
        the batch is pending -- that is exactly what spreads selection across
        distinct branches during collection.
        """
        self.pending.append(node)

    @property
    def pending_count(self) -> int:
        """Number of leaves currently accumulated (not yet flushed)."""
        return len(self.pending)

    @property
    def is_full(self) -> bool:
        """True once the pending batch has reached ``batch_size`` leaves."""
        return len(self.pending) >= self.batch_size

    def flush(self) -> "list[tuple[Node, np.ndarray, float]]":
        """Evaluate every pending leaf in one network forward.

        Encodes the pending leaves with :func:`omigamax.network.features.encode_batch`
        (AGZ 17 planes), runs a single ``eval()``/``no_grad()`` forward, masks
        each row's policy logits to that leaf's legal moves, and returns
        ``[(node, prior, value), ...]`` **in submission order**. The pending
        batch is cleared; a tail batch (fewer than ``batch_size`` leaves) is
        handled naturally -- it is flushed as-is.

        ``expand``/backup are the caller's job (``mcts.py`` owns the tree), so
        this method never touches ``node.children``.
        """
        leaves = list(self.pending)
        self.pending = []
        if not leaves:
            return []

        device = next(self.network.parameters()).device
        board_size = leaves[0].board.size
        features = encode_batch(
            [node.history for node in leaves],
            [node.color for node in leaves],
            board_size=board_size,
        )
        x = torch.from_numpy(features).to(device)
        with torch.no_grad():
            logits, value = self.network(x)

        logits_np = logits.detach().cpu().numpy()
        values_np = value.detach().cpu().numpy().reshape(-1)
        results: list[tuple["Node", np.ndarray, float]] = []
        for node, logit_row, v in zip(leaves, logits_np, values_np):
            prior = decode_policy(logit_row, node.board, color=node.color)
            results.append((node, prior, float(v)))

        self.batch_sizes.append(len(results))
        self.forwards += 1
        self.leaves_evaluated += len(results)
        return results

    # -- per-leaf contract (todo-9 ``Evaluator``) -------------------------

    def __call__(self, node: "Node") -> tuple[np.ndarray, float]:
        """Evaluate a single leaf synchronously (batch-size-1 convenience).

        Intended for the degenerate / single-leaf case; if other leaves are
        already pending they are flushed together with ``node`` and this
        returns ``node``'s own result (results are matched by identity).
        """
        self.submit(node)
        for fnode, prior, value in self.flush():
            if fnode is node:
                return prior, value
        raise RuntimeError(f"evaluated leaf {node!r} not found in flush results")
