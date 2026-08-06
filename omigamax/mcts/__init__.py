"""Monte Carlo tree search: selection, expansion, backup (todo 9) + AGZ
details (todo 10: Dirichlet root noise / temperature / virtual loss) +
batched leaf evaluation (todo 11: :class:`BatchedNetworkEvaluator`, default
for ``MCTS(network=...)``).

Todo 12 (strength vs. simulation count) and todo 13 (self-play) build on this
module -- see ``omigamax/mcts/mcts.py`` for the search driver.
"""

from omigamax.mcts.batched_evaluator import DEFAULT_LEAF_BATCH, BatchedNetworkEvaluator
from omigamax.mcts.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPS,
    DEFAULT_KOMI,
    DEFAULT_TEMPERATURE_THRESHOLD,
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

__all__ = [
    "DEFAULT_C_PUCT",
    "DEFAULT_DIRICHLET_ALPHA",
    "DEFAULT_DIRICHLET_EPS",
    "DEFAULT_KOMI",
    "DEFAULT_LEAF_BATCH",
    "DEFAULT_TEMPERATURE_THRESHOLD",
    "DEFAULT_VIRTUAL_LOSS",
    "MCTS",
    "Node",
    "NetworkEvaluator",
    "BatchedNetworkEvaluator",
    "TAU_ARGMAX_THRESHOLD",
    "apply_dirichlet_noise",
    "clear_root_noise",
    "descend",
    "expand",
    "legal_actions",
    "make_root",
    "most_visited_action",
    "run_search",
    "sample_action",
    "select_child",
    "temperature_policy",
    "terminal_value",
    "visit_count_policy",
]
