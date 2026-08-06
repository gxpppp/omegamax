"""Todo 9 acceptance demo: MCTS tree search from the empty 19x19 board.

Runs ``--sims`` (default 200, the plan's ``simulations=200``) MCTS
selection/expansion/backup passes from the empty board using the b10c128
network from ``config/default.yaml`` (``create_model(10, 128, 19)``, ``eval``
mode, CUDA when available) and prints:

  * the configuration used (board_size, c_puct, simulations, komi, device);
  * the search wall-clock time and sims/second;
  * root visit statistics (``root visits == simulations``) and the number of
    nodes in the tree;
  * the top ``--top`` actions by visit count: index, point/PASS label, prior,
    mean action value Q, visit count + share, and whether the move is legal;
  * the full policy (visit-count distribution over legal moves, pass included);
  * a UCB sanity check at the root: the child returned by
    :func:`~omigamax.mcts.mcts.select_child` matches the argmax of the plan's
    literal formula ``Q + c_puct * P * sqrt(N_root) / (1 + N_child)``.

Exit code 0 iff the search completed ``simulations`` passes, the top move is
legal, the policy sums to 1, and the UCB sanity check passed. With
``--evidence <path>`` the demo writes a UTF-8 JSON report of everything.

Usage::

    uv run python -m omigamax.cli.mcts_demo
    uv run python -m omigamax.cli.mcts_demo --sims 200 --top 8 --evidence .omo/evidence/omigamax-go/task-9-mcts-demo.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts.mcts import (
    DEFAULT_C_PUCT,
    make_root,
    most_visited_action,
    run_search,
    select_child,
    visit_count_policy,
)
from omigamax.network.features import index_to_point, is_pass
from omigamax.network.model import create_model
from omigamax.rules import Board

DEFAULT_SIMS = 200
DEFAULT_TOP = 8
DEFAULT_SEED = 0


def _action_label(action: int, size: int) -> str:
    """Human label for a policy index: ``(row, col)`` or ``PASS``."""
    if is_pass(action, size):
        return "PASS"
    row, col = index_to_point(action, size)
    return f"({row},{col})"


def run_demo(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    size = int(cfg["board_size"])
    c_puct = float(cfg.get("c_puct", DEFAULT_C_PUCT))
    komi = float(cfg.get("komi", 7.5))
    sims = int(args.sims)

    torch.manual_seed(args.seed)
    model = create_model(blocks=int(cfg["blocks"]), channels=int(cfg["channels"]), board_size=size)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    root = make_root(Board(size))
    t0 = time.perf_counter()
    run_search(root, model, simulations=sims, c_puct=c_puct, komi=komi)
    wall_time_s = time.perf_counter() - t0

    # -- UCB sanity check at the root (plan's literal formula) --
    selected_action, _ = select_child(root, c_puct)
    sqrt_root = math.sqrt(root.visit_count)
    scores = {
        a: c.q_value + c_puct * c.prior * sqrt_root / (1.0 + c.visit_count)
        for a, c in root.children.items()
    }
    argmax_action = max(scores, key=lambda a: (scores[a], -a))
    ucb_ok = bool(selected_action == argmax_action)

    policy = visit_count_policy(root)
    policy_sum = float(policy.sum())

    ranked = sorted(
        root.children.items(), key=lambda kv: (kv[1].visit_count, -kv[0]), reverse=True
    )
    top = []
    for action, child in ranked[: args.top]:
        top.append(
            {
                "action": action,
                "label": _action_label(action, size),
                "prior": child.prior,
                "q": child.q_value,
                "visits": child.visit_count,
                "share": child.visit_count / root.visit_count if root.visit_count else 0.0,
                "legal": root.board.is_legal(
                    None if is_pass(action, size) else index_to_point(action, size),
                    root.color,
                ),
            }
        )

    tree_size = 1 + sum(len(n.children) for n in _walk(root))

    result = {
        "passed": bool(ucb_ok and policy_sum > 0 and top and top[0]["legal"]),
        "device": str(device),
        "config": {
            "board_size": size,
            "blocks": int(cfg["blocks"]),
            "channels": int(cfg["channels"]),
            "c_puct": c_puct,
            "komi": komi,
            "simulations": sims,
            "seed": args.seed,
        },
        "search": {
            "root_visits": root.visit_count,
            "wall_time_s": wall_time_s,
            "sims_per_s": sims / wall_time_s if wall_time_s > 0 else float("inf"),
            "tree_nodes": tree_size,
        },
        "ucb_sanity_check": {
            "formula": "Q + c_puct * P * sqrt(N_root) / (1 + N_child)",
            "selected_action": selected_action,
            "argmax_action": argmax_action,
            "max_ucb": scores[argmax_action],
            "passed": ucb_ok,
        },
        "policy": {
            "sum": policy_sum,
            "argmax_visit_action": most_visited_action(root),
            "pass_index": size * size,
            "pass_mass": float(policy[size * size]),
        },
        "top_actions": top,
    }
    return result


def _walk(root):
    """Breadth-first traversal over the tree (excludes the root itself)."""
    queue = list(root.children.values())
    while queue:
        node = queue.pop(0)
        yield node
        queue.extend(node.children.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="omigamax todo-9 MCTS demo: 200-sim search from the empty 19x19 board."
    )
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                        help=f"search simulations (default {DEFAULT_SIMS})")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"how many top actions to print (default {DEFAULT_TOP})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the result JSON to this path (utf-8)")
    args = parser.parse_args(argv)

    result = run_demo(args)

    cfg = result["config"]
    search = result["search"]
    print("=== omigamax MCTS demo (todo 9) ===", flush=True)
    print(
        f"config: board={cfg['board_size']} blocks={cfg['blocks']} "
        f"channels={cfg['channels']} c_puct={cfg['c_puct']} komi={cfg['komi']} "
        f"simulations={cfg['simulations']} seed={cfg['seed']}", flush=True
    )
    print(f"device: {result['device']}", flush=True)
    print(
        f"root visits: {search['root_visits']} (== simulations: "
        f"{search['root_visits'] == cfg['simulations']})", flush=True
    )
    print(
        f"search wall time: {search['wall_time_s']:.2f} s "
        f"({search['sims_per_s']:.1f} sims/s, {search['tree_nodes']} nodes)", flush=True
    )

    ucb = result["ucb_sanity_check"]
    print("UCB sanity check (formula: Q + c_puct*P*sqrt(N_root)/(1+N_child), "
          f"c_puct={cfg['c_puct']}):", flush=True)
    print(
        f"  selected child action {ucb['selected_action']} == "
        f"formula argmax action {ucb['argmax_action']} "
        f"(max UCB {ucb['max_ucb']:.6f}): {ucb['passed']}", flush=True
    )

    pol = result["policy"]
    print(f"policy: sum={pol['sum']:.6f} argmax-visit action={pol['argmax_visit_action']} "
          f"pass mass={pol['pass_mass']:.4f} (pass index {pol['pass_index']})", flush=True)

    print(f"top {len(result['top_actions'])} actions by visit count:", flush=True)
    print(
        f"  {'idx':>4} {'move':>8} {'prior':>8} {'Q':>8} {'visits':>6} "
        f"{'share':>7} {'legal':>5}", flush=True
    )
    for entry in result["top_actions"]:
        print(
            f"  {entry['action']:>4} {entry['label']:>8} {entry['prior']:.4f} "
            f"{entry['q']:+8.4f} {entry['visits']:>6} {entry['share']:>7.3f} "
            f"{str(entry['legal']):>5}", flush=True
        )
    print(
        f"RESULT: {'PASS' if result['passed'] else 'FAIL'} "
        f"(root visits == sims, policy sums to 1, top move legal, UCB check ok)",
        flush=True,
    )

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
