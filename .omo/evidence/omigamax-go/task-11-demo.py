"""Todo-11 evidence script: batched vs per-leaf equivalence + memory + sims/s.

Runs the plan's todo-11 acceptance checks against the real implementation and
writes ``task-11-batch.json``:

1. **Forward equivalence (全等)**: a 16-leaf batch forward vs 16 per-leaf
   forwards -- the plan's "同网络同局面" policy/value equality (reported as
   max abs deviation; float32 kernel-shape rounding is ~1e-9, see the test
   module for the precision rationale);
2. **search equivalence**: ``batch_size=1`` batched search == per-leaf search
   (identical visit counts + final policy, bit-exact);
3. **determinism**: two identical batched searches -> identical trees;
4. **memory**: peak ``torch.cuda.max_memory_allocated`` of a 200-sim batched
   search on a real 19x19 b10c128 network with ``leaf_batch=16`` (< 5 GB);
5. **timing baseline (sims/s)**: 200 and 800 sims from the empty 19x19 root,
   batched vs per-leaf, plus the speed-up.

Run from the repo root:

    uv run python .omo/evidence/omigamax-go/task-11-demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts import (
    DEFAULT_LEAF_BATCH,
    DEFAULT_VIRTUAL_LOSS,
    BatchedNetworkEvaluator,
    NetworkEvaluator,
    expand,
    make_root,
    run_search,
    visit_count_policy,
)
from omigamax.network.model import create_model
from omigamax.rules import Board

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GIB = 1024**3
EVIDENCE = Path(__file__).resolve().parent / "task-11-batch.json"


def uniform_prior(size: int) -> np.ndarray:
    prior = np.ones(size * size + 1, dtype=np.float32)
    prior /= prior.sum()
    return prior


def walk_tree(root):
    queue = list(root.children.values())
    while queue:
        node = queue.pop(0)
        yield node
        queue.extend(node.children.values())


def node_key(node) -> tuple:
    return tuple(node.board.moves)


def count_visits(root) -> dict:
    visits = {node_key(root): root.visit_count}
    for node in walk_tree(root):
        visits[node_key(node)] = node.visit_count
    return visits


def forward_equivalence() -> dict:
    """Batch forward vs per-leaf forward on the same positions (small model)."""
    torch.manual_seed(0)
    model = create_model(blocks=1, channels=8, board_size=5).eval()
    root = make_root(Board(5))
    expand(root, uniform_prior(5))
    for child in root.children.values():
        expand(child, uniform_prior(5))
    leaves = [gc for c in root.children.values() for gc in c.children.values()][:16]

    per_leaf = NetworkEvaluator(model)
    leaf_results = [per_leaf(leaf) for leaf in leaves]
    ev = BatchedNetworkEvaluator(model, batch_size=16)
    for leaf in leaves:
        ev.submit(leaf)
    batch_results = ev.flush()

    max_prior_diff = 0.0
    max_value_diff = 0.0
    for (_, prior_b, value_b), (prior_l, value_l) in zip(batch_results, leaf_results):
        max_prior_diff = max(max_prior_diff, float(np.abs(prior_b - prior_l).max()))
        max_value_diff = max(max_value_diff, abs(value_b - value_l))
    return {
        "leaves": len(leaves),
        "forwards": ev.forwards,
        "max_prior_abs_diff": max_prior_diff,
        "max_value_abs_diff": max_value_diff,
        "equal_within_float32": max_prior_diff < 1e-6 and max_value_diff < 1e-6,
    }


def search_equivalence_batch_size_one() -> dict:
    """batch_size=1 batched search vs per-leaf search: identical tree."""
    torch.manual_seed(11)
    model = create_model(blocks=1, channels=8, board_size=5).eval()
    r1 = make_root(Board(5))
    r2 = make_root(Board(5))
    run_search(r1, None, 25, evaluator=NetworkEvaluator(model))
    run_search(r2, None, 25, evaluator=BatchedNetworkEvaluator(model, batch_size=1))
    identical = (
        count_visits(r1) == count_visits(r2)
        and bool(np.array_equal(visit_count_policy(r1), visit_count_policy(r2)))
    )
    return {"identical_visit_counts_and_policy": identical}


def determinism() -> dict:
    torch.manual_seed(42)
    model = create_model(blocks=1, channels=8, board_size=5).eval()
    ra = make_root(Board(5))
    rb = make_root(Board(5))
    run_search(ra, None, 30, evaluator=BatchedNetworkEvaluator(model, batch_size=4))
    run_search(rb, None, 30, evaluator=BatchedNetworkEvaluator(model, batch_size=4))
    identical = (
        count_visits(ra) == count_visits(rb)
        and bool(np.array_equal(visit_count_policy(ra), visit_count_policy(rb)))
    )
    return {"identical": identical}


def time_search(cfg: dict, sims: int, batched: bool) -> dict:
    blocks, channels, board = int(cfg["blocks"]), int(cfg["channels"]), int(cfg["board_size"])
    leaf_batch = int(cfg.get("leaf_batch", DEFAULT_LEAF_BATCH))
    torch.manual_seed(0)
    model = create_model(blocks, channels, board).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    ev = (
        BatchedNetworkEvaluator(model, batch_size=leaf_batch)
        if batched
        else NetworkEvaluator(model)
    )
    root = make_root(Board(board))
    t0 = time.perf_counter()
    run_search(root, None, sims, evaluator=ev, virtual_loss=int(cfg.get("virtual_loss", 3)))
    wall = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / GIB
    return {
        "mode": "batched" if batched else "per-leaf",
        "simulations": sims,
        "wall_time_s": wall,
        "sims_per_sec": sims / wall,
        "peak_GB": peak_gb,
        "forwards": getattr(ev, "forwards", None),
        "avg_batch_size": (
            ev.leaves_evaluated / ev.forwards if batched and ev.forwards else None
        ),
    }


def memory_and_timing(cfg: dict) -> dict:
    mem_200 = time_search(cfg, 200, batched=True)          # memory gate
    mem_200_perleaf = time_search(cfg, 200, batched=False)  # before-batch baseline
    timing_800 = time_search(cfg, 800, batched=True)
    return {
        "memory_gate": {
            **mem_200,
            "under_5GB": mem_200["peak_GB"] < 5.0,
        },
        "timing_200_batched": mem_200,
        "timing_200_per_leaf": mem_200_perleaf,
        "speedup_200_batched_vs_per_leaf": (
            mem_200_perleaf["wall_time_s"] / mem_200["wall_time_s"]
        ),
        "timing_800_batched": timing_800,
    }


def main() -> int:
    cfg = load_config()
    result = {
        "plan_todo": "11",
        "task": "MCTS 批推理 (batched leaf evaluation)",
        "config": {
            "leaf_batch": int(cfg["leaf_batch"]),
            "virtual_loss": int(cfg["virtual_loss"]),
            "simulations": int(cfg["simulations"]),
        },
    }
    print("=== todo-11 evidence: batched leaf evaluation ===", flush=True)

    eq = forward_equivalence()
    result["forward_equivalence"] = eq
    print(f"[forward equivalence] leaves={eq['leaves']} "
          f"forwards={eq['forwards']} max_prior_diff={eq['max_prior_abs_diff']:.3e} "
          f"max_value_diff={eq['max_value_abs_diff']:.3e} "
          f"equal_within_float32={eq['equal_within_float32']}", flush=True)

    s1 = search_equivalence_batch_size_one()
    result["search_equivalence_batch_size_one"] = s1
    print(f"[search equivalence batch_size=1] {s1['identical_visit_counts_and_policy']}", flush=True)

    det = determinism()
    result["determinism"] = det
    print(f"[determinism] {det['identical']}", flush=True)

    if torch.cuda.is_available():
        mt = memory_and_timing(cfg)
        result["memory_and_timing"] = mt
        g = mt["memory_gate"]
        print(f"[memory gate] peak_GB={g['peak_GB']:.4f} (200 sims, 19x19, "
              f"leaf_batch=16) under_5GB={g['under_5GB']}", flush=True)
        b200, p200, b800 = mt["timing_200_batched"], mt["timing_200_per_leaf"], mt["timing_800_batched"]
        print(f"[timing 200 sims] batched: {b200['sims_per_sec']:.1f} sims/s "
              f"({b200['wall_time_s']:.2f}s, avg_batch={b200['avg_batch_size']:.1f}); "
              f"per-leaf: {p200['sims_per_sec']:.1f} sims/s ({p200['wall_time_s']:.2f}s); "
              f"speedup {mt['speedup_200_batched_vs_per_leaf']:.2f}x", flush=True)
        print(f"[timing 800 sims] batched: {b800['sims_per_sec']:.1f} sims/s "
              f"({b800['wall_time_s']:.2f}s)", flush=True)
    else:
        result["memory_and_timing"] = {"skipped": "no CUDA device"}

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"evidence written: {EVIDENCE}", flush=True)

    ok = all(
        (
            eq["equal_within_float32"],
            s1["identical_visit_counts_and_policy"],
            det["identical"],
        )
    )
    if "memory_and_timing" in result and "memory_gate" in result["memory_and_timing"]:
        ok = ok and result["memory_and_timing"]["memory_gate"]["under_5GB"]
    print(f"RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
