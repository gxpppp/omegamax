"""Todo 10 acceptance demo: AGZ search details -- Dirichlet root noise,
temperature move selection, virtual loss.

Runs on the real b10c128 network from ``config/default.yaml`` (``eval`` mode,
CUDA when available) and demonstrates, with exact formulas:

* **Dirichlet root noise** -- on the empty 19x19 board with the network
  priors as ``P``, the blend ``P'(a) = (1 - eps) * P(a) + eps * eta(a)``,
  ``eta ~ Dir(alpha)`` (``alpha = 0.03``, ``eps = 0.25``), applied once at the
  root *before* the simulations: distribution stats (min/max/mean, sum),
  the count of children whose prior visibly changed, determinism under a
  seeded rng, and the fact that stored ``child.prior`` values are untouched
  (the blend lives in the transient ``root.noisy_prior`` override only);
* **Temperature selection** -- for a searched root: ``tau = 1.0`` policy
  matches the visit-count distribution exactly; ``tau -> 0`` concentrates all
  mass on the most-visited children; sampling at ``tau = 1`` reaches multiple
  children, sampling at ``tau = 0`` always returns the argmax (ties uniform);
* **Virtual loss** -- a hand-built UCB tree: a child claimed with
  ``virtual_loss = 3`` scores ``Q + c_puct * P * sqrt(N_root) / (1 + N + vl)``
  (exact numbers), its selection is depressed below an unclaimed sibling, and
  reverting the claim restores the original UCB. Plus: during a real search a
  recording evaluator sees ``node.virtual_loss == 3`` on the leaf it is
  evaluating, and after the search every ``virtual_loss`` is back to 0.

Exit code 0 iff every check passes. With ``--evidence <path>`` a UTF-8 JSON
report is written (in addition to the printed transcript, which is what the
plan's ``task-10-agz.txt`` evidence captures).

Usage::

    uv run python -m omigamax.cli.agz_demo
    uv run python -m omigamax.cli.agz_demo --sims 200 --seed 0 --evidence .omo/evidence/omigamax-go/task-10-agz-demo.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPS,
    DEFAULT_VIRTUAL_LOSS,
    apply_dirichlet_noise,
    make_root,
    most_visited_action,
    run_search,
    sample_action,
    select_child,
    temperature_policy,
)
from omigamax.network.model import create_model
from omigamax.rules import Board

DEFAULT_SIMS = 200
DEFAULT_SEED = 0
TAU_EARLY = 1.0
TAU_LATE = 0.0


class _UniformEvaluator:
    """Mock leaf evaluator: uniform prior over legal moves, zero value."""

    def __call__(self, node):
        size = node.board.size
        prior = np.ones(size * size + 1, dtype=np.float32)
        prior /= prior.sum()
        return prior, 0.0


class _RecordingEvaluator:
    """Uniform evaluator that also records the virtual loss it observes."""

    def __init__(self):
        self.seen_virtual_loss = []

    def __call__(self, node):
        self.seen_virtual_loss.append(node.virtual_loss)
        size = node.board.size
        prior = np.ones(size * size + 1, dtype=np.float32)
        prior /= prior.sum()
        return prior, 0.0


def _tree_nodes(root):
    """Count every node in the tree (root included) via BFS."""
    queue = list(root.children.values())
    count = 1
    while queue:
        node = queue.pop(0)
        count += 1
        queue.extend(node.children.values())
    return count


def _virtual_loss_clean(root) -> bool:
    """Every node's virtual_loss is back to 0 after the search."""
    queue = list(root.children.values())
    nodes = [root] + queue
    while queue:
        node = queue.pop(0)
        nodes.append(node)
        queue.extend(node.children.values())
    return all(n.virtual_loss == 0 for n in nodes)


def run_demo(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    size = int(cfg["board_size"])
    c_puct = float(cfg.get("c_puct", DEFAULT_C_PUCT))
    alpha = float(cfg.get("dirichlet_alpha", DEFAULT_DIRICHLET_ALPHA))
    eps = float(cfg.get("dirichlet_eps", DEFAULT_DIRICHLET_EPS))
    vl = int(cfg.get("virtual_loss", DEFAULT_VIRTUAL_LOSS))
    sims = int(args.sims)

    torch.manual_seed(args.seed)
    model = create_model(blocks=int(cfg["blocks"]), channels=int(cfg["channels"]), board_size=size)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    result: dict = {"config": {
        "board_size": size, "blocks": int(cfg["blocks"]), "channels": int(cfg["channels"]),
        "c_puct": c_puct, "dirichlet_alpha": alpha, "dirichlet_eps": eps,
        "virtual_loss": vl, "simulations": sims, "seed": args.seed,
        "temperature_threshold": int(cfg.get("temperature_threshold", 30)),
    }}

    # ------------------------------------------------------------------
    # 1) Dirichlet root noise: blend applied to the root's legal children
    # ------------------------------------------------------------------
    root = make_root(Board(size))
    run_search(root, model, simulations=sims, c_puct=c_puct)  # network priors P
    stored_priors = {a: c.prior for a, c in root.children.items()}
    rng = np.random.default_rng(args.seed)
    noisy = apply_dirichlet_noise(root, alpha, eps, rng=rng)
    noisy_values = list(noisy.values())
    noise = {
        "formula": "P'(a) = (1 - eps) * P(a) + eps * eta(a), eta ~ Dir(alpha)",
        "children": len(noisy),
        "sum": float(sum(noisy_values)),
        "min": float(min(noisy_values)),
        "max": float(max(noisy_values)),
        "mean": float(np.mean(noisy_values)),
        "changed_children": sum(
            not math.isclose(noisy[a], stored_priors[a], abs_tol=1e-9) for a in noisy
        ),
    }
    noise["in_unit_interval"] = all(0.0 <= v <= 1.0 for v in noisy_values)
    noise["stored_priors_untouched"] = all(
        math.isclose(root.children[a].prior, stored_priors[a], abs_tol=1e-9) for a in stored_priors
    )
    # determinism: same seed -> identical blend on a fresh identical root
    root2 = make_root(Board(size))
    run_search(root2, model, simulations=sims, c_puct=c_puct)
    noisy2 = apply_dirichlet_noise(root2, alpha, eps, rng=np.random.default_rng(args.seed))
    noise["deterministic_same_seed"] = noisy == noisy2
    result["dirichlet_noise"] = noise

    # ------------------------------------------------------------------
    # 2) Temperature selection on the (noisy) searched root
    # ------------------------------------------------------------------
    pi_tau1 = temperature_policy(root, TAU_EARLY)
    visit_pi = np.zeros(size * size + 1, dtype=np.float32)
    total_v = sum(c.visit_count for c in root.children.values())
    for a, c in root.children.items():
        visit_pi[a] = c.visit_count / total_v
    max_visits = max(c.visit_count for c in root.children.values())
    winners = [a for a, c in root.children.items() if c.visit_count == max_visits]
    pi_tau0 = temperature_policy(root, TAU_LATE)

    rng_tau1 = np.random.default_rng(args.seed + 1)
    sampled_tau1 = [sample_action(root, TAU_EARLY, rng=rng_tau1) for _ in range(500)]
    distinct_tau1 = len(set(sampled_tau1))
    rng_tau0 = np.random.default_rng(args.seed + 2)
    sampled_tau0 = [sample_action(root, TAU_LATE, rng=rng_tau0) for _ in range(100)]
    tau0_all_argmax = all(root.children[a].visit_count == max_visits for a in sampled_tau0)

    temp = {
        "tau_early": TAU_EARLY, "tau_late": TAU_LATE,
        "tau1_equals_visit_policy": bool(np.allclose(pi_tau1, visit_pi, atol=1e-6)),
        "tau1_sum": float(pi_tau1.sum()),
        "tau0_mass_on_winners": float(pi_tau0[winners].sum()),
        "tau0_sum": float(pi_tau0.sum()),
        "argmax_winners": winners,
        "sampled_tau1_distinct_actions": distinct_tau1,
        "sampled_tau0_all_argmax": tau0_all_argmax,
    }
    result["temperature"] = temp

    # ------------------------------------------------------------------
    # 3) Virtual loss: UCB depression (exact numbers) + search claim/release
    # ------------------------------------------------------------------
    # hand-built tree, root visits 10 (sqrt(10) shared), fresh children
    root3 = make_root(Board(3))
    root3.visit_count = 10
    c0 = _child(root3, prior=0.6, visits=0)
    c1 = _child(root3, prior=0.4, visits=0)
    root3.children = dict(sorted({0: c0, 1: c1}.items()))
    root3.legal_moves = (0, 1)
    sq = math.sqrt(10.0)
    ucb_base = [0.0 + c_puct * 0.6 * sq / 1.0, 0.0 + c_puct * 0.4 * sq / 1.0]
    selected_base = select_child(root3, c_puct)[0]
    c0.virtual_loss = vl
    ucb_claimed = 0.0 + c_puct * 0.6 * sq / (1.0 + 0 + vl)
    ucb_sibling = 0.0 + c_puct * 0.4 * sq / (1.0 + 0 + 0)
    selected_claimed = select_child(root3, c_puct)[0]
    c0.virtual_loss = 0
    selected_reverted = select_child(root3, c_puct)[0]

    # real search with a recording evaluator
    rec = _RecordingEvaluator()
    root4 = make_root(Board(3))
    run_search(root4, None, simulations=12, evaluator=rec, virtual_loss=vl)
    virtual = {
        "formula": "Q + c_puct * P * sqrt(N_root) / (1 + N_child + virtual_loss)",
        "ucb_no_claim": ucb_base,
        "selected_no_claim": selected_base,
        "ucb_claimed_child": ucb_claimed,
        "ucb_unclaimed_sibling": ucb_sibling,
        "selected_while_claimed": selected_claimed,
        "selection_flipped_by_virtual_loss": selected_base != selected_claimed,
        "selected_after_revert": selected_reverted,
        "restored_after_revert": selected_reverted == selected_base,
        "leaf_virtual_loss_during_eval": sorted(set(rec.seen_virtual_loss)),
        "clean_after_search": _virtual_loss_clean(root4),
        "search_nodes": _tree_nodes(root4),
    }
    result["virtual_loss"] = virtual

    result["passed"] = bool(
        noise["sum"] is not None and math.isclose(noise["sum"], 1.0, abs_tol=1e-6)
        and noise["in_unit_interval"]
        and noise["stored_priors_untouched"]
        and noise["deterministic_same_seed"]
        and temp["tau1_equals_visit_policy"]
        and math.isclose(temp["tau0_mass_on_winners"], 1.0, abs_tol=1e-6)
        and temp["sampled_tau1_distinct_actions"] > 1
        and temp["sampled_tau0_all_argmax"]
        and virtual["selection_flipped_by_virtual_loss"]
        and virtual["restored_after_revert"]
        and virtual["leaf_virtual_loss_during_eval"] == [vl]
        and virtual["clean_after_search"]
    )
    result["device"] = str(device)
    return result


def _child(root, prior, visits):
    """A fresh child of ``root`` with a fixed prior and visit statistics."""
    from omigamax.mcts.mcts import Node

    child = Node(board=root.board, prior=prior, parent=root)
    child.visit_count = visits
    child.value_sum = 0.0
    return child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="omigamax todo-10 AGZ-details demo (Dirichlet noise / temperature / virtual loss)."
    )
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                        help=f"search simulations for the demo root (default {DEFAULT_SIMS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the result JSON to this path (utf-8)")
    args = parser.parse_args(argv)

    result = run_demo(args)
    cfg = result["config"]

    print("=== omigamax AGZ-details demo (todo 10) ===", flush=True)
    print(
        f"config: board={cfg['board_size']} blocks={cfg['blocks']} channels={cfg['channels']} "
        f"c_puct={cfg['c_puct']} alpha={cfg['dirichlet_alpha']} eps={cfg['dirichlet_eps']} "
        f"virtual_loss={cfg['virtual_loss']} simulations={cfg['simulations']} "
        f"temperature_threshold={cfg['temperature_threshold']} seed={cfg['seed']}", flush=True
    )
    print(f"device: {result['device']}", flush=True)

    print("\n[1] Dirichlet root noise", flush=True)
    noise = result["dirichlet_noise"]
    print(f"  formula: {noise['formula']}", flush=True)
    print(
        f"  blend over {noise['children']} legal children: sum={noise['sum']:.6f} "
        f"min={noise['min']:.4f} max={noise['max']:.4f} mean={noise['mean']:.4f}", flush=True
    )
    print(
        f"  changed children: {noise['changed_children']}/{noise['children']} "
        f"(prior visibly shifted)", flush=True
    )
    print(
        f"  in [0,1]: {noise['in_unit_interval']} | stored child.prior untouched: "
        f"{noise['stored_priors_untouched']} | same-seed deterministic: "
        f"{noise['deterministic_same_seed']}", flush=True
    )

    print("\n[2] Temperature selection (tau=1 early, tau->0 later)", flush=True)
    temp = result["temperature"]
    print(
        f"  tau=1 policy == visit-count policy: {temp['tau1_equals_visit_policy']} "
        f"(sum {temp['tau1_sum']:.6f})", flush=True
    )
    print(
        f"  tau=0: all mass ({temp['tau0_mass_on_winners']:.6f}) on argmax winners "
        f"{temp['argmax_winners']} (sum {temp['tau0_sum']:.6f})", flush=True
    )
    print(
        f"  sampling 500x @tau=1 reached {temp['sampled_tau1_distinct_actions']} distinct "
        f"actions; 100x @tau=0 all argmax: {temp['sampled_tau0_all_argmax']}", flush=True
    )

    print("\n[3] Virtual loss", flush=True)
    virtual = result["virtual_loss"]
    print(f"  formula: {virtual['formula']}", flush=True)
    print(
        f"  hand-built tree (root visits 10): UCB no-claim {virtual['ucb_no_claim']} -> "
        f"selected {virtual['selected_no_claim']}; child claimed (vl={cfg['virtual_loss']}): "
        f"UCB {virtual['ucb_claimed_child']:.6f} vs sibling {virtual['ucb_unclaimed_sibling']:.6f} -> "
        f"selected {virtual['selected_while_claimed']} (flipped: "
        f"{virtual['selection_flipped_by_virtual_loss']}); after revert selected "
        f"{virtual['selected_after_revert']} (restored: {virtual['restored_after_revert']})", flush=True
    )
    print(
        f"  real search: leaf virtual_loss during eval = {virtual['leaf_virtual_loss_during_eval']} "
        f"({cfg['virtual_loss']} nodes searched); clean after search: "
        f"{virtual['clean_after_search']}", flush=True
    )

    print(
        f"\nRESULT: {'PASS' if result['passed'] else 'FAIL'} "
        f"(noise blend ok, tau1==visit policy, tau0 argmax, virtual-loss UCB "
        f"depression + revert ok)", flush=True
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
