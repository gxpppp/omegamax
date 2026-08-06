"""Todo 8: network smoke training + 6GB memory validation.

Runs ``--steps`` SGD optimizer steps (default 200, per the plan) on synthetic
random batch data using the locked AGZ hyper-parameters from
``config/default.yaml`` (b10c128, ``batch_size=128``, ``lr=0.2``,
``momentum=0.9``, ``l2=1e-4``, ``fp16=false``) and asserts two things:

1. the AGZ loss (policy cross-entropy + value MSE, with L2 applied via SGD
   weight decay -- see :mod:`omigamax.train.loss`) is finite and *measurably*
   decreasing (``loss_last < loss_first * 0.8`` -- the plan's
   "loss 有限且下降" gate made quantitative);
2. the peak GPU memory during batch-128 forward+backward training stays within
   the 6GB card's budget: ``torch.cuda.max_memory_allocated()``
   <= ``--max-mem-gb`` (default 5.5 GB -- the plan's "peak <= 5.5GB @ batch
   128" gate; 0.5 GB head-room for the OS/driver/cuDNN workspace).

``--fp16`` runs the same smoke under ``torch.autocast`` (the plan's
"FP16（autocast）开关冒烟"); the locked config default is fp16 OFF.

Data source (per plan "随机数据", cross-entropy + regression smoke): synthetic
-- inputs ``N(0, 1)`` over ``(B, 17, N, N)``, policy targets a random
one-hot label over the ``N**2 + 1`` classes, value targets uniform in
``{-1, +1}``. No real training data is produced (that is todo 13/14).

On a budget breach or OOM the script prints diagnostics
(``torch.cuda.memory_summary``) and exits 1 -- it NEVER edits
``config/default.yaml`` (run guardrail: no config changes); the executor
stops and reports instead of silently shrinking the batch.

Exit 0 iff loss finite + decreasing AND peak memory within budget. Acceptable
output includes ``peak_GB`` and ``final loss`` (plan acceptance line).

Usage::

    uv run python -m omigamax.cli.smoke_net
    uv run python -m omigamax.cli.smoke_net --fp16 --steps 200 --evidence .omo/evidence/omigamax-go/task-8-smoke.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train.loss import make_sgd_optimizer, train_step, weight_l2

DEFAULT_STEPS = 200
DEFAULT_MAX_MEM_GB = 5.5
DEFAULT_SEED = 0
# 1 GiB = 1024**3 bytes, consistent with torch.cuda.max_memory_allocated.
GIB = 1024**3


def _sample_losses(losses: list[float], steps: int, samples: int = 4) -> list[int]:
    """Pick ``samples`` evenly spaced step indices (including first and last)."""
    if steps <= samples:
        return list(range(steps))
    idx = sorted({round(i * (steps - 1) / (samples - 1)) for i in range(samples)})
    return idx


def run_smoke(args: argparse.Namespace) -> dict:
    """Execute the smoke; return a result dict (also used for the evidence)."""
    cfg = load_config(args.config)
    batch = args.batch if args.batch is not None else int(cfg["batch_size"])
    blocks, channels, board = int(cfg["blocks"]), int(cfg["channels"]), int(cfg["board_size"])
    lr, momentum, l2 = float(cfg["lr"]), float(cfg["momentum"]), float(cfg["l2"])
    fp16 = bool(args.fp16 or cfg.get("fp16", False))
    steps = int(args.steps)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- the todo-8 smoke is a GPU run "
            "(peak-memory gate requires a CUDA device)."
        )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)

    model = create_model(blocks, channels, board).to(device)
    # L2 (config l2=1e-4) is applied as SGD weight decay -- see loss.py.
    optimizer = make_sgd_optimizer(model, lr, momentum, l2)
    n_logits = board * board + 1

    # Synthetic random batch (plan: "随机数据"; CE + regression smoke):
    # random inputs, a random one-hot policy label per sample, random +-1
    # value outcome.
    inputs = torch.randn(batch, 17, board, board, device=device)
    target_idx = torch.randint(0, n_logits, (batch,), device=device)
    pi = torch.zeros(batch, n_logits, device=device)
    pi[torch.arange(batch), target_idx] = 1.0
    z = torch.randint(0, 2, (batch, 1), device=device).float() * 2.0 - 1.0

    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        losses.append(train_step(model, optimizer, inputs, pi, z, use_fp16=fp16))
    wall_time_s = time.perf_counter() - t0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gb = peak_bytes / GIB
    # Analytic magnitude of the plan's L2 term (monitor only; regularization
    # itself is applied via weight decay in the optimizer).
    l2_magnitude = float(weight_l2(model) * l2)

    first, last = losses[0], losses[-1]
    all_finite = all(torch.isfinite(torch.tensor(l)).item() for l in losses)
    loss_decreased = bool(all_finite and last < first * 0.8)
    peak_within_budget = bool(peak_gb <= args.max_mem_gb)
    passed = bool(loss_decreased and peak_within_budget)

    return {
        "passed": passed,
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "steps": steps,
        "hyperparams": {
            "blocks": blocks,
            "channels": channels,
            "board_size": board,
            "batch_size": batch,
            "lr": lr,
            "momentum": momentum,
            "l2": l2,
            "fp16": fp16,
        },
        "loss": {
            "first": first,
            "last": last,
            "ratio_last_over_first": last / first if first != 0 else None,
            "all_finite": all_finite,
            "loss_decreased": loss_decreased,
            "l2_magnitude": l2_magnitude,
            "sampled": {str(i): losses[i] for i in _sample_losses(losses, steps)},
            "curve": losses,
        },
        "memory": {
            "peak_bytes": peak_bytes,
            "peak_GB": peak_gb,
            "limit_GB": args.max_mem_gb,
            "within_budget": peak_within_budget,
        },
        "wall_time_s": wall_time_s,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="omigamax todo-8 network smoke training + memory validation."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"optimizer steps (default {DEFAULT_STEPS})")
    parser.add_argument("--batch", type=int, default=None,
                        help="batch size (default: config batch_size=128)")
    parser.add_argument("--max-mem-gb", type=float, default=DEFAULT_MAX_MEM_GB,
                        help=f"peak-memory budget in GB (default {DEFAULT_MAX_MEM_GB})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--fp16", action="store_true",
                        help="run the smoke under torch.autocast (FP16 toggle)")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the result JSON to this path (utf-8)")
    args = parser.parse_args(argv)

    try:
        result = run_smoke(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", flush=True)
        if torch.cuda.is_available():
            try:
                print(torch.cuda.memory_summary(), flush=True)
            except Exception:  # pragma: no cover - diagnostics only
                pass
        return 1
    except torch.cuda.OutOfMemoryError:
        # Plan fallback decision: do NOT shrink the batch / edit config;
        # report and stop (run guardrail).
        print("ERROR: CUDA out of memory at the requested batch size. "
              "Stopping and reporting (no config change made).", flush=True)
        print(torch.cuda.memory_summary(), flush=True)
        return 1

    hp = result["hyperparams"]
    loss = result["loss"]
    mem = result["memory"]
    print("=== omigamax smoke_net (todo 8) ===", flush=True)
    print(f"device: {result['device']}", flush=True)
    print(f"config: blocks={hp['blocks']} channels={hp['channels']} "
          f"board={hp['board_size']} batch={hp['batch_size']} lr={hp['lr']} "
          f"momentum={hp['momentum']} l2={hp['l2']} fp16={hp['fp16']} "
          f"steps={result['steps']} seed={result['seed']}", flush=True)
    print(f"loss_first={loss['first']:.6f} loss_last={loss['last']:.6f} "
          f"ratio={loss['ratio_last_over_first']:.4f} "
          f"(final < initial*0.8: {loss['loss_decreased']})", flush=True)
    sampled = loss["sampled"]
    print("sampled losses: " + " | ".join(
        f"step {i}: {v:.6f}" for i, v in sampled.items()
    ), flush=True)
    print(f"l2_magnitude={loss['l2_magnitude']:.6f} (l2*||W||^2 monitor; "
          f"regularization via SGD weight_decay)", flush=True)
    print(f"peak_GB={mem['peak_GB']:.4f} (limit {mem['limit_GB']} GB, "
          f"within budget: {mem['within_budget']})", flush=True)
    print(f"wall_time_s={result['wall_time_s']:.2f}", flush=True)
    print(f"all losses finite: {loss['all_finite']}", flush=True)
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'} "
          f"(loss decreased={loss['loss_decreased']}, "
          f"memory within budget={mem['within_budget']})", flush=True)

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
