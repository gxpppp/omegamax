"""P5: supervised-pretraining CLI on the human-game-record chunk corpus.

Trains the approved b20c256 policy-value net (via explicit
``--blocks/--channels/--board-size`` args, default 20/256/19 -- it does NOT
touch ``config/default.yaml``) on ``data/pretrain/chunk_*.npz`` SL samples
(17-plane 0/1 features, human-move index, game outcome +-1) with:

    L = policy_ce(logits, onehot(pi)) + value_mse(tanh_value, z)

SGD momentum 0.9, L2 1e-4, grad-norm clip 1.0, ``--lr`` 0.02 (deliberately
10x below the RL 0.2 -- see :mod:`omigamax.train.pretrain` for the warm-start
justification) with an optional piecewise-halving schedule. Checkpoints are
written in the existing ``omigamax/train/train.py`` format (arch + optimizer +
global_step + rng_state) to ``models/pretrain.pt``; ``--resume`` continues.

Per-step metrics (loss components, top-1 policy accuracy vs the human move,
lr) are JSONL-logged to ``logs/pretrain.jsonl``.

Usage:
    uv run python -m omigamax.cli.pretrain [--steps 200] [--batch-size 64]
        [--lr 0.02] [--blocks 20] [--channels 256] [--resume]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from omigamax.network.model import create_model
from omigamax.train.pretrain import (
    DEFAULT_GRAD_CLIP,
    DEFAULT_L2,
    DEFAULT_LR,
    DEFAULT_LR_STEPS,
    DEFAULT_MOMENTUM,
    PretrainChunks,
    make_pretrain_optimizer,
    pretrain_checkpoint_path,
    resume_from_checkpoint,
    run_pretrain,
    save_pretrain_checkpoint,
)
from omigamax.train.train import load_checkpoint

GIB = 1024 ** 3


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="supervised pretraining on human game records (P5)"
    )
    ap.add_argument("--data-dir", type=str, default="data/pretrain",
                    help="dir with the P3 chunk npz files (default data/pretrain)")
    ap.add_argument("--steps", type=int, default=200,
                    help="optimizer steps (default 200)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="batch size (default 64; b20c256 @64 = 2.15GB peak)")
    ap.add_argument("--blocks", type=int, default=20,
                    help="network residual blocks (default 20)")
    ap.add_argument("--channels", type=int, default=256,
                    help="network channels (default 256)")
    ap.add_argument("--board-size", type=int, default=19,
                    help="board edge (default 19)")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR,
                    help=f"learning rate (default {DEFAULT_LR}; SL warm-start "
                         f"rate, 10x below RL 0.2 -- see omigamax/train/pretrain.py)")
    ap.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2,
                    help="SGD weight decay (L2), default 1e-4")
    ap.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
                    help="gradient-norm clip (default 1.0, todo-14 evidence)")
    ap.add_argument("--lr-steps", type=int, nargs="*", default=list(DEFAULT_LR_STEPS),
                    help="halve lr past each boundary (default 50000; empty = constant)")
    ap.add_argument("--seed", type=int, default=0, help="master random seed")
    ap.add_argument("--device", type=str, default=None,
                    help="torch device (default: cuda if available)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="checkpoint path (default models/pretrain.pt)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the checkpoint (arch must match args)")
    ap.add_argument("--log-path", type=str, default="logs/pretrain.jsonl",
                    help="JSONL per-step metric log (default logs/pretrain.jsonl)")
    ap.add_argument("--log-every", type=int, default=10,
                    help="log a JSONL line every N steps (default 10)")
    return ap


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = args.checkpoint or str(pretrain_checkpoint_path())

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = create_model(args.blocks, args.channels, args.board_size).to(device)
    optimizer = make_pretrain_optimizer(model, lr=args.lr, momentum=args.momentum,
                                        l2=args.l2)
    rng = np.random.default_rng(args.seed)
    global_step = 0

    resumed = None
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"ERROR: --resume but no checkpoint at {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        arch = ckpt.get("arch", {})
        if (int(arch.get("blocks", -1)) != args.blocks
                or int(arch.get("channels", -1)) != args.channels
                or int(arch.get("board_size", -1)) != args.board_size):
            raise SystemExit(
                f"ERROR: checkpoint arch {arch} != requested "
                f"({args.blocks}/{args.channels}/{args.board_size})"
            )
        global_step = resume_from_checkpoint(ckpt, model, optimizer, rng)
        resumed = {"step": global_step, "arch": arch}

    with PretrainChunks(args.data_dir) as chunks:
        report = chunks.validate()
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        metrics, final_step, rng = run_pretrain(
            model, optimizer, chunks, steps=args.steps, rng=rng,
            seed=args.seed, global_step=global_step, batch_size=args.batch_size,
            device=device, grad_clip=args.grad_clip, lr_base=args.lr,
            lr_steps=tuple(args.lr_steps) if args.lr_steps else (),
            log_path=args.log_path, log_every=args.log_every,
        )
        wall = time.perf_counter() - t0
        peak_gb = (torch.cuda.max_memory_allocated() / GIB
                   if torch.cuda.is_available() else 0.0)

        config = {
            "blocks": args.blocks, "channels": args.channels,
            "board_size": args.board_size, "lr": args.lr,
            "momentum": args.momentum, "l2": args.l2,
            "grad_clip": args.grad_clip, "lr_steps": list(args.lr_steps),
            "batch_size": args.batch_size, "seed": args.seed,
        }
        save_pretrain_checkpoint(
            ckpt_path, model, optimizer, global_step=final_step, rng=rng,
            config=config,
            extra={"total_positions": report["total_positions"],
                   "resumed": resumed},
        )

    ms_per_step = wall / max(1, len(metrics)) * 1e3
    first = metrics[0] if metrics else {}
    last = metrics[-1] if metrics else {}
    print("=" * 68)
    print("P5 supervised pretraining smoke")
    print(f"  device        : {device}"
          f"{f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}")
    print(f"  arch          : blocks={args.blocks} channels={args.channels} "
          f"board={args.board_size} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  data          : {report['total_positions']:,} positions "
          f"across {chunks.num_chunks} chunks")
    print(f"  steps         : {len(metrics):,} (global_step {final_step:,}"
          f"{' after resume from ' + str(resumed['step']) if resumed else ''})")
    print(f"  batch size    : {args.batch_size}   lr {args.lr}  "
          f"grad_clip {args.grad_clip}")
    if first:
        print(f"  loss first    : total {first['loss_total']:.4f} "
              f"policy {first['loss_policy']:.4f} value {first['loss_value']:.4f} "
              f"top1 {first['acc_top1']*100:.2f}%")
        print(f"  loss last     : total {last['loss_total']:.4f} "
              f"policy {last['loss_policy']:.4f} value {last['loss_value']:.4f} "
              f"top1 {last['acc_top1']*100:.2f}%")
        print(f"  loss ratio    : {last['loss_total']/max(first['loss_total'], 1e-9):.3f}")
    print(f"  speed         : {ms_per_step:.1f} ms/step "
          f"({wall:.1f}s / {len(metrics)} steps)")
    if device.type == "cuda":
        print(f"  peak gpu mem  : {peak_gb:.3f} GB "
              f"(budget 5.5 GB, {'OK' if peak_gb <= 5.5 else 'OVER'})")
    print(f"  checkpoint    : {ckpt_path} (global_step {final_step})")
    print(f"  jsonl log     : {args.log_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
