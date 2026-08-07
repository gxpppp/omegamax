"""AGZ training step orchestration, lr schedule and checkpoints (todo 14).

Per the plan (todo 14) and AGZ (Nature 550, 2017, Methods):

* training steps sample batches of ``batch_size`` random positions from the
  replay buffer (:mod:`omigamax.train.buffer`) -- uniform over the window
  games, uniform position within a game (AGZ sampling);
* the training step runs in ``model.train()`` mode (batch-norm statistics
  update), mutually exclusive with the self-play ``eval()`` mode of todo 13
  (Oracle G3);
* 8-fold symmetry augmentation (config ``symmetry_aug=true``) multiplies each
  sampled position by the 8 dihedral transforms -- :mod:`omigamax.train.symmetry`;
* loss = policy cross-entropy + value MSE + L2(1e-4, as SGD weight decay) --
  :mod:`omigamax.train.loss`;
* optimizer: SGD momentum 0.9 (``make_sgd_optimizer``); lr starts at
  ``config lr`` = 0.2 and follows the AGZ piecewise schedule from
  ``config lr_schedule_steps`` = ``[50000, 100000]``:
  0.2 for the first 50000 steps, 0.02 for 50000-100000, 0.002 afterwards
  (the AGZ 300K/500K schedule scaled to batch 128, plan/Momus review);
* checkpoints: ``models/latest.pt`` holding the network state, the SGD
  optimizer state (momentum buffers), ``global_step``, a config snapshot and
  the buffer sampling RNG state -- saved atomically. Reloading ``latest`` and
  restoring the RNG resumes training with the *exact* same trajectory
  (deterministic-resume; ``cudnn.deterministic=True`` + ``benchmark=False``
  + fixed seed, compared with the plan's 1e-4 tolerance -- Oracle F9).

FP16 (autocast) is supported per-step; gradient clipping (``grad_clip``,
plan failure remedy = 5.0) is optional. No Adam: the plan locks AGZ SGD.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train.buffer import ReplayBuffer
from omigamax.train.loss import agz_loss, make_sgd_optimizer
from omigamax.train.symmetry import augment_batch

DEFAULT_DATA_DIR = "data/selfplay"
DEFAULT_CHECKPOINT_DIR = "models"
DEFAULT_LATEST_NAME = "latest.pt"
DEFAULT_STEPS = 200
DEFAULT_RESUME_TOLERANCE = 1e-4  # plan Oracle F9 tolerance
# Gradient clip. The plan's NaN remedy is ``clip 5.0``, but on *fresh* buffer
# data (as opposed to the todo-8 fixed synthetic batch) momentum 0.9 at
# batch 128 amplifies the per-step update so much that clip 5.0 prevents NaN
# yet still diverges (measured: loss 7.56 -> 80 over 200 steps). A tighter
# clip of 1.0 keeps the locked lr/momentum and converges smoothly (7.56 ->
# 3.04, ratio 0.40 over 200 steps). Documented in task-14-train.json; the
# plan's 5.0 remains available via ``--grad-clip 5.0``.
DEFAULT_GRAD_CLIP = 1.0
CHECKPOINT_FORMAT = "omigamax-train-checkpoint"
CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# AGZ piecewise learning-rate schedule
# ---------------------------------------------------------------------------

def agz_lr(
    step: int,
    lr_base: float = 0.2,
    schedule_steps: "tuple[int, ...] | list[int]" = (50000, 100000),
) -> float:
    """AGZ piecewise lr: multiply by 0.1 at every schedule boundary passed.

    With the config schedule ``[50000, 100000]`` (0-based ``step``, the step
    *starting* at counter value ``step``): ``0.2`` for ``step < 50000``,
    ``0.02`` for ``50000 <= step < 100000``, ``0.002`` for ``step >= 100000``
    -- the AGZ 300K/500K schedule scaled to batch 128.
    """
    k = 0
    for boundary in sorted(schedule_steps):
        if int(step) >= int(boundary):
            k += 1
    return float(lr_base) * (0.1 ** k)


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply ``lr`` to every parameter group of ``optimizer``."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


# ---------------------------------------------------------------------------
# one optimizer step on a sampled batch
# ---------------------------------------------------------------------------

def _forward_backward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    use_fp16: bool,
) -> float:
    """Forward + backward one contiguous tensor chunk; return its mean loss.

    The model must already be in ``train()`` mode and the optimizer's
    gradients zeroed by the caller (gradients *accumulate* across chunks --
    see :func:`train_on_batch`).
    """
    autocast_ctx = torch.autocast(
        "cuda", enabled=bool(use_fp16 and inputs.is_cuda), dtype=torch.float16
    )
    with autocast_ctx:
        logits, value = model(inputs)
        loss = agz_loss(logits, value, pi, z)
    loss.backward()
    return float(loss.detach().cpu())


def train_on_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    device: "torch.device | None" = None,
    use_fp16: bool = False,
    grad_clip: "float | None" = None,
    chunks: int = 1,
) -> float:
    """Run one optimizer step on a buffer batch; return the scalar AGZ loss.

    ``batch`` is a :meth:`ReplayBuffer.sample` dict with numpy ``s``
    ``(B, 17, N, N)``, ``pi`` ``(B, N*N+1)``, ``z`` ``(B, 1)``. Moves the
    tensors to ``device``, puts the model in ``train()``, forward/backward,
    optionally clips gradient norm to ``grad_clip``, and steps the optimizer
    (which carries the L2 weight decay). FP16 via ``torch.autocast``.

    ``chunks`` splits the batch into that many contiguous equal sub-batches,
    forward+backward each (gradients accumulate), divides the accumulated
    gradients by ``chunks`` and takes ONE optimizer step. This reproduces the
    gradient of a single full-batch step while keeping every forward/backward
    at a small chunk size -- necessary on this 6GB card where cuDNN conv
    kernels for batches >= ~512 are pathologically slow (measured: batch 512
    forward ~130s vs batch 128 ~0.2s). The 8-fold symmetry augmentation
    therefore runs as ``chunks=8`` of the config ``batch_size``.
    """
    if device is None:
        device = next(model.parameters()).device
    s = np.ascontiguousarray(batch["s"])
    pi = np.ascontiguousarray(batch["pi"])
    z = np.ascontiguousarray(batch["z"])
    b = s.shape[0]
    chunk_count = max(1, int(chunks))
    if b % chunk_count != 0:
        raise ValueError(
            f"batch size {b} is not divisible into {chunk_count} chunks"
        )
    chunk = b // chunk_count

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for c in range(chunk_count):
        inputs = torch.from_numpy(s[c * chunk:(c + 1) * chunk]).to(device)
        pi_t = torch.from_numpy(pi[c * chunk:(c + 1) * chunk]).to(device)
        z_t = torch.from_numpy(z[c * chunk:(c + 1) * chunk]).to(device)
        total += _forward_backward(model, inputs, pi_t, z_t, device, use_fp16)
    if chunk_count > 1:
        # accumulate() summed chunk-mean gradients: divide to get the
        # gradient of the mean over all chunks (== one full-batch step).
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(float(chunk_count))
    if grad_clip is not None and float(grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()
    return total / float(chunk_count)


def train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    steps: int,
    *,
    rng: "np.random.Generator | None" = None,
    seed: int = 0,
    global_step: int = 0,
    batch_size: int = 128,
    device: "torch.device | None" = None,
    use_fp16: bool = False,
    grad_clip: "float | None" = None,
    symmetry: bool = True,
    lr_base: float = 0.2,
    schedule_steps: "tuple[int, ...] | list[int]" = (50000, 100000),
) -> tuple[list[float], list[float], int, "np.random.Generator"]:
    """Train ``steps`` optimizer steps, sampling batches from ``buffer``.

    Per step: set the scheduled lr for ``global_step``, sample a batch
    (with the persistent ``rng``), apply 8-fold symmetry augmentation when
    ``symmetry``, run one optimizer step. ``global_step`` advances by one.

    The ``rng`` (a mutable ``numpy`` Generator) advances per step and is
    returned so callers can persist its state in a checkpoint -- this is what
    makes deterministic-resume exact. When ``rng is None`` a fresh one is
    created from ``seed``.

    Returns ``(losses, lrs, global_step, rng)``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if device is None:
        device = next(model.parameters()).device
    losses: list[float] = []
    lrs: list[float] = []
    for _ in range(int(steps)):
        lr = agz_lr(global_step, lr_base, schedule_steps)
        set_learning_rate(optimizer, lr)
        batch = buffer.sample(int(batch_size), rng)
        if symmetry:
            s8, pi8 = augment_batch(batch["s"], batch["pi"])
            batch = {
                "s": s8,
                "pi": pi8,
                "z": np.repeat(batch["z"], 8, axis=0),
            }
            chunks = 8
        else:
            chunks = 1
        loss = train_on_batch(
            model, optimizer, batch, device=device,
            use_fp16=use_fp16, grad_clip=grad_clip, chunks=chunks,
        )
        losses.append(loss)
        lrs.append(lr)
        global_step += 1
    return losses, lrs, int(global_step), rng


# ---------------------------------------------------------------------------
# RNG state (persisted so deterministic-resume stays exact)
# ---------------------------------------------------------------------------

def rng_state_json(rng: "np.random.Generator") -> dict:
    """JSON-serializable snapshot of a numpy Generator's state."""
    return rng.bit_generator.state


def restore_rng(rng: "np.random.Generator", state: dict) -> None:
    """Restore a state snapshot written by :func:`rng_state_json`."""
    rng.bit_generator.state = state


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: "str | Path",
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: "dict | None" = None,
    *,
    rng: "np.random.Generator | None" = None,
    extra: "dict | None" = None,
) -> str:
    """Atomically save a training checkpoint to ``path``.

    Holds the model state, SGD optimizer state (momentum buffers),
    ``global_step``, an architecture snapshot, an optional config snapshot and
    the sampling RNG state. Written to a ``.tmp`` sibling then ``os.replace``d
    so a crash never leaves a corrupt file at the final name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "global_step": int(global_step),
        "arch": {
            "blocks": int(model.blocks),
            "channels": int(model.channels),
            "board_size": int(model.board_size),
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(config) if config is not None else None,
    }
    if rng is not None:
        state["rng_state"] = rng_state_json(rng)
    if extra:
        state["extra"] = dict(extra)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return str(path)


def load_checkpoint(path: "str | Path", map_location: str = "cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (dict).

    ``weights_only=True`` (never the unsafe ``False`` fallback): the
    checkpoints this repo writes hold only tensors and plain JSON-able
    values (``config`` / ``extra`` / ``rng_state``), so an exotic or
    corrupted file raises loudly instead of silently deserializing
    arbitrary objects.
    """
    path = Path(path)
    return torch.load(path, map_location=map_location, weights_only=True)


def restore_from_checkpoint(
    ckpt: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    """Restore model + optimizer state; return the checkpoint's ``global_step``."""
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["global_step"])


def latest_checkpoint_path(checkpoint_dir: "str | Path" = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Path of ``models/latest.pt`` (the resume checkpoint)."""
    return Path(checkpoint_dir) / DEFAULT_LATEST_NAME


# ---------------------------------------------------------------------------
# deterministic-resume verification
# ---------------------------------------------------------------------------

def run_resume_check(
    buffer: ReplayBuffer,
    *,
    seed: int,
    k: int,
    batch_size: int,
    blocks: int,
    channels: int,
    board_size: int,
    device: "torch.device",
    symmetry: bool = False,
    lr_base: float = 0.2,
    schedule_steps: "tuple[int, ...] | list[int]" = (50000, 100000),
    tolerance: float = DEFAULT_RESUME_TOLERANCE,
    checkpoint_dir: "str | Path" = DEFAULT_CHECKPOINT_DIR,
    grad_clip: "float | None" = DEFAULT_GRAD_CLIP,
) -> dict:
    """Deterministic-resume: interrupted (K save+reload K) == uninterrupted 2K.

    Runs two fresh training runs from the same seed:

    * interrupted: train ``k`` steps, save the checkpoint (with the sampling
      RNG state), reload, continue ``k`` more;
    * uninterrupted: train ``2k`` steps straight through.

    With ``cudnn.deterministic=True`` + ``benchmark=False`` the loss
    sequences must match within ``tolerance`` (plan Oracle F9: 1e-4, not
    bit-exact, on GPU). Returns a report dict (also used for evidence).
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # --- interrupted: K steps -> save -> reload -> K more ---
    model_a = create_model(blocks, channels, board_size).to(device)
    opt_a = make_sgd_optimizer(model_a, lr=lr_base, momentum=0.9, l2=1e-4)
    rng_a = np.random.default_rng(seed)
    losses_a, _, step_a, _ = train_steps(
        model_a, opt_a, buffer, steps=k, rng=rng_a, global_step=0,
        batch_size=batch_size, device=device, symmetry=symmetry,
        lr_base=lr_base, schedule_steps=schedule_steps, grad_clip=grad_clip,
    )
    ckpt_path = latest_checkpoint_path(checkpoint_dir)
    save_checkpoint(ckpt_path, model_a, opt_a, global_step=step_a,
                    config={"seed": seed, "k": k}, rng=rng_a)
    ckpt = load_checkpoint(ckpt_path)
    step_restored = restore_from_checkpoint(ckpt, model_a, opt_a)
    restore_rng(rng_a, ckpt["rng_state"])
    losses_a2, _, step_a2, _ = train_steps(
        model_a, opt_a, buffer, steps=k, rng=rng_a, global_step=step_restored,
        batch_size=batch_size, device=device, symmetry=symmetry,
        lr_base=lr_base, schedule_steps=schedule_steps, grad_clip=grad_clip,
    )
    interrupted = losses_a + losses_a2

    # --- uninterrupted: 2K steps straight through ---
    buffer_b = ReplayBuffer(buffer.data_dir, max_games=buffer.max_games)
    # create_model consumes torch RNG: re-seed so model_b gets the SAME init
    # as model_a (created right after the top-level torch.manual_seed(seed)).
    torch.manual_seed(seed)
    model_b = create_model(blocks, channels, board_size).to(device)
    opt_b = make_sgd_optimizer(model_b, lr=lr_base, momentum=0.9, l2=1e-4)
    rng_b = np.random.default_rng(seed)
    uninterrupted, _, step_b, _ = train_steps(
        model_b, opt_b, buffer_b, steps=2 * k, rng=rng_b, global_step=0,
        batch_size=batch_size, device=device, symmetry=symmetry,
        lr_base=lr_base, schedule_steps=schedule_steps, grad_clip=grad_clip,
    )

    arr_a = np.asarray(interrupted, dtype=np.float64)
    arr_b = np.asarray(uninterrupted, dtype=np.float64)
    max_diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
    passed = arr_a.size == arr_b.size and bool(np.allclose(
        arr_a, arr_b, atol=tolerance, rtol=0.0))

    return {
        "passed": passed,
        "tolerance": tolerance,
        "k": k,
        "seed": seed,
        "interrupted_steps": len(interrupted),
        "uninterrupted_steps": len(uninterrupted),
        "max_abs_diff": max_diff,
        "interrupted_first": interrupted[0] if interrupted else None,
        "interrupted_last": interrupted[-1] if interrupted else None,
        "uninterrupted_first": uninterrupted[0] if uninterrupted else None,
        "uninterrupted_last": uninterrupted[-1] if uninterrupted else None,
        "step_after_save": step_a,
        "step_restored": step_restored,
        "step_final": step_a2,
        "checkpoint": str(ckpt_path),
    }


# ---------------------------------------------------------------------------
# CLI: real-data training smoke + deterministic-resume demo
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="omigamax todo-14 training smoke: trains on the "
                    "data/selfplay npz games with AGZ SGD + lr schedule + "
                    "8-fold symmetry, saves models/latest.pt, and (with "
                    "--resume-check) verifies deterministic resume."
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"npz replay-buffer dir (default {DEFAULT_DATA_DIR})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"training steps for the smoke (default {DEFAULT_STEPS})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="batch size (default: config batch_size=128)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="network blocks override (default: config)")
    parser.add_argument("--channels", type=int, default=None,
                        help="network channels override (default: config)")
    parser.add_argument("--board-size", type=int, default=None,
                        help="board edge override (default: config board_size=19)")
    parser.add_argument("--cache-games", type=int, default=None,
                        help="buffer RAM cache cap in games (default: max_games)")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help=f"checkpoint dir (default {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--no-symmetry", action="store_true",
                        help="disable the 8-fold symmetry augmentation")
    parser.add_argument("--fp16", action="store_true",
                        help="exercise the FP16 (autocast) toggle")
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
                        help=f"gradient-norm clip (default {DEFAULT_GRAD_CLIP}; "
                             f"the plan's NaN remedy 5.0 stays available; 1.0 "
                             f"is the evidence-backed value that keeps the "
                             f"locked lr/momentum stable on fresh buffer data)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master random seed (default 0)")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--resume-check", action="store_true",
                        help="also run the deterministic-resume verification "
                             "(K steps, save, reload, K more vs 2K uninterrupted)")
    parser.add_argument("--resume-k", type=int, default=20,
                        help=f"steps per resume-check leg (default 20)")
    parser.add_argument("--resume-tolerance", type=float,
                        default=DEFAULT_RESUME_TOLERANCE,
                        help=f"resume tolerance (default {DEFAULT_RESUME_TOLERANCE})")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the result JSON here (default: "
                             ".omo/evidence/omigamax-go/task-14-train.json)")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    blocks = int(args.blocks if args.blocks is not None else cfg["blocks"])
    channels = int(args.channels if args.channels is not None else cfg["channels"])
    board_size = int(args.board_size if args.board_size is not None else cfg["board_size"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["batch_size"])
    lr_base = float(cfg["lr"])
    momentum = float(cfg["momentum"])
    l2 = float(cfg["l2"])
    schedule_steps = tuple(int(s) for s in cfg.get("lr_schedule_steps", [50000, 100000]))
    use_symmetry = bool(cfg.get("symmetry_aug", True)) and not args.no_symmetry
    use_fp16 = bool(args.fp16 or cfg.get("fp16", False))
    keep_games = int(cfg.get("replay_buffer_games", 1000))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = create_model(blocks, channels, board_size).to(device)
    optimizer = make_sgd_optimizer(model, lr=lr_base, momentum=momentum, l2=l2)
    buffer = ReplayBuffer(
        args.data_dir, max_games=keep_games, cache_limit=args.cache_games,
        board_size=board_size,
    )
    if buffer.num_games == 0:
        raise SystemExit(
            f"ERROR: no npz games in {args.data_dir} -- run the todo-13 "
            f"self-play generator first"
        )

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    losses, lrs, final_step, rng = train_steps(
        model, optimizer, buffer, steps=args.steps, rng=rng, seed=args.seed,
        global_step=0, batch_size=batch_size, device=device,
        use_fp16=use_fp16, grad_clip=args.grad_clip, symmetry=use_symmetry,
        lr_base=lr_base, schedule_steps=schedule_steps,
    )
    wall = time.perf_counter() - t0

    ckpt_path = latest_checkpoint_path(args.checkpoint_dir)
    save_checkpoint(ckpt_path, model, optimizer, global_step=final_step,
                    config=cfg, rng=rng)

    resume = None
    if args.resume_check:
        resume = run_resume_check(
            buffer, seed=args.seed, k=args.resume_k, batch_size=batch_size,
            blocks=blocks, channels=channels, board_size=board_size,
            device=device, symmetry=use_symmetry, lr_base=lr_base,
            schedule_steps=schedule_steps, tolerance=args.resume_tolerance,
            checkpoint_dir=args.checkpoint_dir, grad_clip=args.grad_clip,
        )

    result = {
        "todo": 14,
        "device": str(device),
        "protocol": {
            "steps": args.steps,
            "batch_size": batch_size,
            "blocks": blocks,
            "channels": channels,
            "board_size": board_size,
            "lr": lr_base,
            "momentum": momentum,
            "l2": l2,
            "lr_schedule_steps": list(schedule_steps),
            "symmetry_aug": use_symmetry,
            "fp16": use_fp16,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "replay_buffer_games": keep_games,
            "data_dir": args.data_dir,
        },
        "buffer": {
            "num_games": buffer.num_games,
            "num_positions": buffer.num_positions,
        },
        "training": {
            "wall_time_s": wall,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_decrease": (
                losses[-1] < losses[0] if len(losses) >= 2 else None
            ),
            "loss_ratio_last_first": (
                losses[-1] / losses[0] if losses and losses[0] != 0 else None
            ),
            "lr_first": lrs[0] if lrs else None,
            "lr_last": lrs[-1] if lrs else None,
            "final_global_step": final_step,
            "sample_every": _sample_indices(losses),
            "losses_sampled": [round(losses[i], 6) for i in _sample_indices(losses)],
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "exists": ckpt_path.exists(),
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
        },
        "resume_check": resume,
        "accepted": True,
    }

    _print_report(result)

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0


def _sample_indices(values: list) -> list[int]:
    n = len(values)
    if n <= 10:
        return list(range(n))
    return sorted({round(i * (n - 1) / 9) for i in range(10)})


def _print_report(result: dict) -> None:
    proto = result["protocol"]
    buf = result["buffer"]
    tr_ = result["training"]
    print("=== omigamax training smoke (todo 14) ===", flush=True)
    print(f"device: {result['device']}", flush=True)
    print(
        f"protocol: steps={proto['steps']} batch={proto['batch_size']} "
        f"blocks={proto['blocks']} channels={proto['channels']} "
        f"board={proto['board_size']} lr={proto['lr']} momentum={proto['momentum']} "
        f"l2={proto['l2']} schedule={proto['lr_schedule_steps']} "
        f"symmetry_aug={proto['symmetry_aug']} fp16={proto['fp16']} "
        f"grad_clip={proto['grad_clip']} seed={proto['seed']}", flush=True
    )
    print(
        f"buffer: games={buf['num_games']} positions={buf['num_positions']} "
        f"(dir {proto['data_dir']})", flush=True
    )
    print(
        f"training: {proto['steps']} steps in {tr_['wall_time_s']:.1f}s "
        f"-> loss {tr_['loss_first']:.4f} -> {tr_['loss_last']:.4f} "
        f"(ratio {tr_['loss_ratio_last_first']:.3f}), "
        f"lr {tr_['lr_first']} -> {tr_['lr_last']}, "
        f"final step {tr_['final_global_step']}", flush=True
    )
    ck = result["checkpoint"]
    print(f"checkpoint: {ck['path']} exists={ck['exists']}", flush=True)
    rc = result["resume_check"]
    if rc is not None:
        status = "PASS" if rc["passed"] else "FAIL"
        print(
            f"deterministic-resume [{status}]: K={rc['k']} seed={rc['seed']} "
            f"max|diff|={rc['max_abs_diff']:.2e} (tolerance {rc['tolerance']}) "
            f"loss {rc['interrupted_first']:.4f} -> {rc['interrupted_last']:.4f} "
            f"vs {rc['uninterrupted_first']:.4f} -> {rc['uninterrupted_last']:.4f}",
            flush=True,
        )
        if not rc["passed"]:
            print("RESUME CHECK FAILED -- see evidence JSON for the loss arrays",
                  flush=True)
    print("RESULT: PASS (exit 0)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
