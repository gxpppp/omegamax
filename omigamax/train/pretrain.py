"""Supervised pretraining on human game records (P5).

Reads the P3 chunk corpus (``data/pretrain/chunk_*.npz``: ``s``
``(N,17,19,19)`` uint8 0/1 feature planes, ``pi`` ``(N,)`` uint16 move index
0..360 + 361 = pass, ``z`` ``(N,)`` int8 +-1 from the mover's perspective),
and trains the approved b20c256 policy-value net with supervised learning:

    L = policy_ce(logits, onehot(pi)) + value_mse(tanh_value, z)

* **Loader** (:class:`PretrainChunks`): memory-maps the chunk files, samples
  batches uniformly over *all* positions (every recorded position is an
  independent SL example). Sampling is driven by a caller-supplied
  ``np.random.Generator`` so a fixed seed reproduces a run exactly, and its
  state is persisted in the checkpoint for exact deterministic resume (same
  mechanism as the AGZ replay buffer in ``omigamax/train/train.py``).
* **Loss**: plain cross entropy over all ``N*N + 1`` logits (the human move is
  legal by construction, and we deliberately do NOT mask -- masking would leak
  legality supervision) plus value MSE. Both reuse
  :mod:`omigamax.train.loss` components: the one-hot policy target matches the
  AGZ soft-target shape ``pi (B, N*N+1)``, so ``agz_loss`` / ``policy_cross_entropy``
  / ``value_mse`` apply unchanged.
* **Optimizer**: SGD momentum 0.9, ``weight_decay`` 1e-4 (L2), grad-norm clip
  1.0 (the todo-14 evidence-backed value), lr 0.02 default with optional
  halving schedule (``--lr-steps [50000]`` => lr*0.5 past each boundary).
  The SL lr is deliberately 10x below the RL ``lr=0.2``: the human-move target
  is a sparse one-hot label (no MCTS temperature, many equivalent moves), so a
  large lr would overfit the shared feature backbone to single human choices;
  a conservative lr also protects the warm-start representation that P7's RL
  loop continues from (``models/pretrain.pt`` must load via
  ``create_model(arch)`` + ``load_checkpoint``).
* **Checkpoints**: reuse ``omigamax/train/train.py.save_checkpoint`` /
  ``load_checkpoint`` (same format: ``model_state_dict`` + ``optimizer_state_dict``
  + ``global_step`` + ``arch`` + ``rng_state``), so the RL loop's
  ``load_checkpoint(pretrain.pt)`` -> ``create_model(20, 256, 19)`` path works
  for P7.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from omigamax.train.loss import (
    agz_loss,
    make_sgd_optimizer,
    policy_cross_entropy,
    value_mse,
)
from omigamax.train.train import (
    load_checkpoint,
    restore_from_checkpoint,
    restore_rng,
    rng_state_json,
    save_checkpoint,
    set_learning_rate,
)

DEFAULT_LR = 0.02
DEFAULT_MOMENTUM = 0.9
DEFAULT_L2 = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LR_STEPS = (50000,)  # halve lr past each boundary
LR_DECAY_FACTOR = 0.5

CHUNK_GLOB = "chunk_*.npz"


# ---------------------------------------------------------------------------
# data loader: chunk npz corpus -> uniform batches
# ---------------------------------------------------------------------------

class PretrainChunks:
    """Memory-mapped reader + uniform sampler over the P3 chunk corpus.

    Each chunk file holds ``s (N,17,19,19) uint8``, ``pi (N,) uint16`` (move
    index 0..N*N-1, ``N*N`` = pass) and ``z (N,) int8`` (+-1). Files are
    memory-mapped (``np.load(..., mmap_mode="r")``) so a 31 GB corpus costs
    almost no RAM; only the sampled rows are ever read into memory.
    """

    def __init__(self, data_dir: "str | os.PathLike", chunk_glob: str = CHUNK_GLOB) -> None:
        self.data_dir = Path(data_dir)
        paths = sorted(self.data_dir.glob(chunk_glob))
        if not paths:
            raise FileNotFoundError(
                f"no '{chunk_glob}' files under {self.data_dir} -- run "
                f"scripts/convert_corpus.py first"
            )
        self.paths = paths
        self._npz = {}  # chunk idx -> open NpzFile (mmap)
        self._cache = {}  # chunk idx -> {key: ndarray view}
        self.sizes: list[int] = []
        self.starts: list[int] = []
        start = 0
        for p in paths:
            z = np.load(p, mmap_mode="r")
            n = int(z["pi"].shape[0])
            self._npz[len(self.sizes)] = z
            self.sizes.append(n)
            self.starts.append(start)
            start += n
        self.total_n = int(start)
        self.cum = np.cumsum(self.sizes, dtype=np.int64)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close every open mmap handle (call when done; idempotent)."""
        for z in self._npz.values():
            z.close()
        self._npz.clear()
        self._cache.clear()

    def __enter__(self) -> "PretrainChunks":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def num_chunks(self) -> int:
        return len(self.sizes)

    # -- access ------------------------------------------------------------

    def _get(self, chunk_idx: int, key: str) -> np.ndarray:
        cached = self._cache.get(chunk_idx)
        if cached is None or key not in cached:
            arr = self._npz[chunk_idx][key]
            if cached is None:
                cached = {}
                self._cache[chunk_idx] = cached
            cached[key] = arr
        return self._cache[chunk_idx][key]

    def sample_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict:
        """Uniformly sample ``batch_size`` positions (no replacement concerns:
        positions are independent SL examples, sampling is with replacement).

        Returns ``{"s": (B,17,19,19) uint8, "pi": (B,) uint16, "z": (B,) int8}``
        exactly in the stored dtypes (consumers convert for torch).
        """
        idx = rng.integers(0, self.total_n, size=int(batch_size))
        chunk_of = np.searchsorted(self.cum, idx, side="right")
        s_parts, pi_parts, z_parts = [], [], []
        for ci in range(self.num_chunks):
            m = chunk_of == ci
            if not m.any():
                continue
            local = idx[m] - self.starts[ci]
            s_parts.append(self._get(ci, "s")[local])
            pi_parts.append(self._get(ci, "pi")[local])
            z_parts.append(self._get(ci, "z")[local])
        return {
            "s": np.concatenate(s_parts, axis=0),
            "pi": np.concatenate(pi_parts),
            "z": np.concatenate(z_parts),
        }

    # -- validation --------------------------------------------------------

    def validate(self) -> dict:
        """Check every chunk's shape/dtype/value range; raise on violation."""
        report = {"chunks": [], "total_positions": int(self.total_n)}
        for ci in range(self.num_chunks):
            s = self._get(ci, "s")
            pi = self._get(ci, "pi")
            z = self._get(ci, "z")
            n = int(s.shape[0])
            assert s.dtype == np.uint8, f"chunk {ci}: s dtype {s.dtype}"
            assert pi.dtype == np.uint16, f"chunk {ci}: pi dtype {pi.dtype}"
            assert z.dtype == np.int8, f"chunk {ci}: z dtype {z.dtype}"
            assert s.ndim == 4 and s.shape[1] == 17 and s.shape[2] == s.shape[3], (
                f"chunk {ci}: s shape {s.shape}"
            )
            assert pi.shape == (n,) and z.shape == (n,), f"chunk {ci}: shape mismatch"
            assert int(s.size) == int(np.count_nonzero((s == 0) | (s == 1))), (
                f"chunk {ci}: s not 0/1"
            )
            assert bool(((z == 1) | (z == -1)).all()), f"chunk {ci}: z not +-1"
            assert bool((pi <= s.shape[2] * s.shape[3]).all()), (
                f"chunk {ci}: pi out of range"
            )
            report["chunks"].append({
                "chunk": self.paths[ci].name,
                "n": n,
                "shape": tuple(s.shape),
            })
        return report


# ---------------------------------------------------------------------------
# async prefetch: overlap CPU-side sampling with GPU compute
# ---------------------------------------------------------------------------

def _prepare_batch(batch: dict) -> dict:
    """Pre-convert a sampled batch to the torch-ready dtypes/contiguity.

    Runs in the producer thread so the ``(B,17,19,19)`` uint8->float32 etc.
    conversions never touch the main loop. Consumers like :func:`pretrain_step`
    do exactly these conversions themselves (and are dtype-tolerant), so a
    prepared batch is interchangeable with a raw one -- only the CPU work
    moves off the hot path.
    """
    return {
        "s": np.ascontiguousarray(batch["s"], dtype=np.float32),
        "pi": np.ascontiguousarray(batch["pi"], dtype=np.int64),
        "z": np.ascontiguousarray(batch["z"], dtype=np.float32),
    }


class _PrefetchSampler:
    """Background batch producer: overlap CPU sampling with GPU compute.

    A single daemon thread owns its own seeded ``np.random.Generator`` and
    loops ``chunks.sample_batch`` -> :func:`_prepare_batch`, handing each
    finished batch to the caller through a size-1 ``queue.Queue`` -- a classic
    double buffer where the worker is always exactly one batch ahead, so the
    next step's sampling overlaps this step's GPU work.

    *Thread, not process:* the hot path is numpy mmap reading (random ints,
    ``searchsorted``, fancy indexing of the mmap arrays, ``concatenate``), all
    C code that releases the GIL, so a thread overlaps the torch main thread
    at zero IPC cost. A subprocess would additionally have to re-open the
    chunk mmaps in the child (mmap handles don't inherit across Windows'
    spawn) and pickle multi-MB batches through a pipe -- strictly worse here.

    Sampling stays i.i.d. and deterministic per seed: the worker's generator
    is seeded once (by the caller, from the persistent run RNG) and consumed
    strictly in batch order, so timing never changes the produced stream.
    """

    def __init__(self, chunks: PretrainChunks, batch_size: int,
                 seed: int = 0, maxsize: int = 1) -> None:
        self._chunks = chunks
        self._batch_size = int(batch_size)
        self._rng = np.random.default_rng(int(seed))
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._failed: "BaseException | None" = None
        self._thread = threading.Thread(
            target=self._run, name="pretrain-prefetch", daemon=True,
        )

    def start(self) -> "_PrefetchSampler":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                batch = self._chunks.sample_batch(self._rng, self._batch_size)
                if self._stop.is_set():
                    break
                self._queue.put(_prepare_batch(batch))
        except BaseException as exc:  # never die silently: surface to caller
            self._failed = exc
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(None)  # sentinel: unblock a main-thread get()

    def get(self) -> dict:
        """Return the next prepared batch (blocks until it is ready)."""
        item = self._queue.get()
        if item is None:
            raise RuntimeError(
                "prefetch sampler failed"
                + (f": {self._failed!r}" if self._failed is not None else "")
            ) from self._failed
        return item

    def stop(self, timeout: float = 10.0) -> None:
        """Request shutdown and join the worker (safe to call once).

        Drains the queue (no sentinel -- a sentinel would occupy the single
        slot and deadlock a worker that is mid-``put``), so a worker blocked
        on ``put`` completes its batch, sees the stop event and exits.
        """
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# lr schedule / one optimizer step / run loop
# ---------------------------------------------------------------------------

def pretrain_lr(
    step: int,
    lr_base: float = DEFAULT_LR,
    lr_steps: "tuple[int, ...] | list[int]" = DEFAULT_LR_STEPS,
    decay: float = LR_DECAY_FACTOR,
) -> float:
    """Piecewise-halving schedule: multiply ``lr_base`` by ``decay`` for every
    boundary in ``lr_steps`` already passed (simple SL warm-start schedule)."""
    k = 0
    for boundary in sorted(int(s) for s in lr_steps):
        if int(step) >= boundary:
            k += 1
    return float(lr_base) * (decay ** k)


def make_pretrain_optimizer(
    model: torch.nn.Module,
    lr: float = DEFAULT_LR,
    momentum: float = DEFAULT_MOMENTUM,
    l2: float = DEFAULT_L2,
) -> torch.optim.Optimizer:
    """SGD momentum + L2 weight decay (same recipe as the AGZ optimizer)."""
    return make_sgd_optimizer(model, lr=lr, momentum=momentum, l2=l2)


def pretrain_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    device: "torch.device | None" = None,
    grad_clip: "float | None" = DEFAULT_GRAD_CLIP,
    *,
    scaler: "torch.amp.GradScaler | None" = None,
) -> dict:
    """One supervised optimizer step; return per-component metrics.

    ``batch`` comes from :meth:`PretrainChunks.sample_batch` (numpy). Converts
    the stored dtypes for torch (uint8 -> float32, uint16 move index -> int64
    one-hot, int8 z -> float32), runs CE over the full ``N*N + 1`` logits
    (no legal mask -- human moves are legal by construction and masking would
    leak legality supervision), MSE value, clips gradients and steps.

    ``scaler`` (optional ``torch.amp.GradScaler``) opts into AMP: the forward
    pass runs under ``torch.autocast("cuda", float16)`` and backward / grad
    clip / step go through the scaler (unscale-before-clip keeps the locked
    grad-norm-1.0 recipe intact). ``None`` (default) keeps the deterministic
    fp32 path byte-identical to the pre-AMP behavior. The scaler is created by
    the caller (:func:`run_pretrain`) so its internal state persists across
    steps.
    """
    if device is None:
        device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    s = torch.from_numpy(np.ascontiguousarray(batch["s"])).to(
        device, dtype=torch.float32
    )
    pi_idx = torch.from_numpy(
        np.ascontiguousarray(batch["pi"]).astype(np.int64)
    ).to(device)
    z = torch.from_numpy(np.ascontiguousarray(batch["z"])).to(
        device, dtype=torch.float32
    )

    autocast_ctx = torch.autocast(
        "cuda", enabled=scaler is not None, dtype=torch.float16
    )
    with autocast_ctx:
        logits, value = model(s)
        pi_onehot = F.one_hot(pi_idx, num_classes=logits.shape[-1]).float()
        policy_ce = policy_cross_entropy(logits, pi_onehot)
        value_loss = value_mse(value, z.view(-1, 1))
        total = agz_loss(logits, value, pi_onehot, z.view(-1, 1))

    if scaler is not None:
        scaler.scale(total).backward()
        if grad_clip is not None and float(grad_clip) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
    else:
        total.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

    with torch.no_grad():
        acc = (logits.detach().argmax(dim=-1) == pi_idx).float().mean().item()
    return {
        "loss_total": float(total.detach().cpu()),
        "loss_policy": float(policy_ce.detach().cpu()),
        "loss_value": float(value_loss.detach().cpu()),
        "acc_top1": float(acc),
    }


def run_pretrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    chunks: PretrainChunks,
    steps: int,
    *,
    rng: "np.random.Generator | None" = None,
    seed: int = 0,
    global_step: int = 0,
    batch_size: int = 64,
    device: "torch.device | None" = None,
    grad_clip: "float | None" = DEFAULT_GRAD_CLIP,
    lr_base: float = DEFAULT_LR,
    lr_steps: "tuple[int, ...] | list[int]" = DEFAULT_LR_STEPS,
    log_path: "str | os.PathLike | None" = None,
    log_every: int = 10,
    amp: bool = False,
    prefetch: bool = True,
) -> tuple[list[dict], int, "np.random.Generator"]:
    """Train ``steps`` supervised steps; return ``(metrics, global_step, rng)``.

    Per step: set the scheduled lr, get a uniform batch, run one
    :func:`pretrain_step`, append metrics (loss components, top-1 policy
    accuracy vs the human move, lr) and JSONL-log every ``log_every`` steps.
    ``global_step`` advances by one and the mutable ``rng`` is returned
    (persist both for exact resume).

    ``amp=True`` runs each step with fp16 ``autocast`` + a persistent
    ``torch.amp.GradScaler`` (opt-in; requires CUDA -- raises ``ValueError``
    otherwise). ``False`` (default) is the deterministic fp32 path.

    ``prefetch=True`` (default) runs an async :class:`_PrefetchSampler`: a
    daemon thread samples the next batch while the main thread runs the GPU
    step, so the ~350 ms/step CPU-side mmap sampling overlaps GPU compute
    (the P10 "GPU util 56% ceiling" fix). Sampling is i.i.d.; with prefetch
    the worker's generator is seeded by drawing once from the persistent
    ``rng`` per call (so distinct blocks / resumes sample distinct streams),
    and a fixed ``seed`` still reproduces a run exactly. ``rng`` is advanced
    by exactly that one draw per call and returned as usual, so checkpoint
    round-trips and ``--resume`` are unchanged. ``prefetch=False`` keeps the
    original fully-serial sampling driven directly by ``rng``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if device is None:
        device = next(model.parameters()).device
    if amp and device.type != "cuda":
        raise ValueError("amp=True requires a CUDA device")
    scaler = torch.amp.GradScaler("cuda") if amp else None
    sampler = None
    if prefetch and int(steps) > 0:
        # draw the worker's stream seed from the persistent rng (one draw per
        # call); the worker never touches `rng` again, so the returned rng
        # state is a faithful, checkpointable continuation point.
        sampler = _PrefetchSampler(
            chunks, int(batch_size), seed=int(rng.integers(0, 2**32 - 1))
        ).start()
    metrics: list[dict] = []
    log_fh = None
    try:
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8")
        for _ in range(int(steps)):
            lr = pretrain_lr(global_step, lr_base, lr_steps)
            set_learning_rate(optimizer, lr)
            if sampler is not None:
                batch = sampler.get()
            else:
                batch = chunks.sample_batch(rng, int(batch_size))
            m = pretrain_step(
                model, optimizer, batch, device=device, grad_clip=grad_clip,
                scaler=scaler,
            )
            m["step"] = int(global_step)
            m["lr"] = float(lr)
            metrics.append(m)
            if log_fh is not None and int(global_step) % int(log_every) == 0:
                log_fh.write(json.dumps(m, sort_keys=True) + "\n")
                log_fh.flush()
            global_step += 1
    finally:
        if sampler is not None:
            sampler.stop()
        if log_fh is not None:
            log_fh.close()
    return metrics, int(global_step), rng


# ---------------------------------------------------------------------------
# checkpoint helpers (reuse the train.py format so P7's RL loop can load)
# ---------------------------------------------------------------------------

def pretrain_checkpoint_path() -> Path:
    """Default pretrain checkpoint: ``models/pretrain.pt``."""
    return Path("models") / "pretrain.pt"


def save_pretrain_checkpoint(
    path: "str | os.PathLike",
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    rng: "np.random.Generator",
    *,
    config: "dict | None" = None,
    extra: "dict | None" = None,
) -> str:
    """Persist a pretrain checkpoint in the existing train.py format.

    Reuses :func:`omigamax.train.train.save_checkpoint` (writes ``arch`` from
    the model, ``global_step``, ``optimizer_state_dict`` and ``rng_state``), so
    ``load_checkpoint(models/pretrain.pt)`` + ``create_model(arch)`` works for
    the P7 RL warm-start.
    """
    return save_checkpoint(
        path, model, optimizer, global_step, config=config, rng=rng, extra=extra
    )


def resume_from_checkpoint(
    ckpt: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: "np.random.Generator",
) -> int:
    """Restore model/optimizer/rng from a pretrain checkpoint; return step."""
    step = restore_from_checkpoint(ckpt, model, optimizer)
    if "rng_state" in ckpt:
        restore_rng(rng, ckpt["rng_state"])
    return int(step)
