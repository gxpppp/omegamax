"""Live check 2 (F2 MAJOR 2): the training loop feeds the live window frames.

Runs a real ``run_loop`` (real self-play, real train, real viz thread -- a
real SnapshotQueue + VizThread) headlessly under ``SDL_VIDEODRIVER=dummy`` on
a tiny 9x9 net on CPU, and instruments :func:`loop.push_viz_frame` to prove
Snapshot frames ARE pushed during training (the F2 bug: nothing was pushed).
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import tempfile
from pathlib import Path

import torch

from omigamax.train import loop

pushed = {"frames": 0}
_orig_push = loop.push_viz_frame


def counting_push(viz, snap):
    if snap is not None:
        pushed["frames"] += 1
    return _orig_push(viz, snap)


loop.push_viz_frame = counting_push

cfg = {
    "board_size": 9, "komi": 7.5, "blocks": 1, "channels": 8,
    "lr": 0.2, "momentum": 0.9, "l2": 1e-4, "lr_schedule_steps": [50000, 100000],
    "batch_size": 16, "replay_buffer_games": 100, "symmetry_aug": False,
    "simulations": 20, "eval_games": 2, "eval_sims": 8,
    "eval_interval_steps": 2000, "replace_threshold": 0.55,
    "virtual_loss": 3, "viz_enabled": True, "viz_queue_size": 16,
    "cycle_games": 2, "cycle_steps": 5, "selfplay_max_moves": 80,
    "eval_max_moves": 80,
}

tmp = Path(tempfile.mkdtemp(prefix="f2-viz-"))
report = loop.run_loop(
    cfg, device=torch.device("cpu"),
    data_dir=tmp / "data", checkpoint_dir=tmp / "models",
    train_log=tmp / "train.jsonl", history=tmp / "eval_history.jsonl",
    cycles=1, viz_enabled=True, seed=0,
)

print("viz reason:", report["protocol"]["viz"]["reason"])
print("viz started:", report["protocol"]["viz"]["started"])
print("steps trained:", report["loop"]["steps_trained"])
print("Snapshot frames pushed during training:", pushed["frames"])
assert report["protocol"]["viz"]["started"] is True, "viz must be mounted"
assert pushed["frames"] == report["loop"]["steps_trained"], \
    "every train step must push one viz frame"
print("LIVE CHECK 2: PASS")
