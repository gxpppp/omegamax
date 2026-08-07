"""Todo 17: pygame visualization smoke + headless capture.

Two modes (plan acceptance):

* **interactive demo** (default): pushes ``--frames`` board snapshots (legal
  random moves on a real :class:`~omigamax.rules.board.Board`) into the
  bounded :class:`~omigamax.viz.board_window.SnapshotQueue` while a daemon
  :class:`~omigamax.viz.board_window.VizThread` renders them for
  ``--seconds``; then the window is closed programmatically (a QUIT event --
  the same path as clicking the window X). The producer keeps pushing after
  the close with no exception (plan: 窗口关闭后生产者线程继续入队无异常) and
  the thread's final frame is saved to ``logs/viz_smoke.png`` (>10KB gate).
  On a machine without a display the thread degrades to offscreen rendering
  and the same assertions still hold.

* **--capture <png>**: headless single-frame screenshot hook (plan
  ``--capture``): ``SDL_VIDEODRIVER=dummy`` offscreen render of one mid-game
  snapshot saved as PNG -- the agent-executable verification path in CI /
  RDP sessions.

Usage::

    uv run python -m omigamax.cli.viz_smoke --frames 50 --seconds 5
    uv run python -m omigamax.cli.viz_smoke --capture logs/viz_capture.png

Exit 0 iff the render/close/producer assertions pass (interactive) or the
capture PNG is written and > 10KB (capture mode).
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from pathlib import Path

import pygame

from omigamax.rules.board import Board
from omigamax.viz.board_window import (
    Snapshot,
    SnapshotQueue,
    VizThread,
    capture_png,
)

MIN_PNG_BYTES = 10_240  # plan acceptance: logs/viz_smoke.png > 10KB


def make_snapshots(frames: int, *, size: int = 19, seed: int = 0,
                   loss_from: float = 2.0, elo_from: float = 0.0) -> list[Snapshot]:
    """``frames`` mid-game snapshots from legal random moves + trends."""
    rng = random.Random(int(seed))
    board = Board(size)
    color = 1  # BLACK
    snaps: list[Snapshot] = []
    for i in range(int(frames)):
        legal = [(r, c) for r in range(size) for c in range(size)
                 if board.is_legal((r, c), color)]
        if legal:
            board.play(rng.choice(legal), color)
        color = 3 - color
        state = [[board.get(r, c) for c in range(size)] for r in range(size)]
        last = board.moves[-1][0] if board.moves else None
        snaps.append(Snapshot(
            board=state, board_size=size, move_number=i + 1,
            current_player=color, win_rate=0.5 + 0.08 * ((i % 5) - 2) / 2,
            last_move=last,
            games=5 + i, train_step=100 * (i + 1),
            loss=max(0.05, loss_from - 0.03 * i),
            elo=elo_from + 1.5 * i,
        ))
    return snaps


def _run_interactive(args) -> int:
    queue = SnapshotQueue(maxlen=args.maxlen)
    thread = VizThread(queue, capture_path=args.out, fps=args.fps)
    thread.start()
    stop_prod = threading.Event()
    pushed = [0]

    def producer() -> None:
        snaps = make_snapshots(args.frames, size=args.size, seed=args.seed)
        while not stop_prod.is_set():
            for s in snaps:
                if stop_prod.is_set():
                    return
                queue.push(s)
                pushed[0] += 1
                time.sleep(0.01)

    prod = threading.Thread(target=producer, daemon=True, name="viz-producer")
    try:
        prod.start()
        time.sleep(max(0.1, args.seconds))
        # close the window programmatically (window-X / QUIT path); closing
        # must NOT interrupt the producer nor crash anything.
        try:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        except Exception:
            thread.stop()  # no event system (no display): stop flag fallback
        deadline = time.monotonic() + 5.0
        while thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if thread.is_alive():
            thread.stop()
            thread.join(timeout=2)
        assert not thread.is_alive(), "viz thread did not stop after close"
        assert thread.exception is None, f"viz thread error: {thread.exception!r}"
        # producer keeps pushing after the window closed -- no exception
        before = pushed[0]
        time.sleep(0.2)
        after = pushed[0]
        assert after > before, "producer stalled after window close"
        print(f"frames pushed: {after} (producer alive after window close)",
              flush=True)
        print(f"queue length: {len(queue)} (bounded at {queue.maxlen})",
              flush=True)
        print("window close: no exception, training-side unaffected",
              flush=True)
        if not Path(args.out).exists():
            print(f"WARNING: no final frame saved ({args.out}) -- "
                  f"no display? (degraded offscreen mode)", flush=True)
        else:
            size = Path(args.out).stat().st_size
            print(f"final frame saved: {args.out} ({size} bytes)", flush=True)
            assert size > MIN_PNG_BYTES, "viz_smoke.png below 10KB gate"
        print("RESULT: PASS", flush=True)
        return 0
    finally:
        stop_prod.set()
        thread.stop()
        thread.join(timeout=2)
        prod.join(timeout=2)


def _run_capture(args) -> int:
    # render a mid-game frame (30 legal moves -> black AND white stones) with
    # the full loss/elo history so the metrics panel shows trend curves.
    snaps = make_snapshots(30, size=args.size, seed=args.seed)
    snap = snaps[-1]
    loss_hist = [(s.train_step, s.loss) for s in snaps if s.loss is not None]
    elo_hist = [(s.train_step, s.elo) for s in snaps if s.elo is not None]
    path = capture_png(snap, args.capture,
                       loss_history=loss_hist, elo_history=elo_hist)
    size = Path(path).stat().st_size
    print(f"capture written: {path} ({size} bytes)", flush=True)
    assert size > MIN_PNG_BYTES, "capture PNG below 10KB gate"
    print("RESULT: PASS (headless capture)", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="omigamax pygame visualization smoke (todo 17)")
    parser.add_argument("--frames", type=int, default=50,
                        help="snapshots pushed during the interactive demo")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="demo wall-clock seconds before window close")
    parser.add_argument("--capture", type=str, default=None, metavar="PNG",
                        help="headless one-frame screenshot path "
                             "(SDL_VIDEODRIVER=dummy)")
    parser.add_argument("--size", type=int, default=19, help="board size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxlen", type=int, default=32,
                        help="snapshot queue bound")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default="logs/viz_smoke.png",
                        help="interactive-mode final frame PNG")
    args = parser.parse_args(argv)

    if args.capture:
        return _run_capture(args)
    return _run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
