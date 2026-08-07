"""Tests for the pygame real-time visualization (todo 17).

Per the plan's todo-17 spec (authoritative), this covers:

* **snapshot channel** -- :class:`~omigamax.viz.board_window.SnapshotQueue`
  is bounded, thread-safe, never blocks the producer and drops the OLDEST
  frame when full (plan: ``queue.Queue`` 线程安全; 仅快照消费, 不阻塞 -- the
  trainer must never stall on a slow/paused window);
* **headless import** -- the window module imports cleanly under the SDL
  ``dummy`` video driver (no display / RDP session) so agent + CI
  verification works (plan ``--capture``: ``SDL_VIDEODRIVER=dummy`` 离屏渲染);
* **rendering** -- a board snapshot renders to a valid surface, and the
  headless capture hook writes a PNG > 10KB (plan acceptance gate for
  ``logs/viz_smoke.png``);
* **window close** -- QUIT (window X) / ESC set the stop flag and end the viz
  thread gracefully WITHOUT raising into the training loop; the producer can
  keep pushing snapshots after the window closes (plan: 关窗仅停可视化,
  训练不中断);
* **seam integration** -- ``loop.start_viz_if_available`` now finds the todo-17
  module and starts a viz thread; the graceful-degradation path for a missing
  module is covered by the updated tests in test_loop.py.

All window-dependent tests force ``SDL_VIDEODRIVER=dummy`` so they run on a
headless / CI box exactly as on a machine with a display.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import pygame
import pytest

import omigamax.train.loop as loop
from omigamax.rules.board import Board
from omigamax.viz.board_window import (
    Snapshot,
    SnapshotQueue,
    VizThread,
    capture_png,
    render_surface,
    window_size,
)

DUMMY_DRIVER = "dummy"


def _use_dummy_driver() -> None:
    """Force SDL into offscreen (no window) video mode for this process."""
    os.environ["SDL_VIDEODRIVER"] = DUMMY_DRIVER


def wait_until(pred, timeout: float = 6.0, interval: float = 0.02) -> bool:
    """Poll ``pred`` until it is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def make_snapshot(move: int = 12, *, size: int = 19, seed: int = 0,
                  loss: float | None = 1.25, elo: float | None = 3.0,
                  games: int | None = 7, train_step: int | None = 1200,
                  win_rate: float | None = 0.55) -> Snapshot:
    """A realistic mid-game snapshot from legal random moves on a real Board."""
    rng = random.Random(int(seed))
    board = Board(size)
    color = 1  # BLACK
    for _ in range(int(move)):
        legal = [(r, c) for r in range(size) for c in range(size)
                 if board.is_legal((r, c), color)]
        if not legal:
            break
        board.play(rng.choice(legal), color)
        color = 3 - color
    state = [[board.get(r, c) for c in range(size)] for r in range(size)]
    last = board.moves[-1][0] if board.moves else None
    return Snapshot(
        board=state, board_size=size, move_number=int(move),
        current_player=color, win_rate=win_rate,
        last_move=last, games=games, train_step=train_step,
        loss=loss, elo=elo,
    )


# ---------------------------------------------------------------------------
# snapshot channel semantics (bounded, non-blocking, drop-oldest)
# ---------------------------------------------------------------------------

class TestSnapshotQueue:
    def test_fifo_order(self):
        q = SnapshotQueue(maxlen=8)
        snaps = [Snapshot(board=[[0]] * 1, board_size=1) for _ in range(3)]
        for s in snaps:
            q.push(s)
        got = [q.poll() for _ in range(3)]
        assert got == snaps
        assert q.poll(timeout=0.05) is None

    def test_bounded_drops_oldest(self):
        """Full queue drops the OLDEST frame, keeps the newest (plan: 丢旧帧)."""
        q = SnapshotQueue(maxlen=4)
        for i in range(10):
            q.push(Snapshot(board=[[i]], board_size=1))
        assert len(q) == 4
        remaining = [q.poll().board[0][0] for _ in range(4)]
        assert remaining == [6, 7, 8, 9]

    def test_push_never_blocks_or_raises_when_full(self):
        q = SnapshotQueue(maxlen=3)
        for _ in range(1000):  # no exception, bounded the whole time
            q.push(Snapshot(board=[[0]], board_size=1))
            assert len(q) <= 3

    def test_poll_has_timeout_and_returns_none_when_empty(self):
        q = SnapshotQueue(maxlen=8)
        started = time.monotonic()
        got = q.poll(timeout=0.05)
        assert got is None
        assert time.monotonic() - started >= 0.04


# ---------------------------------------------------------------------------
# snapshot data
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_defaults(self):
        s = Snapshot(board=[[0]], board_size=1)
        assert s.move_number == 0
        assert s.current_player == 1
        assert s.win_rate is None
        assert s.games is None and s.train_step is None
        assert s.loss is None and s.elo is None
        assert s.last_move is None

    def test_round_trip_fields(self):
        s = make_snapshot()
        assert s.board_size == 19
        assert len(s.board) == 19 and len(s.board[0]) == 19
        assert s.games == 7 and s.train_step == 1200
        assert s.loss == 1.25 and s.elo == 3.0


# ---------------------------------------------------------------------------
# headless rendering + capture
# ---------------------------------------------------------------------------

class TestRenderHeadless:
    def test_board_window_imports_headless_without_display(self):
        """Plan: dummy driver / RDP session must import and construct cleanly."""
        _use_dummy_driver()
        # re-import in this test is fine -- the module must not need a display
        import omigamax.viz.board_window as bw
        q = bw.SnapshotQueue(maxlen=4)
        q.push(make_snapshot())
        assert bw.SnapshotQueue is SnapshotQueue

    def test_render_surface_produces_valid_surface(self):
        _use_dummy_driver()
        snap = make_snapshot(move=24, seed=3)
        surf = render_surface(snap)
        assert isinstance(surf, pygame.Surface)
        assert surf.get_size() == window_size(snap.board_size)
        # a stone-carrying board must differ from an empty-board render
        empty = render_surface(Snapshot(board=[[0] * 19 for _ in range(19)],
                                        board_size=19))
        assert pygame.image.tostring(surf, "RGB") != pygame.image.tostring(
            empty, "RGB")

    def test_render_with_metrics_history(self):
        """Loss/ELO rolling curves render without error given history."""
        _use_dummy_driver()
        snap = make_snapshot()
        loss_hist = [(i * 100, 2.0 - 0.05 * i) for i in range(30)]
        elo_hist = [(i * 100, 0.5 * i) for i in range(30)]
        surf = render_surface(snap, loss_history=loss_hist,
                              elo_history=elo_hist)
        assert surf.get_size() == window_size(19)

    def test_capture_png_writes_file_over_10kb(self, tmp_path):
        """Plan acceptance: headless ``--capture`` PNG exists and > 10KB."""
        _use_dummy_driver()
        out = tmp_path / "capture.png"
        capture_png(make_snapshot(move=30, seed=11), out)
        assert out.exists()
        assert out.stat().st_size > 10_240


# ---------------------------------------------------------------------------
# window close handling (QUIT / stop flag, graceful, no crash)
# ---------------------------------------------------------------------------

class TestWindowClose:
    def test_quit_event_sets_stop_flag_and_ends_thread(self, tmp_path):
        """Window X (QUIT) stops the viz thread without raising."""
        _use_dummy_driver()
        q = SnapshotQueue(maxlen=8)
        t = VizThread(q, capture_path=str(tmp_path / "final.png"))
        t.start()
        try:
            q.push(make_snapshot())
            assert wait_until(lambda: t.last_surface() is not None), \
                "thread never rendered the first frame"
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            assert wait_until(lambda: not t.is_alive()), \
                "thread did not stop after QUIT"
            assert t.stopped is True
            assert t.exception is None
            assert (tmp_path / "final.png").exists()  # saved before quit
        finally:
            t.stop()
            t.join(timeout=3)

    def test_esc_key_ends_thread(self):
        _use_dummy_driver()
        q = SnapshotQueue(maxlen=8)
        t = VizThread(q)
        t.start()
        try:
            q.push(make_snapshot())
            assert wait_until(lambda: t.last_surface() is not None)
            # feed a KEYDOWN event for ESC -- same close path as window X
            ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
            assert t._close_requested([ev]) is True
            pygame.event.post(ev)
            assert wait_until(lambda: not t.is_alive())
            assert t.exception is None
        finally:
            t.stop()
            t.join(timeout=3)

    def test_stop_flag_ends_thread_gracefully(self, tmp_path):
        _use_dummy_driver()
        q = SnapshotQueue(maxlen=8)
        t = VizThread(q, capture_path=str(tmp_path / "final.png"))
        t.start()
        try:
            q.push(make_snapshot())
            assert wait_until(lambda: t.last_surface() is not None)
            t.stop()
            assert wait_until(lambda: not t.is_alive())
            assert t.exception is None
        finally:
            t.join(timeout=3)

    def test_close_does_not_break_producer(self):
        """Plan: 窗口关闭后生产者线程继续入队无异常."""
        _use_dummy_driver()
        q = SnapshotQueue(maxlen=8)
        t = VizThread(q)
        t.start()
        try:
            for _ in range(10):
                q.push(make_snapshot())
            assert wait_until(lambda: t.last_surface() is not None)
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            assert wait_until(lambda: not t.is_alive())
            # producer keeps pushing after the window closed: no exception,
            # still bounded
            for _ in range(50):
                q.push(make_snapshot())
            assert len(q) <= 8
            assert t.exception is None
        finally:
            t.stop()
            t.join(timeout=3)


# ---------------------------------------------------------------------------
# loop seam integration (todo 16 mount point now finds todo 17)
# ---------------------------------------------------------------------------

class TestSeamIntegration:
    def test_start_viz_if_available_finds_module_and_starts_thread(self):
        _use_dummy_driver()
        out = loop.start_viz_if_available({"viz_enabled": True})
        assert out["started"] is True
        assert out["reason"] == "available"
        assert out["queue"] is not None and out["thread"] is not None
        assert callable(out["stop"])
        try:
            # a live thread behind the handle; closing it is the loop's job
            assert out["thread"].is_alive() or out["thread"].stopped
        finally:
            out["stop"]()
            out["thread"].join(timeout=3)

    def test_missing_module_degrades_without_crash(self, monkeypatch, caplog):
        """Graceful-degradation path (module absent) is preserved."""
        import logging
        monkeypatch.setitem(sys.modules, "omigamax.viz.board_window", None)
        logger = logging.getLogger("test_viz_degrade")
        caplog.set_level(logging.WARNING, logger="test_viz_degrade")
        out = loop.start_viz_if_available({"viz_enabled": True},
                                          logger=logger)
        assert out["started"] is False
        assert out["reason"] == "module_unavailable"
        assert "viz" in caplog.text.lower()
