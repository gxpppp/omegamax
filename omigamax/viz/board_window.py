"""Real-time pygame board window + training metrics (todo 17).

Design (per the plan's todo-17 spec, authoritative):

* **layout** -- left: 19x19 wooden-board grid with black/white stones, the
  last-move marker, the move number, the player to move and the current win
  rate; right: a metrics panel with games / training step / loss / ELO and
  rolling trend curves for loss and ELO;
* **data channel** -- :class:`SnapshotQueue`: a thread-safe, BOUNDED channel
  (``queue.Queue``, plan) whose producer is the training/self-play loop. It
  NEVER blocks the producer and, when full, drops the OLDEST frame (plan:
  丢旧帧, 仅快照消费 -- the window must not slow the training loop);
* **threading** -- :class:`VizThread` is a *daemon* thread that owns the
  pygame window and consumes snapshots. Window close (X / ESC) only sets a
  stop flag that ends the thread gracefully -- it never raises into the
  training loop. Any crash inside the thread is caught, logged and swallowed;
* **dummy / capture mode** -- ``SDL_VIDEODRIVER=dummy`` renders offscreen with
  no display (CI / RDP), and :func:`capture_png` saves a single headless frame
  so an agent can verify the window content (plan ``--capture <png>``);
* **degradation** -- pygame init failure degrades to pure-log mode (the loop
  seam in ``omigamax/train/loop.py`` keeps running either way).

The seam import target is ``omigamax.viz.board_window`` (module name matches
``loop.start_viz_if_available``).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pygame

log = logging.getLogger("omigamax.viz")

# Board color encoding -- matches omigamax.rules.board (EMPTY/BLACK/WHITE).
EMPTY, BLACK, WHITE = 0, 1, 2

# ---------------------------------------------------------------------------
# geometry / palette
# ---------------------------------------------------------------------------

CELL = 32          # px per board intersection
PAD = 36           # board margin inside the window
PANEL = 300        # px width of the metrics panel
PANEL_MARGIN = 20  # gap between board and panel
WINDOW_BG = (26, 28, 36)
WOOD = (222, 184, 135)
WOOD_EDGE = (160, 122, 82)
GRID = (40, 32, 24)
STAR = (40, 32, 24)
STONE_BLACK = (24, 24, 28)
STONE_BLACK_HI = (90, 90, 96)
STONE_WHITE = (248, 248, 246)
STONE_WHITE_EDGE = (70, 70, 70)
LAST_MOVE = (220, 46, 46)
TEXT = (230, 230, 230)
MUTED = (150, 158, 170)
ACCENT = (122, 190, 255)
CURVE_LOSS = (240, 120, 100)
CURVE_ELO = (120, 210, 140)


def window_size(board_size: int, *, cell: int = CELL, pad: int = PAD,
                panel: int = PANEL, margin: int = PANEL_MARGIN) -> tuple[int, int]:
    """(width, height) of the window for a ``board_size`` board."""
    board_px = int(board_size) * cell
    w = pad * 2 + board_px + margin + panel
    h = pad * 2 + board_px
    return w, h


def _star_points(size: int) -> list[tuple[int, int]]:
    """Hoshi points for a ``size`` board (none for tiny test boards)."""
    if size >= 13:
        axis = (3, size // 2, size - 4)
    elif size >= 9:
        axis = (size // 2,)
    else:
        axis = ()
    return [(r, c) for r in axis for c in axis]


# ---------------------------------------------------------------------------
# snapshot data + bounded channel
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """One board state + metadata frame pushed to the visualization.

    ``board`` is a 2-D array (``board[r][c]``) with ``0`` empty, ``1`` black
    stone, ``2`` white stone -- the encoding of ``omigamax.rules.board``.
    ``win_rate`` is the current-player win-rate estimate in ``[0, 1]`` when
    available; ``last_move`` is the ``(row, col)`` of the most recent stone
    or ``None``. The ``games`` / ``train_step`` / ``loss`` / ``elo`` fields
    carry the training metrics shown in the right-hand panel; any of them may
    be ``None`` while that signal is unavailable.
    """

    board: Any
    board_size: int = 19
    move_number: int = 0
    current_player: int = 1
    win_rate: Optional[float] = None
    last_move: Optional[tuple[int, int]] = None
    komi: float = 7.5
    # training metrics
    games: Optional[int] = None
    train_step: Optional[int] = None
    loss: Optional[float] = None
    elo: Optional[float] = None
    meta: dict = field(default_factory=dict)


class SnapshotQueue:
    """Thread-safe bounded snapshot channel (drop-oldest on overflow).

    The producer (training loop) calls :meth:`push` on every frame it wants
    the window to show; it never blocks and never grows unbounded -- a full
    queue evicts its OLDEST frame so the window always presents the freshest
    state (plan: 丢旧帧, 不阻塞训练).
    """

    def __init__(self, maxlen: int = 32):
        self._maxlen = max(1, int(maxlen))
        self._q: queue.Queue = queue.Queue(maxsize=self._maxlen)
        self._lock = threading.Lock()

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def push(self, snap: Snapshot) -> None:
        """Push a frame; drop the oldest when the channel is full."""
        with self._lock:
            try:
                self._q.put_nowait(snap)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:  # pragma: no cover - race guard
                    pass
                self._q.put_nowait(snap)

    def poll(self, timeout: Optional[float] = None) -> Optional[Snapshot]:
        """Consume the oldest frame, or ``None`` after ``timeout``."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def __len__(self) -> int:
        return self._q.qsize()


# ---------------------------------------------------------------------------
# rendering (pure surface code -- works with no display)
# ---------------------------------------------------------------------------

def _font(size: int) -> pygame.font.Font:
    """Idempotent font init + a monospace-ish default font."""
    pygame.font.init()
    return pygame.font.Font(None, size)


def _text(surface, text: str, x: int, y: int, size: int = 20,
          color: tuple = TEXT) -> None:
    surf = _font(size).render(text, True, color)
    surface.blit(surf, (x, y))


def _round_rect(surface, rect, radius: int, color: tuple) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def _plot_curve(surface, points, rect, color: tuple) -> None:
    """Minimal rolling trend curve inside ``rect`` (2+ points required)."""
    if len(points) < 2:
        return
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y0, y1 = y0 - 0.5, y0 + 0.5
    l, t, w, h = rect
    pad = 4
    def px(x):  # noqa: E306
        return l + pad + (x - x0) / (x1 - x0) * max(1, w - 2 * pad)
    def py(y):  # noqa: E306
        return t + h - pad - (y - y0) / (y1 - y0) * max(1, h - 2 * pad)
    pts = [(px(x), py(y)) for x, y in points]
    pygame.draw.lines(surface, color, False, pts, 2)


def _render_board(surface, snap: Snapshot, origin_x: int, origin_y: int,
                  cell: int) -> None:
    """Draw the wooden board, stones, last-move marker and status line."""
    n = snap.board_size
    board_px = n * cell
    # board background
    board_rect = pygame.Rect(origin_x, origin_y, board_px, board_px)
    _round_rect(surface, board_rect, 8, WOOD)
    pygame.draw.rect(surface, WOOD_EDGE, board_rect, 2, border_radius=8)
    # grid lines
    for i in range(n):
        p = origin_x + i * cell
        pygame.draw.line(surface, GRID, (p, origin_y), (p, origin_y + board_px), 1)
        p = origin_y + i * cell
        pygame.draw.line(surface, GRID, (origin_x, p), (origin_x + board_px, p), 1)
    # hoshi
    for r, c in _star_points(n):
        cx = origin_x + c * cell
        cy = origin_y + r * cell
        pygame.draw.circle(surface, STAR, (cx, cy), 4)
    # stones + last-move marker
    for r in range(n):
        for c in range(n):
            try:
                v = int(snap.board[r][c])
            except (TypeError, IndexError, ValueError):
                v = EMPTY
            if v not in (BLACK, WHITE):
                continue
            cx = origin_x + c * cell
            cy = origin_y + r * cell
            rad = int(cell * 0.46)
            if v == BLACK:
                pygame.draw.circle(surface, STONE_BLACK, (cx, cy), rad)
                pygame.draw.circle(
                    surface, STONE_BLACK_HI,
                    (cx - rad // 3, cy - rad // 3), max(2, rad // 3))
            else:
                pygame.draw.circle(surface, STONE_WHITE_EDGE, (cx, cy), rad)
                pygame.draw.circle(surface, STONE_WHITE,
                                   (cx - 1, cy - 1), rad - 2)
            if snap.last_move == (r, c):
                pygame.draw.circle(surface, LAST_MOVE, (cx, cy), max(3, cell // 7))


def _format_win_rate(wr: Optional[float]) -> str:
    if wr is None:
        return "--"
    if 0.0 <= wr <= 1.0:
        return f"{wr * 100.0:.1f}%"
    return f"{wr:+.2f}"


def render_surface(snap: Snapshot, *,
                   loss_history: Iterable[tuple] = (),
                   elo_history: Iterable[tuple] = (),
                   surface: Optional[pygame.Surface] = None) -> pygame.Surface:
    """Render ``snap`` (board + metrics panel) onto a surface.

    Pure surface rendering -- no window/display required, so it works under
    ``SDL_VIDEODRIVER=dummy`` and in unit tests. ``loss_history`` /
    ``elo_history`` are ``(step, value)`` sequences used for the rolling
    trend curves in the metrics panel.
    """
    w, h = window_size(snap.board_size)
    if surface is None:
        surface = pygame.Surface((w, h))
    surface.fill(WINDOW_BG)

    cell = CELL
    board_x = PAD
    board_y = PAD + 28  # leave a status line above the board
    _render_board(surface, snap, board_x, board_y, cell)

    # status line (move / player / win rate) above the board
    player = "Black" if snap.current_player == BLACK else "White"
    status = f"move {snap.move_number}   to play: {player}"
    _text(surface, status, board_x, PAD - 2, size=22, color=TEXT)
    wr = _format_win_rate(snap.win_rate)
    wr_surf = _font(22).render(f"win rate {wr}", True, ACCENT)
    surface.blit(wr_surf, (board_x, PAD + 28 + snap.board_size * cell + 6))

    # metrics panel on the right
    panel_x = board_x + snap.board_size * cell + PANEL_MARGIN
    panel_rect = pygame.Rect(panel_x, PAD, PANEL, h - 2 * PAD)
    _round_rect(surface, panel_rect, 8, (36, 38, 48))
    pygame.draw.rect(surface, (60, 64, 78), panel_rect, 1, border_radius=8)

    _text(surface, "omigamax", panel_x + 14, PAD + 12, size=24, color=ACCENT)
    _text(surface, "training metrics", panel_x + 14, PAD + 40, size=16,
          color=MUTED)

    rows = [
        ("games", snap.games),
        ("step", snap.train_step),
        ("loss", snap.loss),
        ("elo", snap.elo),
    ]
    ry = PAD + 72
    for label, value in rows:
        if value is None:
            text = f"{label}: --"
        elif isinstance(value, float):
            text = f"{label}: {value:.4f}" if label == "loss" \
                else f"{label}: {value:+.2f}"
        else:
            text = f"{label}: {int(value)}"
        _text(surface, text, panel_x + 14, ry, size=20)
        ry += 26

    # rolling trend curves
    chart_w = PANEL - 28
    chart_h = 72
    label_y = ry + 4
    _text(surface, "loss curve", panel_x + 14, label_y, size=16, color=MUTED)
    loss_rect = pygame.Rect(panel_x + 14, label_y + 22, chart_w, chart_h)
    pygame.draw.rect(surface, (24, 26, 34), loss_rect, border_radius=4)
    loss_pts = [(float(a), float(b)) for a, b in loss_history]
    if len(loss_pts) < 2:
        _text(surface, "(collecting...)", loss_rect.x + 8, loss_rect.y + 6,
              size=16, color=MUTED)
    _plot_curve(surface, loss_pts, loss_rect, CURVE_LOSS)

    elo_y = label_y + 22 + chart_h + 26
    _text(surface, "elo curve", panel_x + 14, elo_y - 26, size=16,
          color=MUTED)
    elo_rect = pygame.Rect(panel_x + 14, elo_y, chart_w, chart_h)
    pygame.draw.rect(surface, (24, 26, 34), elo_rect, border_radius=4)
    elo_pts = [(float(a), float(b)) for a, b in elo_history]
    if len(elo_pts) < 2:
        _text(surface, "(collecting...)", elo_rect.x + 8, elo_rect.y + 6,
              size=16, color=MUTED)
    _plot_curve(surface, elo_pts, elo_rect, CURVE_ELO)

    return surface


# ---------------------------------------------------------------------------
# headless capture hook (plan --capture)
# ---------------------------------------------------------------------------

def capture_png(snap: Snapshot, path: "str | Path", *,
                loss_history: Iterable[tuple] = (),
                elo_history: Iterable[tuple] = ()) -> str:
    """Render one frame offscreen and save it as PNG (headless / RDP-safe).

    Uses ``SDL_VIDEODRIVER=dummy`` (plan: 无头截图钩子) so no display is
    needed; an agent can verify the window content by inspecting the PNG.
    Returns the absolute path written.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    try:
        surf = render_surface(snap, loss_history=loss_history,
                              elo_history=elo_history)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surf, str(out))
        return str(out)
    finally:
        pygame.display.quit()


# ---------------------------------------------------------------------------
# the window thread (daemon; close never disturbs the trainer)
# ---------------------------------------------------------------------------

def _append_metric_history(history: list, step: float, value: float,
                           maxlen: int) -> None:
    history.append((float(step), float(value)))
    if len(history) > maxlen:
        del history[0]


class VizThread(threading.Thread):
    """Daemon thread owning the pygame window.

    Consumes :class:`Snapshot` frames from a :class:`SnapshotQueue`, renders
    them, and shows the window once the first frame arrives (lazy display
    creation -- a loop that never pushes snapshots never opens a window).
    Window close (X / ESC) sets the internal stop flag and the thread exits
    gracefully; the capture path, if given, is written on shutdown.

    The thread never raises into the producer: every exception is caught,
    logged and stored in :attr:`exception`, and the frame loop simply ends.
    """

    def __init__(self, snap_queue: SnapshotQueue, *, fps: int = 30,
                 title: str = "omigamax", logger=None,
                 capture_path: "str | Path | None" = None,
                 max_history: int = 500):
        super().__init__(name="omigamax-viz", daemon=True)
        self._queue = snap_queue
        self._fps = max(1, int(fps))
        self._title = title
        self._logger = logger or log
        self._capture_path = Path(capture_path) if capture_path else None
        self._max_history = max(10, int(max_history))
        self._stop_evt = threading.Event()
        self._surface = None
        self._surface_lock = threading.Lock()
        self._window_disabled = False
        self.exception: Optional[BaseException] = None
        self._started_evt = threading.Event()

    # -- public control ---------------------------------------------------

    @property
    def stopped(self) -> bool:
        return self._stop_evt.is_set()

    def stop(self) -> None:
        """Request a graceful shutdown (same effect as the window X / ESC)."""
        self._stop_evt.set()

    def last_surface(self) -> Optional[pygame.Surface]:
        """Most recently rendered frame (or ``None`` before the first)."""
        with self._surface_lock:
            return self._surface

    def wait_started(self, timeout: float = 6.0) -> bool:
        return self._started_evt.wait(timeout)

    # -- event handling (unit-testable) -----------------------------------

    def _close_requested(self, events: list) -> bool:
        for ev in events:
            if ev.type == pygame.QUIT:
                return True
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return True
        return False

    # -- the loop ---------------------------------------------------------

    def run(self) -> None:
        try:
            self._run_loop()
        except BaseException as exc:  # never propagate into the trainer
            self.exception = exc
            self._logger.warning("viz thread stopped with error: %r", exc)
        finally:
            try:
                pygame.quit()
            except Exception:  # pragma: no cover - best-effort
                pass

    def _run_loop(self) -> None:
        pygame.init()
        self._started_evt.set()
        window = None
        clock = pygame.time.Clock()
        surf = None
        loss_hist: list[tuple] = []
        elo_hist: list[tuple] = []
        while not self._stop_evt.is_set():
            if window is not None and self._close_requested(pygame.event.get()):
                self._stop_evt.set()
                break
            snap = self._queue.poll(timeout=0.05)
            if snap is not None:
                step = snap.train_step if snap.train_step is not None \
                    else snap.move_number
                if snap.loss is not None:
                    _append_metric_history(loss_hist, step, snap.loss,
                                           self._max_history)
                if snap.elo is not None:
                    _append_metric_history(elo_hist, step, snap.elo,
                                           self._max_history)
                surf = render_surface(
                    snap, loss_history=loss_hist, elo_history=elo_hist)
                with self._surface_lock:
                    self._surface = surf
                if window is None:
                    window = self._try_open_window(surf.get_size())
                if window is not None:
                    window.blit(surf, (0, 0))
                    pygame.display.flip()
            clock.tick(self._fps)
        self._save_final_frame(surf)

    def _try_open_window(self, size: tuple[int, int]):
        if self._window_disabled:
            return None
        try:
            win = pygame.display.set_mode(size)
            pygame.display.set_caption(self._title)
            return win
        except pygame.error as exc:
            self._logger.warning(
                "no display surface available (headless?): %s -- "
                "rendering offscreen only", exc)
            self._window_disabled = True
            return None

    def _save_final_frame(self, surf: Optional[pygame.Surface]) -> None:
        if self._capture_path is None or surf is None:
            return
        try:
            self._capture_path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surf, str(self._capture_path))
            self._logger.info("viz final frame saved: %s", self._capture_path)
        except Exception as exc:  # pragma: no cover - best-effort
            self._logger.warning("could not save viz final frame: %r", exc)
