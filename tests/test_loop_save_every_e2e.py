"""P16-8: E2E smoke -- REAL tiny loop, hard-kill mid-selfplay, then --resume.

While the fake-based tests in ``test_loop_save_every_games.py`` prove the
``--save-every-games`` bookkeeping with stubbed self-play, this file runs the
ACTUAL CLI (``python -m omigamax.train.loop``) as a subprocess in a temp dir:
9x9 b1c8 net on CPU, ``--selfplay-workers 2``, ``--save-every-games 5``,
2 cycles of 10 games, ``--data-dir`` in ``tmp_path`` (never the real
``data/selfplay``), ``CUDA_VISIBLE_DEVICES=`` + ``--device cpu`` (never the
GPU).

Scenario (the plan's todo-8 acceptance):

1. leg 1 starts the real loop; once the 7th self-play game of cycle 1 has
   landed (the game-5 mid-selfplay checkpoint is already on disk), the whole
   process tree is hard-killed (``taskkill /T /F`` on Windows) -- a kill can
   never corrupt ``latest.pt`` because both checkpoints and npz games are
   written atomically (tmp + ``os.replace``);
2. the killed checkpoint records ``games_this_cycle == 5`` with
   ``games_generated`` at the CYCLE BASE (0), ``steps_into_cycle == 0``;
3. leg 2 resumes: only the remaining 5 games are generated, seeded from
   ``base + 5`` (i.e. seeds ``{5..9}`` -- ``cycle_start`` event says
   ``games=5``), training continues (train.jsonl steps ``1..6`` contiguous,
   no restart), both cycles complete, and the final checkpoint clears
   ``games_this_cycle`` (0) with ``games_generated == 20``;
4. the final npz seed set is exactly ``{0..19}`` -- no gaps, no duplicates
   (games 6-7 may have been generated pre-kill; deterministic regeneration
   is harmless);
5. ``latest.pt`` is polled for loadability during the whole resumed run --
   every save (mid-selfplay at games 5/10, eval-gate, final) leaves it
   loadable.

~30-60 s wall time: two real legs with 20 real self-play games at sims=4.
"""

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

import omigamax.train.loop as loop

REPO_ROOT = Path(__file__).resolve().parents[1]

SIZE = 9
BLOCKS = 1
CHANNELS = 8
GAMES_PER_CYCLE = 10
STEPS_PER_CYCLE = 3
SAVE_EVERY = 5          # --save-every-games
KILL_AFTER_GAMES = 7    # ~7th game of cycle 1, i.e. after the game-5 save
CYCLES = 2
SEED = 0


# ---------------------------------------------------------------------------
# subprocess plumbing
# ---------------------------------------------------------------------------

def _loop_argv(tmp_path, *, resume=False, evidence="evidence.json"):
    argv = [
        sys.executable, "-m", "omigamax.train.loop",
        "--device", "cpu",
        "--blocks", str(BLOCKS), "--channels", str(CHANNELS),
        "--board-size", str(SIZE),
        "--games", str(GAMES_PER_CYCLE),
        "--train-steps", str(STEPS_PER_CYCLE),
        "--simulations", "4", "--selfplay-max-moves", "80",
        "--batch-size", "8", "--no-symmetry",
        "--eval-games", "1", "--eval-sims", "2", "--eval-max-moves", "60",
        "--selfplay-workers", "2", "--save-every-games", str(SAVE_EVERY),
        "--cycles", str(CYCLES), "--seed", str(SEED), "--viz", "off",
        "--data-dir", str(tmp_path / "data"),
        "--checkpoint-dir", str(tmp_path / "models"),
        "--train-log", str(tmp_path / "train.jsonl"),
        "--history", str(tmp_path / "eval_history.jsonl"),
        "--evidence", str(tmp_path / evidence),
    ]
    if resume:
        argv.append("--resume")
    return argv


def _loop_env():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""   # never touch the GPU
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _npz_seeds(data_dir: Path) -> list:
    """Sorted self-play seeds present in the data dir (game_%010d.npz)."""
    return sorted(int(p.stem.split("_")[1]) for p in data_dir.glob("game_*.npz"))


def _wait_for_games(data_dir: Path, n: int, timeout: float = 240.0) -> list:
    deadline = time.monotonic() + timeout
    while True:
        seeds = _npz_seeds(data_dir)
        if len(seeds) >= n:
            return seeds
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"only {len(seeds)}/{n} self-play games after {timeout:.0f}s")
        time.sleep(0.03)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Hard-kill the loop process AND its spawned self-play workers."""
    if os.name == "nt":
        r = subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, text=True)
        if r.returncode != 0:          # taskkill unavailable -> plain kill
            proc.kill()
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _jsonl_lines(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_checkpoint_shareable(path: Path) -> dict:
    """Load ``latest.pt`` WITHOUT holding a blocking handle on it.

    Plain ``torch.load`` opens the file with the default share mode (no
    ``FILE_SHARE_DELETE``), so a read landing exactly while the loop process
    ``os.replace``s its tmp save onto ``latest.pt`` makes the WRITER's rename
    fail with ``PermissionError: [WinError 5]`` on Windows. A watcher that
    polls loadability must therefore read with share-delete semantics (the
    same contract a well-behaved external reader -- e.g. the training
    dashboard -- should use), then deserialize from the in-memory bytes.
    """
    if os.name != "nt":
        return loop.load_checkpoint(path)
    import ctypes
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x1
    FILE_SHARE_WRITE = 0x2
    FILE_SHARE_DELETE = 0x4
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    handle = ctypes.windll.kernel32.CreateFileW(
        str(path), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle in (-1, ctypes.c_void_p(-1).value):  # INVALID_HANDLE_VALUE
        raise FileNotFoundError(path)
    try:
        size = ctypes.windll.kernel32.GetFileSize(handle, None)
        buf = ctypes.create_string_buffer(size)
        read = wintypes.DWORD(0)
        if not ctypes.windll.kernel32.ReadFile(
                handle, buf, size, ctypes.byref(read), None):
            raise OSError(f"ReadFile failed on {path}")
        data = buf.raw[: read.value]
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# the E2E scenario
# ---------------------------------------------------------------------------

class TestRealLoopKillResumeE2E:
    def test_kill_mid_selfplay_then_resume_continues(self, tmp_path):
        data_dir = tmp_path / "data"
        models = tmp_path / "models"
        train_log = tmp_path / "train.jsonl"

        # --- leg 1: real loop, hard-killed mid-cycle-1 self-play ------------
        proc = subprocess.Popen(_loop_argv(tmp_path),
                                cwd=REPO_ROOT, env=_loop_env())
        try:
            pre_kill_seeds = _wait_for_games(data_dir, KILL_AFTER_GAMES)
        except Exception:
            _kill_tree(proc)
            raise
        _kill_tree(proc)

        # the process died mid-self-play of cycle 1 (no training happened)
        assert not any(e.get("event") == "train_step"
                       for e in _jsonl_lines(train_log))
        # pre-kill npz set: exactly KILL_AFTER_GAMES distinct cycle-1 seeds
        # (workers>1 finish in arbitrary order, so any 7-subset of {0..9})
        assert len(pre_kill_seeds) == KILL_AFTER_GAMES
        assert len(set(pre_kill_seeds)) == KILL_AFTER_GAMES
        assert all(0 <= s < GAMES_PER_CYCLE for s in pre_kill_seeds)

        # the game-5 mid-selfplay checkpoint is on disk and loadable (the
        # atomic tmp+rename write means a hard kill can never corrupt it)
        killed = loop.load_checkpoint(models / "latest.pt")
        assert killed["extra"]["games_this_cycle"] == SAVE_EVERY
        assert killed["extra"]["games_generated"] == 0        # cycle base
        assert killed["extra"]["steps_into_cycle"] == 0
        assert killed["extra"]["cycles_completed"] == 0

        # --- leg 2: --resume, watching latest.pt loadability ----------------
        load_stats = {"ok": 0, "fail": 0, "last_err": ""}
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                try:
                    _load_checkpoint_shareable(models / "latest.pt")
                    load_stats["ok"] += 1
                except Exception as exc:  # noqa: BLE001 - record any corruption
                    load_stats["fail"] += 1
                    load_stats["last_err"] = repr(exc)
                time.sleep(0.02)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        proc2 = subprocess.Popen(_loop_argv(tmp_path, resume=True,
                                            evidence="evidence2.json"),
                                 cwd=REPO_ROOT, env=_loop_env())
        rc = proc2.wait(timeout=600)
        stop.set()
        watcher.join(timeout=5)

        assert rc == 0, f"resumed loop exited {rc}"
        # latest.pt was loadable at every sampled moment of the resumed run
        assert load_stats["fail"] == 0, load_stats["last_err"]
        assert load_stats["ok"] >= 5

        # --- seed sets: no gaps, no duplicates -------------------------------
        # continuation generated seeds starting from base+5 = {5..9} for
        # cycle 1 plus the full cycle 2 ({10..19}): the union is exactly
        # {0..19}
        final_seeds = _npz_seeds(data_dir)
        assert final_seeds == list(range(CYCLES * GAMES_PER_CYCLE)), \
            f"seed set has gaps/duplicates: {final_seeds}"

        # the resumed session itself generated 5 games for cycle 1 (the
        # remaining ones, seeds 5..9) and the full 10 for cycle 2
        events = _jsonl_lines(train_log)
        cycle_starts = [e for e in events if e.get("event") == "cycle_start"]
        assert [e["games"] for e in cycle_starts] == \
            [SAVE_EVERY, GAMES_PER_CYCLE]

        # --- training continued (not restarted) ------------------------------
        steps = [e for e in events if e.get("event") == "train_step"]
        assert [e["step"] for e in steps] == \
            list(range(1, CYCLES * STEPS_PER_CYCLE + 1))   # 1..6, contiguous
        assert len([e for e in events if e.get("event") == "eval_gate"]) == 2

        # --- resume report ----------------------------------------------------
        report = json.loads(
            (tmp_path / "evidence2.json").read_text(encoding="utf-8"))
        assert report["loop"]["resumed"] is True
        assert report["loop"]["interrupted"] is False
        assert report["loop"]["steps_trained"] == CYCLES * STEPS_PER_CYCLE
        assert report["loop"]["global_step_final"] == CYCLES * STEPS_PER_CYCLE
        assert report["loop"]["games_generated"] == CYCLES * GAMES_PER_CYCLE
        assert report["loop"]["cycles_done"] == CYCLES
        assert report["loop"]["loss_first"] is not None   # real training ran
        assert report["protocol"]["save_every_games"] == SAVE_EVERY
        assert report["protocol"]["selfplay_workers"] == 2
        assert report["checkpoint"]["best_exists"] is True  # gates ran
        assert report["checkpoint"]["latest_exists"] is True

        # --- final checkpoint: cycle complete -> games_this_cycle cleared ----
        final_ckpt = loop.load_checkpoint(models / "latest.pt")
        assert final_ckpt["extra"]["games_this_cycle"] == 0
        assert final_ckpt["extra"]["games_generated"] == \
            CYCLES * GAMES_PER_CYCLE
        assert final_ckpt["extra"]["cycles_completed"] == CYCLES
        assert final_ckpt["extra"]["steps_into_cycle"] == 0
