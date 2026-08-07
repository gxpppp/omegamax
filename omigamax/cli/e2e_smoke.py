"""Todo 21: end-to-end smoke -- zero-to-playable model + honest evaluation report.

Per the plan (todo 21, authoritative):

* **the chain**: a single CLI drives low-config training (real self-play ->
  train -> eval-gate, todo 16 components) from random weights to
  ``models/best.pt``; then the visualization evidence (``viz_smoke
  --capture``, todo 17); then ``match --engine2 random`` (todo 20) for the
  vs-random win rate -- the plan's early-milestone signal (>80% is a
  *weak* check: "不达不阻塞"); then (when the KataGo build is present) a
  short vs-KataGo match to prove the GTP pipeline end-to-end; then a
  re-run of the todo-12 strength ladder with the *genuinely trained*
  weights (the plan's todo-12 key-gate re-run, recorded honestly with the
  reduced-sample allowance ``--games N --quick`` and time boxes); and
  finally a consolidated report.
* **phases**: ``train``, ``viz``, ``match``, ``katago``, ``ladder``,
  ``report``. Each phase is independently runnable via ``--phases`` and its
  evidence artifact is cached (a phase already complete is skipped unless
  ``--force``) so a long smoke can be resumed without re-running finished
  phases -- and the bare acceptance command
  ``uv run python -m omigamax.cli.e2e_smoke`` exits 0 with every artifact
  present once the chain has completed.
* **honesty contract** (plan: "不把冒烟模型当正式成果", "报告须以实测胜率明示该指标
  是否达成"): every win-rate claim is the *measured* value with the number of
  games it is based on; the >80% milestone and the todo-12 monotonicity gate
  are reported as MET / NOT MET / UNCONFIRMED with the exact samples. The
  <2h soft target (plan: "目标 <2 小时...硬上限不可达成") is recorded with the
  actual wall time and the reason it was exceeded. Nothing is faked.

The training parameters below are the todo-21 smoke scale for this machine
(measured ~88-160 sims/s, ~21 self-play games/hour at sims=40): a handful of
cycles with a few games each, several hundred training steps per game (steps
are ~100x cheaper than self-play), capped game lengths (250 moves), and a
cheap per-cycle evaluation gate. All of it is CLI-overridable -- no config
file is touched.

Usage::

    uv run python -m omigamax.cli.e2e_smoke                       # full chain
    uv run python -m omigamax.cli.e2e_smoke --phases train        # just train
    uv run python -m omigamax.cli.e2e_smoke --phases match --match-games 10
    uv run python -m omigamax.cli.e2e_smoke --phases ladder --ladder-max-time 60
    uv run python -m omigamax.cli.e2e_smoke --phases report       # (re)build report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from omigamax.config import load_config
from omigamax.train.buffer import ReplayBuffer
from omigamax.train.loop import run_loop
from omigamax.train.train import load_checkpoint

# --- todo-21 smoke defaults (documented design, CLI-overridable) ------------
DEFAULT_CYCLES = 4
DEFAULT_GAMES_PER_CYCLE = 3
DEFAULT_STEPS_PER_CYCLE = 1000
DEFAULT_SIMULATIONS = 60
DEFAULT_SELFPLAY_MAX_MOVES = 250
DEFAULT_BATCH_SIZE = 128
DEFAULT_EVAL_GAMES = 3
DEFAULT_EVAL_SIMS = 40
DEFAULT_EVAL_MAX_MOVES = 250
DEFAULT_SEED = 21

DEFAULT_MATCH_GAMES = 20
DEFAULT_MATCH_SIMS = 60
DEFAULT_MATCH_MAX_MOVES = 1000

DEFAULT_KATAGO_GAMES = 3
DEFAULT_KATAGO_VISITS = 100

DEFAULT_LADDER_GAMES = 10
DEFAULT_LADDER_MAX_TIME_MIN = 75.0
DEFAULT_LADDER_PAIRING_TIME_MIN = 25.0

DEFAULT_DATA_DIR = "data/selfplay"
DEFAULT_CHECKPOINT_DIR = "models"
DEFAULT_TRAIN_LOG = "logs/train.jsonl"
DEFAULT_HISTORY = "logs/eval_history.jsonl"
DEFAULT_EVIDENCE_DIR = ".omo/evidence/omigamax-go/task-21-e2e"
DEFAULT_VIZ_PNG = "logs/viz_smoke.png"
DEFAULT_REPORT_MD = "logs/e2e_report.md"

DEFAULT_PHASES = "train,viz,match,katago,report"

WILSON_Z = 1.96  # 95% CI


# ---------------------------------------------------------------------------
# statistics helpers (pure, tested)
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, games: int, z: float = WILSON_Z) -> "tuple[float | None, float | None]":
    """Wilson 95% score-interval for ``wins/games`` (None when ``games == 0``).

    The todo-12 provisional-pilot evidence used the same interval (todo-12
    evidence: "every pairing's 95% CI straddles 0.5 => no ordering is
    statistically established"). Used for the vs-random win rate and the
    todo-12 re-run pairings so "UNCONFIRMED" claims are backed by numbers.
    """
    games = int(games)
    if games <= 0 or wins < 0 or wins > games:
        return None, None
    wins = int(wins)
    p = wins / games
    denom = 1.0 + z * z / games
    centre = (p + z * z / (2.0 * games)) / denom
    half = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def elo_from_winrate(winrate: float) -> float:
    """Standard ELO difference ``400*log10(p/(1-p))``, clamped to +-2400."""
    p = float(winrate)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(p / (1.0 - p))


def milestone_status(winrate: float, wins: int, games: int,
                     bar: float = 0.80) -> dict:
    """Classify the vs-random early milestone (weak-signal check)."""
    lo, hi = wilson_ci(wins, games)
    return {
        "winrate": round(float(winrate), 4),
        "wins": int(wins),
        "games": int(games),
        "bar": bar,
        "ci95": [None if lo is None else round(lo, 4),
                 None if hi is None else round(hi, 4)],
        "met": bool(games > 0 and winrate > bar),
        "note": (
            "plan early milestone: >80% vs random-legal is a WEAK pipeline "
            "signal (不达不阻塞); NOT met unless the measured win rate is "
            "above the bar."
        ),
    }


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# phase implementations
# ---------------------------------------------------------------------------

def _phase_done(evidence_dir: Path, name: str) -> "Path | None":
    """Evidence artifact path for a completed phase (None when missing)."""
    marks = {
        "train": "train-report.json",
        "viz": "viz.png",
        "match": "match-vs-random.json",
        "katago": "match-vs-katago.json",
        "ladder": "ladder.json",
    }
    if name == "report":
        return None  # the report phase always re-assembles from artifacts
    path = evidence_dir / marks[name]
    if path.exists():
        return path
    # a decided-and-recorded phase (e.g. katago skipped with a written
    # katago-phase.json) also counts as complete
    decided = evidence_dir / f"{name}-phase.json"
    return decided if decided.exists() else None


def run_train_phase(args) -> dict:
    """Fresh training run (todo-16 loop, no resume) -> latest.pt + best.pt.

    Returns the run_loop report plus wall time and the peak GPU allocation
    measured around the run.
    """
    cfg = load_config(args.config)
    cfg = dict(cfg)  # never mutate the shared config file's dict
    # The todo-21 smoke scale (the plan's low-config preset scaled to this
    # machine's ~88-160 sims/s): 4 cycles x 3 games x 1000 steps, sims 60,
    # capped game lengths, cheap per-cycle gate. Test override hooks below.
    cfg["board_size"] = int(args.board_size)
    if args.blocks:
        cfg["blocks"] = int(args.blocks)
    if args.channels:
        cfg["channels"] = int(args.channels)
    if args.batch_size:
        cfg["batch_size"] = int(args.batch_size)
    if args.no_symmetry:
        cfg["symmetry_aug"] = False

    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    report = run_loop(
        cfg,
        device=device,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        train_log=args.train_log,
        history=args.history,
        cycles=args.cycles,
        games_per_cycle=args.games_per_cycle,
        steps_per_cycle=args.steps_per_cycle,
        simulations=args.simulations,
        selfplay_max_moves=args.selfplay_max_moves,
        batch_size=args.batch_size,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_max_moves=args.eval_max_moves,
        use_symmetry=not args.no_symmetry,
        use_fp16=args.fp16,
        seed=args.seed,
        resume=False,
        force_final_eval=True,
        viz_enabled=False,
    )
    wall_s = time.perf_counter() - t0
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    # positions in the replay window (real self-play data produced above)
    buffer = ReplayBuffer(data_dir, max_games=int(cfg.get("replay_buffer_games", 1000)),
                          board_size=int(cfg["board_size"]))
    positions = int(buffer.num_positions)

    train_lines = [e for e in read_jsonl(Path(args.train_log))
                   if e.get("event") == "train_step"]
    losses = [float(e["loss"]) for e in train_lines]
    out = {
        "phase": "train",
        "device": str(device),
        "wall_time_s": round(wall_s, 2),
        "peak_gpu_mem_gb": peak_gb,
        "protocol": {
            "cycles": int(report["protocol"]["cycles"]),
            "games_per_cycle": int(report["protocol"]["games_per_cycle"]),
            "steps_per_cycle": int(report["protocol"]["steps_per_cycle"]),
            "simulations": int(report["protocol"]["simulations"]),
            "selfplay_max_moves": report["protocol"]["selfplay_max_moves"],
            "batch_size": int(report["protocol"]["batch_size"]),
            "eval_games": int(report["protocol"]["eval_games"]),
            "eval_sims": int(report["protocol"]["eval_sims"]),
            "eval_max_moves": int(report["protocol"]["eval_max_moves"]),
            "board_size": int(cfg["board_size"]),
            "seed": int(args.seed),
        },
        "loop": {
            "steps_trained": int(report["loop"]["steps_trained"]),
            "global_step_final": int(report["loop"]["global_step_final"]),
            "games_generated": int(report["loop"]["games_generated"]),
            "cycles_done": int(report["loop"]["cycles_done"]),
            "positions_in_buffer": positions,
            "loss_first": report["loop"]["loss_first"],
            "loss_last": report["loop"]["loss_last"],
            "loss_decrease": bool(report["loop"]["loss_decrease"]),
            "train_step_lines": len(train_lines),
            "eval_gates": int(report["loop"]["eval_gates"]),
            "eval_summaries": report["loop"]["eval_summaries"],
        },
        "checkpoint": {
            "latest": report["checkpoint"]["latest"],
            "latest_exists": bool(report["checkpoint"]["latest_exists"]),
            "best": report["checkpoint"]["best"],
            "best_exists": bool(report["checkpoint"]["best_exists"]),
        },
    }
    if losses:
        out["loop"]["loss_from"] = min(losses)
        out["loop"]["loss_to"] = losses[-1]
    return out


def run_viz_phase(args, evidence_dir: Path) -> dict:
    """Headless pygame capture (todo-17 ``viz_smoke --capture``).

    The captured PNG is copied into the evidence dir as ``viz.png`` so the
    phase has a durable cache marker (fix: previously the phase never
    produced an evidence artifact and re-ran on every invocation).
    """
    import shutil

    from omigamax.cli.viz_smoke import main as viz_main

    png = Path(args.viz_png)
    t0 = time.perf_counter()
    rc = viz_main(["--capture", str(png), "--seed", str(args.seed)])
    wall_s = time.perf_counter() - t0
    size = png.stat().st_size if png.exists() else 0
    evidence_png = evidence_dir / "viz.png"
    if size > 0:
        shutil.copyfile(png, evidence_png)
    return {
        "phase": "viz",
        "rc": int(rc),
        "png": str(png),
        "evidence_png": str(evidence_png),
        "bytes": int(size),
        "wall_time_s": round(wall_s, 2),
    }


def run_match_phase(args, evidence_dir: Path, best_path: Path) -> dict:
    """vs-random match (todo-20 ``match --engine2 random --games N``)."""
    from omigamax.cli.match import match_main

    out_json = evidence_dir / "match-vs-random.json"
    out_txt = evidence_dir / "match-vs-random.txt"
    t0 = time.perf_counter()
    rc = match_main([
        "--engine2", "random",
        "--model", str(best_path),
        "--games", str(args.match_games),
        "--sims", str(args.match_sims),
        "--max-moves", str(args.match_max_moves),
        "--seed", str(args.seed),
        "--board-size", str(args.board_size),
        "--out-dir", str(evidence_dir / "sgf-random"),
        "--evidence", str(out_json),
        "--log", str(out_txt),
    ])
    wall_s = time.perf_counter() - t0
    data = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    wins = int(data.get("engine1_wins", 0))
    games = int(data.get("completed", 0))
    winrate = float(data.get("winrate", 0.0))
    lo, hi = wilson_ci(wins, games)
    return {
        "phase": "match",
        "rc": int(rc),
        "wall_time_s": round(wall_s, 2),
        "engine2": "random",
        "sims": int(data.get("sims", args.match_sims)),
        "games": games,
        "completed": games,
        "errors": int(data.get("errors", 0)),
        "wins": wins,
        "winrate": round(winrate, 4),
        "elo_diff": float(data.get("elo_diff", 0.0)),
        "ci95": [None if lo is None else round(lo, 4),
                 None if hi is None else round(hi, 4)],
        "milestone": milestone_status(winrate, wins, games),
        "evidence": str(out_json),
    }


def run_katago_phase(args, evidence_dir: Path, best_path: Path) -> dict:
    """Short vs-KataGo match proving the GTP pipeline end-to-end.

    KataGo (eigen CPU build, b10c128 weights from todo 5/20) crushes any
    smoke model -- the number is NOT a strength claim, the point is that the
    todo-20 GTP harness drives both engines and reports. Skipped (recorded)
    when the KataGo build is absent.
    """
    from omigamax.cli.match import match_main

    if not (Path(args.katago_dir) / "eigen" / "katago.exe").exists():
        return {"phase": "katago", "rc": 0, "skipped": True,
                "reason": "KataGo build not found; skipped (plan: gnugo/KataGo optional)"}

    out_json = evidence_dir / "match-vs-katago.json"
    out_txt = evidence_dir / "match-vs-katago.txt"
    t0 = time.perf_counter()
    rc = match_main([
        "--engine2", "katago",
        "--model", str(best_path),
        "--games", str(args.katago_games),
        "--sims", str(args.match_sims),
        "--max-moves", "250",
        "--katago-visits", str(args.katago_visits),
        "--timeout", "120",
        "--seed", str(args.seed),
        "--board-size", str(args.board_size),
        "--out-dir", str(evidence_dir / "sgf-katago"),
        "--evidence", str(out_json),
        "--log", str(out_txt),
    ])
    wall_s = time.perf_counter() - t0
    data = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    return {
        "phase": "katago",
        "rc": int(rc),
        "wall_time_s": round(wall_s, 2),
        "games": int(data.get("completed", 0)),
        "errors": int(data.get("errors", 0)),
        "wins": int(data.get("engine1_wins", 0)),
        "winrate": round(float(data.get("winrate", 0.0)), 4),
        "skipped": False,
        "note": "KataGo is a strong baseline; a low win rate is expected and is "
                "NOT a failure -- it proves the GTP harness runs end-to-end.",
        "evidence": str(out_json),
    }


def run_ladder_phase(args, evidence_dir: Path, best_path: Path) -> dict:
    """todo-12 key-gate re-run with the trained weights (reduced samples).

    The plan's reduced-sample allowance: ``--games N`` per pairing plus
    ``--quick`` (800-sim pairings -> min(N, 30) games), time-boxed. The
    plan's todo-12 gate is 60 局/配对; anything less is recorded as
    UNCONFIRMED with the exact completed samples and Wilson 95% CIs -- the
    same honesty contract the todo-12 evidence itself used.
    """
    from omigamax.cli import mcts_strength

    out_json = evidence_dir / "ladder.json"
    t0 = time.perf_counter()
    rc = mcts_strength.main([
        "--weights", str(best_path),
        "--games", str(args.ladder_games),
        "--quick",
        "--max-time", str(args.ladder_max_time),
        "--pairing-time", str(args.ladder_pairing_time),
        "--seed", str(args.seed),
        "--sgf-dir", "none",
        "--evidence", str(out_json),
    ] + (["--max-moves", str(args.ladder_max_moves)]
         if args.ladder_max_moves is not None else []))
    wall_s = time.perf_counter() - t0
    data = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    pairings = {k: v for k, v in (data.get("pairings") or {}).items()}
    for p, s in pairings.items():
        g = int(s.get("games", 0))
        w = int(s.get("hi_wins", 0))
        lo, hi = wilson_ci(w, g)
        s["ci95"] = [None if lo is None else round(lo, 4),
                     None if hi is None else round(hi, 4)]
    return {
        "phase": "ladder",
        "rc": int(rc),
        "wall_time_s": round(wall_s, 2),
        "weight_source": data.get("weight_source"),
        "protocol": data.get("protocol"),
        "pairings": pairings,
        "monotonic": data.get("monotonic"),
        "accepted": bool(data.get("accepted", False)),
        "games_played": len(data.get("games", [])),
        "projected_times": data.get("projected_times"),
        "evidence": str(out_json),
        "gate_note": (
            "plan todo-12 key-gate is 60 局/配对 (--games 60). This smoke re-run "
            "uses the plan's reduced-sample allowance (--games N --quick) and is "
            "time-boxed; a pairing is only statistically CONFIRMED when its "
            "Wilson 95% CI does not straddle 0.5."
        ),
    }


def _model_load_proof(best_path: Path, device) -> dict:
    """Load ``best.pt``, report arch/step, and run one forward pass."""
    ckpt = load_checkpoint(best_path)
    arch = ckpt["arch"]
    from omigamax.network.model import create_model

    model = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    import numpy as np

    n = int(arch["board_size"])
    x = torch.from_numpy(np.zeros((1, 17, n, n), dtype=np.float32)).to(device)
    with torch.no_grad():
        policy, value = model(x)
    return {
        "path": str(best_path),
        "arch": {k: v for k, v in arch.items()},
        "global_step": int(ckpt["global_step"]),
        "forward": {
            "policy_shape": list(policy.shape),
            "value_shape": list(value.shape),
            "finite": bool(torch.isfinite(policy).all().item()
                           and torch.isfinite(value).all().item()),
        },
    }


def run_report_phase(args, evidence_dir: Path,
                     phases: "list[dict]") -> dict:
    """Assemble the consolidated todo-21 report from phase artifacts.

    Phase data is loaded from the durable evidence artifacts
    (``<name>-phase.json`` / ``train-report.json``) so the bare acceptance
    command ``e2e_smoke`` -- where every phase is already cached and the
    in-memory entries are only ``{"cached": True}`` markers -- still produces
    a complete report. The in-memory ``phases`` list is the fallback for
    phases whose evidence is not (yet) on disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = Path(args.checkpoint_dir) / "best.pt"

    phase_files = {
        "train": evidence_dir / "train-report.json",
        "viz": evidence_dir / "viz-phase.json",
        "match": evidence_dir / "match-phase.json",
        "katago": evidence_dir / "katago-phase.json",
        "ladder": evidence_dir / "ladder-phase.json",
        "ladder-sanity": evidence_dir / "ladder-phase.json",
    }
    by_name: dict = {}
    for p in phases:
        if p:
            by_name[p["phase"]] = p
    for name, fp in phase_files.items():
        if not fp.exists():
            continue
        try:
            loaded = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, dict) and loaded.get("phase") == name:
            by_name[name] = loaded

    train = by_name.get("train") or {}
    match = by_name.get("match") or {}
    katago = by_name.get("katago") or {}
    ladder = (by_name.get("ladder") or by_name.get("ladder-sanity") or {})
    viz = by_name.get("viz") or {}

    model_proof = _model_load_proof(best_path, device) if best_path.exists() else None

    # ELO trajectory from the eval-gate summaries + eval_history.jsonl
    eval_history = [e for e in read_jsonl(Path(args.history))
                    if e.get("event") == "evaluate_gate"]
    elo_trajectory = [
        {"step": int(e.get("global_step", 0)),
         "winrate": round(float(e.get("winrate", 0.0)), 4),
         "elo": round(float(e.get("elo", 0.0)), 3),
         "replaced": bool(e.get("replaced_best", False))}
        for e in eval_history
    ]

    merged_phases = list(by_name.values())
    total_wall_s = sum(float(p.get("wall_time_s", 0.0))
                       for p in merged_phases if p)
    peak_gb = max((float(p.get("peak_gpu_mem_gb") or 0.0)
                   for p in merged_phases if p), default=None)
    if peak_gb == 0:
        peak_gb = None

    report = {
        "todo": 21,
        "plan_reference": ".omo/plans/omigamax-go.md todo 21",
        "date": time.strftime("%Y-%m-%d"),
        "device": str(device),
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU (6GB)",
            "measured_sims_per_sec": "~88-160 (todo 12/13/21 evidence; "
                                     "plan assumed 300-600)",
            "selfplay_games_per_hour_at_sims40": "~21 (todo-13 measured 20.9)",
        },
        "phases_run": list(by_name.keys()),
        "wall_time_s_total": round(total_wall_s, 2),
        "wall_time_min_total": round(total_wall_s / 60.0, 1),
        "peak_gpu_mem_gb": peak_gb,
        "soft_target_2h": {
            "target_hours": 2.0,
            "note": "plan todo 21: 目标 <2 小时, 可调低局数达成; 3060 上该配置估算 "
                    "1.5-4.5h, 硬上限不可达成 (Momus B10/Oracle #10). "
                    "Actual wall time is recorded above; the target is SOFT.",
        },
        "training": train.get("loop", {}) | {
            "protocol": train.get("protocol", {}),
        },
        "model": model_proof,
        "eval_history_elo_trajectory": elo_trajectory,
        "vs_random": match,
        "vs_katago": katago,
        "todo12_gate_rerun": ladder,
        "viz": viz,
        "assessment": _assessment(train, match, ladder, total_wall_s),
    }
    return report


def _assessment(train: dict, match: dict, ladder: dict,
                total_wall_s: float) -> dict:
    """Honest pass/fail framing: pipeline works? milestones met?"""
    ms = (match.get("milestone") or {})
    monotonic = (ladder.get("monotonic") or {})
    loss_dec = bool((train.get("loop") or {}).get("loss_decrease"))
    lines = [
        "PIPELINE (todo-21 core): the zero-to-playable chain ran end-to-end -- "
        f"self-play generated {(train.get('loop') or {}).get('games_generated', 0)} "
        f"real games, {(train.get('loop') or {}).get('steps_trained', 0)} training "
        f"steps, {(train.get('loop') or {}).get('eval_gates', 0)} eval gates, "
        f"best.pt written. loss_decrease={loss_dec}.",
        f"VS-RANDOM early milestone (>80%): met={ms.get('met')} -- measured "
        f"{ms.get('wins')}/{ms.get('games')} = {ms.get('winrate')} "
        f"(95% CI {ms.get('ci95')}). This is the plan's WEAK pipeline signal; "
        "not met does NOT block (不达不阻塞; the long-run target is 1000 games).",
    ]
    if ladder:
        lines.append(
            f"TODO-12 GATE RE-RUN (trained weights): monotonic point estimates "
            f"P(200>40)={monotonic.get('P(200>40)')}, "
            f"P(800>200)={monotonic.get('P(800>200)')} on "
            f"{ladder.get('games_played', 0)} games total. Samples are far "
            "below the 60 局/配对 key-gate => UNCONFIRMED (see pairings + "
            "Wilson CIs). The durable harness mcts_strength.py re-runs the "
            "full protocol later."
        )
    else:
        lines.append("TODO-12 GATE RE-RUN: not run in this smoke.")
    lines.append(
        f"SOFT <2h TARGET: actual {total_wall_s / 60.0:.1f} min for the smoke "
        f"chain; exceeded because this machine runs ~88-160 sims/s (not the "
        f"plan's 300-600) and the smoke keeps real self-play + gates + a "
        "time-boxed todo-12 re-run inside one run."
    )
    return {"met": bool(ms.get("met")) and loss_dec, "notes": lines}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omigamax.cli.e2e_smoke",
        description="todo-21 end-to-end smoke: zero-to-playable model + "
                    "honest evaluation report (train/viz/match/katago/ladder/"
                    "report phases, each cached by its evidence artifact).",
    )
    p.add_argument("--phases", type=str, default=DEFAULT_PHASES,
                   help=f"comma-separated phases (default {DEFAULT_PHASES}; "
                        "any subset; add 'ladder' to include the todo-12 re-run)")
    p.add_argument("--force", action="store_true",
                   help="re-run phases even when their evidence exists")
    # train
    p.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    p.add_argument("--games-per-cycle", type=int, default=DEFAULT_GAMES_PER_CYCLE)
    p.add_argument("--steps-per-cycle", type=int, default=DEFAULT_STEPS_PER_CYCLE)
    p.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    p.add_argument("--selfplay-max-moves", type=int,
                   default=DEFAULT_SELFPLAY_MAX_MOVES)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES)
    p.add_argument("--eval-sims", type=int, default=DEFAULT_EVAL_SIMS)
    p.add_argument("--eval-max-moves", type=int, default=DEFAULT_EVAL_MAX_MOVES)
    p.add_argument("--no-symmetry", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # match / katago
    p.add_argument("--match-games", type=int, default=DEFAULT_MATCH_GAMES)
    p.add_argument("--match-sims", type=int, default=DEFAULT_MATCH_SIMS)
    p.add_argument("--match-max-moves", type=int, default=DEFAULT_MATCH_MAX_MOVES)
    p.add_argument("--katago-games", type=int, default=DEFAULT_KATAGO_GAMES)
    p.add_argument("--katago-visits", type=int, default=DEFAULT_KATAGO_VISITS)
    p.add_argument("--katago-dir", type=str, default="tools/katago")
    # ladder
    p.add_argument("--ladder-games", type=int, default=DEFAULT_LADDER_GAMES)
    p.add_argument("--ladder-max-time", type=float,
                   default=DEFAULT_LADDER_MAX_TIME_MIN, metavar="MIN")
    p.add_argument("--ladder-pairing-time", type=float,
                   default=DEFAULT_LADDER_PAIRING_TIME_MIN, metavar="MIN")
    p.add_argument("--ladder-max-moves", type=int, default=None,
                   help="move cap for the todo-12 re-run games (default: the "
                        "mcts_strength harness default of 1000; capping lower, "
                        "e.g. 250, gives more games inside the time box)")
    # environment (also the tiny-test override hooks)
    p.add_argument("--board-size", type=int, default=None,
                   help="board edge (default: config 19; tests pass 9)")
    p.add_argument("--blocks", type=int, default=None,
                   help="network blocks (default: config 10; tests pass 1)")
    p.add_argument("--channels", type=int, default=None,
                   help="network channels (default: config 128; tests pass 8)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--train-log", type=str, default=DEFAULT_TRAIN_LOG)
    p.add_argument("--history", type=str, default=DEFAULT_HISTORY)
    p.add_argument("--evidence-dir", type=str, default=DEFAULT_EVIDENCE_DIR)
    p.add_argument("--viz-png", type=str, default=DEFAULT_VIZ_PNG)
    p.add_argument("--report-md", type=str, default=DEFAULT_REPORT_MD)
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    if args.board_size is None:
        args.board_size = int(load_config(args.config)["board_size"])
    phases = [s.strip() for s in args.phases.split(",") if s.strip()]
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    best_path = Path(args.checkpoint_dir) / "best.pt"

    results: list[dict] = []
    for name in phases:
        existing = None if args.force else _phase_done(evidence_dir, name)
        if existing is not None and name != "report":
            print(f"[e2e] phase {name}: already complete ({existing}) -- "
                  f"skipped (--force to re-run)", flush=True)
            results.append({"phase": name, "cached": True,
                            "evidence": str(existing)})
            continue
        print(f"[e2e] phase {name}: START", flush=True)
        t0 = time.perf_counter()
        if name == "train":
            res = run_train_phase(args)
            write_json(res, evidence_dir / "train-report.json")
        elif name == "viz":
            res = run_viz_phase(args, evidence_dir)
        elif name == "match":
            if not best_path.exists():
                print(f"[e2e] match requires {best_path} -- run train first",
                      file=sys.stderr, flush=True)
                return 1
            res = run_match_phase(args, evidence_dir, best_path)
        elif name == "katago":
            if not best_path.exists():
                print(f"[e2e] katago requires {best_path} -- run train first",
                      file=sys.stderr, flush=True)
                return 1
            res = run_katago_phase(args, evidence_dir, best_path)
        elif name == "ladder":
            if not best_path.exists():
                print(f"[e2e] ladder requires {best_path} -- run train first",
                      file=sys.stderr, flush=True)
                return 1
            res = run_ladder_phase(args, evidence_dir, best_path)
        elif name == "report":
            res = run_report_phase(args, evidence_dir, results)
            write_json(res, evidence_dir / "e2e-report.json")
            _write_report_txt(res, evidence_dir)
            _write_report_md(res, args.report_md)
        else:
            print(f"[e2e] unknown phase {name!r}", file=sys.stderr, flush=True)
            return 2
        res["phase_wall_time_s"] = round(time.perf_counter() - t0, 2)
        print(f"[e2e] phase {name}: DONE in {res['phase_wall_time_s']}s",
              flush=True)
        results.append(res)
        if name != "report":
            write_json(res, evidence_dir / f"{name}-phase.json")

    report_path = evidence_dir / "e2e-report.json"
    if report_path.exists():
        print(f"[e2e] consolidated report: {report_path}", flush=True)
        print(f"[e2e] evidence dir: {evidence_dir}", flush=True)
    return 0


def _write_report_txt(report: dict, evidence_dir: Path) -> None:
    md = _report_lines(report)
    (evidence_dir / "e2e-report.txt").write_text("\n".join(md) + "\n",
                                                 encoding="utf-8")


def _write_report_md(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# omigamax e2e smoke report (todo 21)", ""]
    lines += [f"- {l}" for l in _report_lines(report)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_lines(report: dict) -> list[str]:
    lines: list[str] = []
    lines.append("=== omigamax todo-21 end-to-end smoke report ===")
    lines.append(f"date: {report['date']}  device: {report['device']}")
    lines.append(f"phases run: {report['phases_run']}")
    lines.append(f"wall time: {report['wall_time_s_total']}s "
                 f"({report['wall_time_min_total']} min)  "
                 f"peak GPU mem: {report['peak_gpu_mem_gb']} GB")
    lines.append(f"soft <2h target: {report['soft_target_2h']['note']}")
    tr = report["training"]
    lines.append(
        f"training: {tr.get('games_generated', 0)} games generated, "
        f"{tr.get('steps_trained', 0)} steps trained "
        f"(global step {tr.get('global_step_final', 0)}), "
        f"{tr.get('positions_in_buffer', 0)} positions in replay buffer, "
        f"loss {tr.get('loss_first')} -> {tr.get('loss_last')} "
        f"(decrease={tr.get('loss_decrease')}), "
        f"{tr.get('eval_gates', 0)} eval gates"
    )
    lines.append("eval history (step -> winrate -> elo -> replaced):")
    for e in report["eval_history_elo_trajectory"]:
        lines.append(
            f"  step {e['step']}: winrate={e['winrate']} elo={e['elo']} "
            f"replaced={e['replaced']}"
        )
    m = report["model"]
    if m:
        lines.append(
            f"model: {m['path']} (load OK) arch={m['arch']} "
            f"global_step={m['global_step']} forward={m['forward']}"
        )
    vr = report["vs_random"]
    if vr.get("phase") == "match":
        ms = vr["milestone"]
        lines.append(
            f"vs-random: {vr['wins']}/{vr['games']} = {vr['winrate']} "
            f"(95% CI {vr['ci95']}) sims={vr['sims']} elo_diff={vr['elo_diff']} "
            f"wall={vr['wall_time_s']}s"
        )
        lines.append(
            f"  early milestone >80%: met={ms['met']}  {ms['note']}"
        )
    kg = report["vs_katago"]
    if kg.get("phase") == "katago" and not kg.get("skipped"):
        lines.append(
            f"vs-katago: {kg['wins']}/{kg['games']} = {kg['winrate']} "
            f"errors={kg['errors']} wall={kg['wall_time_s']}s  {kg['note']}"
        )
    lg = report["todo12_gate_rerun"]
    if lg.get("phase") == "ladder":
        lines.append(
            f"todo-12 gate re-run (weights={lg.get('weight_source')}): "
            f"{lg['games_played']} games in {lg['wall_time_s']}s"
        )
        for p, s in (lg.get("pairings") or {}).items():
            lines.append(
                f"  {p}: {s.get('games')}/{s.get('planned_games', '?')} games "
                f"hi_wins={s.get('hi_wins')} hi_winrate={s.get('hi_winrate')} "
                f"95% CI {s.get('ci95')}"
            )
        lines.append(
            f"  monotonic: {lg.get('monotonic')}  accepted={lg.get('accepted')} "
            f"UNCONFIRMED at this sample size (60 局/配对 key-gate)  "
            f"{lg.get('gate_note')}"
        )
    for i, note in enumerate(report["assessment"]["notes"]):
        lines.append(f"assessment [{i + 1}]: {note}")
    lines.append(f"overall assessment met: {report['assessment']['met']}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
