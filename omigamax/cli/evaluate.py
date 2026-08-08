"""P8: acceptance-evaluation CLI for the P5 pretrained net (验收评估).

The plan's acceptance question is "did the pretrained net actually learn to
play Go?" -- and the answer needs numbers. This module provides three modes,
all loading checkpoints via ``load_checkpoint`` with the arch read from the
checkpoint itself (so ``models/pretrain.pt``'s b20c256 restores a b20c256
net regardless of ``config/default.yaml``):

1. ``human-match`` -- SL accuracy on a held-out eval set sampled *on the fly*
   from the P3 chunk corpus (``data/pretrain/chunk_*.npz``) with a fixed eval
   seed, reusing the exact ``PretrainChunks`` loader pattern from
   :mod:`omigamax.train.pretrain` but WITHOUT training -- just iterate. No
   files are written under ``data/pretrain``. Reports top-1 / top-5 policy
   accuracy vs the human move, policy CE, value MSE and the Pearson
   correlation between ``z`` and ``tanh(value)``, plus the same metrics for a
   uniform-policy RANDOM baseline (the "untrained floor": top-1 ~ 1/362 on
   19x19, top-5 ~ 5/362, value MSE of predicting 0).

2. ``bench`` -- N MCTS games between the checkpoint and either a second
   checkpoint (``--opponent``) or a random-init "before training" baseline net
   (``create_model`` with the checkpoint's arch, default init), alternating
   colours with the same komi. Reuses the existing two-net eval-game engine
   (:func:`omigamax.train.evaluate.play_eval_game` -- the same
   ``BatchedNetworkEvaluator`` / MCTS ``run_search`` machinery the todo-15
   gate uses). The self-play ``generate_games`` path plays ONE net on both
   sides, so it is not usable for a two-net match. Reports the win rate from
   each side, draws and average game length.

3. ``report`` -- format a human-readable summary of both modes' numbers into
   ``.omo/evidence/omigamax-go/task-P8-eval.txt`` (``--report`` overrides).

All inference is CPU-safe (``eval()`` mode + ``torch.no_grad``); nothing here
trains. The full acceptance run on the trained checkpoint is deferred until
the P6 pretraining run finishes (GPU is busy); the P8 code + tests land now.

Usage::

    uv run python -m omigamax.cli.evaluate human-match --checkpoint models/pretrain.pt \
        --samples 20000
    uv run python -m omigamax.cli.evaluate bench --checkpoint models/pretrain.pt \
        --games 10 --sims 150
    uv run python -m omigamax.cli.evaluate report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from omigamax.config import load_config
from omigamax.network.model import create_model
from omigamax.train.evaluate import play_eval_game
from omigamax.train.loss import policy_cross_entropy, value_mse
from omigamax.train.pretrain import PretrainChunks
from omigamax.train.train import latest_checkpoint_path, load_checkpoint

DEFAULT_DATA_DIR = "data/pretrain"
DEFAULT_SAMPLES = 20000       # CPU-safe default for human-match
DEFAULT_GAMES = 10            # default bench games
DEFAULT_SIMS = 150            # small MCTS config (plan: 100-200) -- CPU-feasible
DEFAULT_MAX_MOVES = 1000      # eval timeout protection (same as the todo-15 gate)
# Fixed eval seed, deliberately distinct from the P5 training seeds (0/..): the
# sampled positions are the "held-out" set for this acceptance run. Sampling is
# deterministic per seed, so a rerun reproduces the exact same metrics.
DEFAULT_EVAL_SEED = 0x5EED
RANDOM_BASE_SEED = 0x4A55     # extra offset so the random-baseline draws differ
EVIDENCE_DIR = Path(".omo") / "evidence" / "omigamax-go"
DEFAULT_HUMAN_JSON = EVIDENCE_DIR / "task-P8-human.json"
DEFAULT_BENCH_JSON = EVIDENCE_DIR / "task-P8-bench.json"
DEFAULT_REPORT = EVIDENCE_DIR / "task-P8-eval.txt"


# ---------------------------------------------------------------------------
# checkpoint / eval-set loading (arch always comes from the checkpoint)
# ---------------------------------------------------------------------------

def load_eval_model(
    path: "str | Path", device: torch.device
) -> "tuple[torch.nn.Module, dict, int]":
    """Load a checkpoint's net (arch from the checkpoint) for pure inference."""
    ckpt = load_checkpoint(path)
    arch = ckpt["arch"]
    model = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, arch, int(ckpt.get("global_step", 0))


def sample_eval_set(
    data_dir: "str | Path", n: int, seed: int
) -> dict:
    """Sample ``n`` positions from the chunk corpus with a fixed seed.

    Reuses the exact ``PretrainChunks`` memory-mapped loader + uniform sampler
    from the pretrain pipeline, but only *reads* -- no training, nothing is
    written to ``data_dir`` (eval temp artifacts live outside the corpus).
    """
    with PretrainChunks(data_dir) as chunks:
        return chunks.sample_batch(np.random.default_rng(int(seed)), int(n))


# ---------------------------------------------------------------------------
# human-match metrics
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module, batch: dict, device: torch.device
) -> dict:
    """Forward the eval set and compute top-1/top-5 accuracy, CE, MSE, Pearson.

    Metrics mirror the pretrain step's definitions: policy CE against the
    one-hot human move (no legal mask -- human moves are legal by
    construction), value MSE against ``z``, top-1 = argmax(logits) == human
    move, top-5 = human move in the top-5 logits, and the Pearson correlation
    between the game outcome ``z`` and the value head's ``tanh`` output.
    """
    model.eval()
    with torch.no_grad():
        s = torch.from_numpy(np.ascontiguousarray(batch["s"])).to(
            device, dtype=torch.float32
        )
        pi_idx = torch.from_numpy(
            np.ascontiguousarray(batch["pi"]).astype(np.int64)
        ).to(device)
        z = torch.from_numpy(np.ascontiguousarray(batch["z"])).to(
            device, dtype=torch.float32
        )
        logits, value = model(s)
        pi_onehot = F.one_hot(pi_idx, num_classes=logits.shape[-1]).float()
        ce = float(policy_cross_entropy(logits, pi_onehot).item())
        mse = float(value_mse(value, z.view(-1, 1)).item())
        top1 = float((logits.argmax(dim=-1) == pi_idx).float().mean().item())
        k = min(5, int(logits.shape[-1]))
        topk = logits.topk(k, dim=-1).indices
        top5 = float(
            (pi_idx.unsqueeze(1) == topk).any(dim=1).float().mean().item()
        )
        v = np.tanh(value.detach().cpu().numpy().reshape(-1).astype(np.float64))
        z_np = z.detach().cpu().numpy().reshape(-1).astype(np.float64)
    pearson = float(np.corrcoef(z_np, v)[0, 1]) if z_np.size > 1 else 0.0
    if not math.isfinite(pearson):
        # degenerate constant value head -> report 0 (uncorrelated)
        pearson = 0.0
    return {
        "n": int(s.shape[0]),
        "top1": float(top1),
        "top5": float(top5),
        "policy_ce": ce,
        "value_mse": mse,
        "pearson": pearson,
    }


def random_baseline(batch: dict, seed: int = RANDOM_BASE_SEED) -> dict:
    """'Untrained floor': uniform-policy random guesses on the same eval set.

    Empirical top-1 / top-5 from seeded uniform draws over the ``N*N+1``
    moves; CE, MSE and Pearson are exact: ``ln(D)``, ``mean(z^2)`` (=1 for
    +-1 outcomes) and 0 (uncorrelated). Deterministic per ``seed``.
    """
    n = int(batch["s"].shape[0])
    D = int(batch["s"].shape[-1]) ** 2 + 1
    rng = np.random.default_rng(int(seed))
    top1 = float((rng.integers(0, D, size=n) == batch["pi"]).mean())
    top5 = float(
        (rng.integers(0, D, size=(n, 5)) == batch["pi"][:, None])
        .any(axis=1)
        .mean()
    )
    z = batch["z"].astype(np.float64)
    return {
        "n": n,
        "D": D,
        "top1": top1,
        "top5": top5,
        "policy_ce": float(math.log(D)),
        "value_mse": float(np.mean(z * z)),
        "pearson": 0.0,
    }


def run_human_match(
    checkpoint: "str | Path",
    data_dir: "str | Path",
    samples: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Load the net, sample a fixed eval set, score it + the random floor."""
    t0 = time.perf_counter()
    model, arch, step = load_eval_model(checkpoint, device)
    batch = sample_eval_set(data_dir, samples, seed)
    metrics = evaluate_model(model, batch, device)
    floor = random_baseline(batch, seed=seed ^ RANDOM_BASE_SEED)
    return {
        "mode": "human-match",
        "checkpoint": str(checkpoint),
        "arch": arch,
        "global_step": step,
        "data_dir": str(data_dir),
        "eval_samples": int(samples),
        "eval_seed": int(seed),
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "model": metrics,
        "random_baseline": floor,
    }


# ---------------------------------------------------------------------------
# bench: checkpoint vs a baseline (or a second checkpoint) over MCTS games
# ---------------------------------------------------------------------------

def run_bench(
    checkpoint: "str | Path",
    opponent: "str | Path | None",
    cfg: dict,
    *,
    games: int,
    sims: int,
    size: "int | None",
    komi: "float | None",
    virtual_loss: "int | None",
    max_moves: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Play ``games`` MCTS games of the checkpoint net vs a baseline.

    ``opponent`` is ``None`` for the "before training" baseline: a fresh
    random-init ``create_model`` of the checkpoint's own arch. Colours
    alternate (checkpoint is black on even game indices, white on odd -- komi
    7.5 favours white, so alternating keeps the comparison fair, exactly the
    todo-15 gate protocol). Games reuse :func:`play_eval_game`: no Dirichlet
    root noise, ``tau=0`` (argmax), ``torch.no_grad`` batched inference.
    """
    model_a, arch_a, step_a = load_eval_model(checkpoint, device)
    if opponent is not None:
        model_b, arch_b, step_b = load_eval_model(opponent, device)
        if int(arch_b["board_size"]) != int(arch_a["board_size"]):
            raise SystemExit(
                f"ERROR: --opponent board size {arch_b['board_size']} != "
                f"checkpoint board size {arch_a['board_size']}"
            )
        opp = {"checkpoint": str(opponent), "arch": arch_b,
               "global_step": step_b}
    else:
        model_b = create_model(
            int(arch_a["blocks"]), int(arch_a["channels"]),
            int(arch_a["board_size"]),
        ).to(device)  # random-init 'before training' baseline (default init)
        opp = {"checkpoint": None, "arch": dict(arch_a),
               "note": "untrained random-init baseline (create_model default init)"}

    size = int(size if size is not None else model_a.board_size)
    if int(size) != int(model_a.board_size):
        raise SystemExit(
            f"ERROR: --board-size {size} != checkpoint board size "
            f"{model_a.board_size}; the net's feature encoder is fixed to its "
            f"own board size"
        )
    komi = float(komi if komi is not None else cfg.get("komi", 7.5))
    virtual_loss = int(
        virtual_loss if virtual_loss is not None
        else cfg.get("virtual_loss", 3)
    )
    games = int(games)
    sims = int(sims)
    records: list[dict] = []
    t0 = time.perf_counter()
    for i in range(games):
        seed_i = int(seed) + i
        if i % 2 == 0:  # checkpoint plays black
            rec = play_eval_game(
                model_a, model_b, sims, size=size, komi=komi, seed=seed_i,
                virtual_loss=virtual_loss, max_moves=int(max_moves),
            )
            rec["a_color"] = "B"
            a_won = rec["winner"] == "B"
        else:  # checkpoint plays white
            rec = play_eval_game(
                model_b, model_a, sims, size=size, komi=komi, seed=seed_i,
                virtual_loss=virtual_loss, max_moves=int(max_moves),
            )
            rec["a_color"] = "W"
            a_won = rec["winner"] == "W"
        rec["a_won"] = bool(a_won)
        records.append(rec)
    wall = time.perf_counter() - t0

    a_wins = sum(1 for r in records if r["a_won"])
    draws = sum(1 for r in records if r["winner"] is None)
    b_wins = games - a_wins - draws
    moves = [int(r["moves"]) for r in records]
    return {
        "mode": "bench",
        "checkpoint": str(checkpoint),
        "arch_a": arch_a,
        "global_step_a": step_a,
        "opponent": opp,
        "games": games,
        "sims": sims,
        "board_size": size,
        "komi": komi,
        "virtual_loss": virtual_loss,
        "seed": int(seed),
        "wall_time_s": round(wall, 2),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "winrate_a": round(a_wins / games, 4) if games else 0.0,
        "winrate_b": round(b_wins / games, 4) if games else 0.0,
        "avg_game_length": round(float(np.mean(moves)), 1) if moves else 0.0,
        "games_detail": records,
    }


# ---------------------------------------------------------------------------
# human-readable report writer
# ---------------------------------------------------------------------------

def write_report(
    path: "str | Path",
    human: dict,
    bench: dict,
    *,
    verdict: "str | None" = None,
) -> str:
    """Write a human-readable summary of both modes' numbers to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h = human.get("model", {})
    hb = human.get("random_baseline", {})
    b = bench

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("P8 acceptance evaluation report -- omigamax pretrained net")
    lines.append("=" * 72)
    lines.append(f"generated : {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"checkpoint: {human.get('checkpoint', bench.get('checkpoint', '?'))}")
    arch = human.get("arch") or bench.get("arch_a") or {}
    lines.append(
        f"arch      : blocks={arch.get('blocks')} channels={arch.get('channels')} "
        f"board={arch.get('board_size')}"
    )
    lines.append("")

    lines.append("[1] human-match -- SL accuracy on a held-out eval set")
    lines.append("-" * 72)
    lines.append(
        f"  eval set : {human.get('eval_samples', '?')} positions sampled on "
        f"the fly from {human.get('data_dir', '?')} with fixed seed "
        f"{human.get('eval_seed', '?')} (no files written to data/pretrain)"
    )
    lines.append(
        f"  model    : top-1 {h.get('top1', float('nan'))*100:.2f}%   "
        f"top-5 {h.get('top5', float('nan'))*100:.2f}%   "
        f"policy CE {h.get('policy_ce', float('nan')):.4f}   "
        f"value MSE {h.get('value_mse', float('nan')):.4f}   "
        f"value corr {h.get('pearson', float('nan')):+.3f}"
    )
    lines.append(
        f"  random   : top-1 {hb.get('top1', float('nan'))*100:.2f}%   "
        f"top-5 {hb.get('top5', float('nan'))*100:.2f}%   "
        f"policy CE {hb.get('policy_ce', float('nan')):.4f}   "
        f"value MSE {hb.get('value_mse', float('nan')):.4f}   "
        f"value corr {hb.get('pearson', float('nan')):+.3f}   "
        f"(uniform floor over {hb.get('D', '?')} moves)"
    )
    gap = h.get("top1", 0.0) - hb.get("top1", 0.0)
    lines.append(f"  top-1 lift over random floor: {gap*100:+.2f} pp")
    lines.append("")

    lines.append("[2] bench -- checkpoint vs baseline (MCTS, alternating colors)")
    lines.append("-" * 72)
    opp = b.get("opponent", {})
    lines.append(
        f"  opponent : {opp.get('checkpoint') or opp.get('note', 'baseline')}"
    )
    lines.append(
        f"  protocol : {b.get('games', '?')} games, {b.get('sims', '?')} sims/move, "
        f"board {b.get('board_size', '?')}, komi {b.get('komi', '?')}, "
        f"virtual-loss {b.get('virtual_loss', '?')}, no Dirichlet noise, tau=0"
    )
    lines.append(
        f"  result   : checkpoint {b.get('a_wins', '?')} wins / "
        f"{b.get('b_wins', '?')} losses / {b.get('draws', '?')} draws "
        f"-> win rate {b.get('winrate_a', 0.0)*100:.1f}% "
        f"(baseline {b.get('winrate_b', 0.0)*100:.1f}%)"
    )
    lines.append(
        f"  game len : avg {b.get('avg_game_length', 0.0):.0f} moves "
        f"across {b.get('games', 0)} games"
    )
    lines.append("")

    if verdict:
        lines.append(f"verdict: {verdict}")
        lines.append("")

    lines.append(
        "note: the full acceptance run on the trained checkpoint is deferred "
        "until the P6 pretraining run finishes (GPU busy); this report "
        "summarizes the P8 code-phase evaluation."
    )
    text = "\n".join(lines) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    import os

    os.replace(tmp, path)
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="omigamax.cli.evaluate",
        description="P8 acceptance evaluation of the pretrained net: "
                    "human-match (SL accuracy vs human moves), bench (MCTS "
                    "games vs a baseline), report (combined text summary). "
                    "CPU-only; arch always read from the checkpoint.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint path (default models/latest.pt)")
    common.add_argument("--device", type=str, default=None,
                        help="torch device (default: cpu -- P8 is CPU-safe)")

    p_hm = sub.add_parser(
        "human-match", parents=[common],
        help="SL accuracy on a held-out eval set sampled from data/pretrain")
    p_hm.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                      help=f"chunk corpus dir (default {DEFAULT_DATA_DIR})")
    p_hm.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                      help=f"eval positions (default {DEFAULT_SAMPLES:,}; "
                           f"CPU-safe)")
    p_hm.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED,
                      help="fixed eval seed (default 0x5EED -- held-out vs "
                           "the P5 training seeds)")
    p_hm.add_argument("--evidence", type=str, default=str(DEFAULT_HUMAN_JSON),
                      help=f"JSON evidence path (default {DEFAULT_HUMAN_JSON})")

    p_b = sub.add_parser(
        "bench", parents=[common],
        help="MCTS games of the checkpoint vs a random-init baseline "
             "(or --opponent checkpoint), colors alternating")
    p_b.add_argument("--opponent", type=str, default=None,
                     help="second checkpoint; default: untrained random-init "
                          "net of the checkpoint's arch")
    p_b.add_argument("--games", type=int, default=DEFAULT_GAMES,
                     help=f"games (default {DEFAULT_GAMES})")
    p_b.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                     help=f"MCTS sims/move (default {DEFAULT_SIMS})")
    p_b.add_argument("--board-size", type=int, default=None,
                     help="board edge (default: the checkpoint's arch)")
    p_b.add_argument("--komi", type=float, default=None,
                     help="komi on white (default: config komi=7.5)")
    p_b.add_argument("--virtual-loss", type=int, default=None,
                     help="virtual loss (default: config virtual_loss=3)")
    p_b.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                     help=f"move cap per game (default {DEFAULT_MAX_MOVES})")
    p_b.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED,
                     help="master seed; games use seed + index")
    p_b.add_argument("--evidence", type=str, default=str(DEFAULT_BENCH_JSON),
                     help=f"JSON evidence path (default {DEFAULT_BENCH_JSON})")

    p_r = sub.add_parser(
        "report",
        help="write the combined human-readable report from both modes' "
             "evidence JSONs")
    p_r.add_argument("--human-json", type=str, default=str(DEFAULT_HUMAN_JSON),
                     help=f"human-match JSON (default {DEFAULT_HUMAN_JSON})")
    p_r.add_argument("--bench-json", type=str, default=str(DEFAULT_BENCH_JSON),
                     help=f"bench JSON (default {DEFAULT_BENCH_JSON})")
    p_r.add_argument("--report", type=str, default=str(DEFAULT_REPORT),
                     help=f"output text (default {DEFAULT_REPORT})")
    return ap


def _write_json(path: str, result: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return str(path)


def _print_human(result: dict) -> None:
    arch = result["arch"]
    print("=== omigamax P8 human-match (SL accuracy) ===", flush=True)
    print(f"checkpoint: {result['checkpoint']} (step {result['global_step']}, "
          f"arch {arch['blocks']}/{arch['channels']}/{arch['board_size']})",
          flush=True)
    print(f"eval set  : {result['eval_samples']:,} positions from "
          f"{result['data_dir']} (seed {result['eval_seed']})", flush=True)
    m = result["model"]
    b = result["random_baseline"]
    print(f"model top-1: {m['top1']*100:.2f}%  top-5: {m['top5']*100:.2f}%  "
          f"policy CE: {m['policy_ce']:.4f}  value MSE: {m['value_mse']:.4f}  "
          f"value corr: {m['pearson']:+.3f}", flush=True)
    print(f"random top-1: {b['top1']*100:.2f}%  top-5: {b['top5']*100:.2f}%  "
          f"policy CE: {b['policy_ce']:.4f}  value MSE: {b['value_mse']:.4f}  "
          f"(uniform floor over {b['D']} moves)", flush=True)
    print(f"top-1 lift: {(m['top1'] - b['top1'])*100:+.2f} pp   "
          f"({result['wall_time_s']}s)", flush=True)


def _print_bench(result: dict) -> None:
    print("=== omigamax P8 bench (checkpoint vs baseline) ===", flush=True)
    opp = result["opponent"]
    print(f"checkpoint: {result['checkpoint']} "
          f"(step {result['global_step_a']}, "
          f"arch {result['arch_a']['blocks']}/"
          f"{result['arch_a']['channels']}/"
          f"{result['arch_a']['board_size']})", flush=True)
    print(f"opponent  : {opp.get('checkpoint') or opp.get('note')}",
          flush=True)
    print(f"protocol  : {result['games']} games, {result['sims']} sims/move, "
          f"board {result['board_size']}, komi {result['komi']}, "
          f"virtual-loss {result['virtual_loss']}, no noise, tau=0",
          flush=True)
    print(f"result    : checkpoint {result['a_wins']} - {result['b_wins']} - "
          f"{result['draws']} -> {result['winrate_a']*100:.1f}% vs baseline "
          f"{result['winrate_b']*100:.1f}%", flush=True)
    print(f"game len  : avg {result['avg_game_length']:.0f} moves  "
          f"({result['wall_time_s']}s)", flush=True)


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    # config/default.yaml is read, never modified (P7 constraint).
    cfg = load_config(None)
    # report has no --device; only the checkpoint-loading modes do.
    device = torch.device(
        getattr(args, "device", None) if getattr(args, "device", None) is not None
        else "cpu"
    )

    if args.command == "human-match":
        result = run_human_match(
            args.checkpoint or str(latest_checkpoint_path()),
            args.data_dir, args.samples, args.seed, device,
        )
        _print_human(result)
        _write_json(args.evidence, result)
        print(f"evidence written: {args.evidence}", flush=True)
        return 0

    if args.command == "bench":
        result = run_bench(
            args.checkpoint or str(latest_checkpoint_path()),
            args.opponent, cfg,
            games=args.games, sims=args.sims, size=args.board_size,
            komi=args.komi, virtual_loss=args.virtual_loss,
            max_moves=args.max_moves, seed=args.seed, device=device,
        )
        _print_bench(result)
        _write_json(args.evidence, result)
        print(f"evidence written: {args.evidence}", flush=True)
        return 0

    # report
    human = json.loads(Path(args.human_json).read_text(encoding="utf-8"))
    bench = json.loads(Path(args.bench_json).read_text(encoding="utf-8"))
    path = write_report(args.report, human, bench)
    print(f"report written: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
