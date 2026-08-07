"""Evaluation gate and ELO recording (todo 15).

Per the plan (todo 15, authoritative) and AGZ (Nature 550, 2017, Methods):

* the *candidate* network (``models/latest.pt`` -- the just-trained net) plays
  ``eval_games`` full games against the current *best* network
  (``models/best.pt``) at ``eval_sims`` MCTS simulations per move, alternating
  colours with the same komi (odd ``eval_games`` = 21 keeps the colour
  balance fair -- komi 7.5 favours white);
* **evaluation discipline** (plan, Oracle #8): NO Dirichlet root noise and
  ``tau = 0`` (argmax) move selection -- Dirichlet is a self-play exploration
  device and would pollute the 55% gate. The existing search
  (:func:`omigamax.mcts.run_search` + the todo-11 batched evaluator) and the
  AGZ temperature selection (:func:`omigamax.mcts.sample_action`) are reused
  -- nothing is reimplemented here;
* the **gate**: the candidate replaces ``best.pt`` iff its win rate is
  ``>= replace_threshold`` (default 0.55) -- the boundary included (plan:
  含等于替换);
* **first-eval bootstrap** (plan, Oracle G1): when ``best.pt`` does not exist
  (the training start), random-init weights are written to ``best.pt`` as the
  baseline opponent and the candidate is then evaluated against them -- so the
  very first gate decision is "candidate vs a random net";
* every evaluation appends a JSONL line to ``logs/eval_history.jsonl`` with
  the win rate and the ELO delta (K=32 simple Elo update, initial 0) -- the
  plan's *internal* gate metric, kept separate from the external vs-KataGo
  ELO of todo 20 (plan: 指标分离).

ELO estimate for a win rate ``p`` -- the standard Elo rating difference
(plan References: Wikipedia "Elo rating system")::

    ELO(p) = 400 * log10(p / (1 - p))

``p`` is clamped to ``[1e-6, 1 - 1e-6]`` so a degenerate all-wins / all-losses
evaluation maps to +/-2400 instead of ``+/-inf``. The *running* rating tracked
in ``eval_history.jsonl`` uses the K=32 update
``R' = R + K * (score - E(R))`` with expected score
``E(R) = 1 / (1 + 10^(-R/400))`` and starts at 0.

Design notes:

* evaluation games never enter the replay buffer (plan Must-NOT: 不引入自对弈
  样本复用);
* both networks (candidate + best, each ~13 MB at b10c128) live in memory at
  once (plan: 6GB 不容两网络同驻 is not a real constraint), each side using
  its own weights -- no tree reuse across moves (the simple, correct baseline
  of todo 12/13).

Usage::

    uv run python -m omigamax.train.evaluate --games 9          # plan acceptance
    uv run python -m omigamax.train.evaluate --games 5 --sims 40  # fast demo
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts import BatchedNetworkEvaluator, make_root, run_search, sample_action
from omigamax.network.features import pass_index
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board
from omigamax.train.loss import make_sgd_optimizer
from omigamax.train.train import (
    DEFAULT_CHECKPOINT_DIR,
    latest_checkpoint_path,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)

# Plan defaults (todo 15; every value is config-driven when available).
DEFAULT_MAX_MOVES = 1000           # eval timeout protection (todo-13 style)
DEFAULT_HISTORY_PATH = "logs/eval_history.jsonl"
DEFAULT_EVIDENCE = ".omo/evidence/omigamax-go/task-15-eval.json"
BEST_NAME = "best.pt"
DEFAULT_THRESHOLD = 0.55           # replace_threshold (config)
DEFAULT_EVAL_GAMES = 21
DEFAULT_EVAL_SIMS = 200
ELO_K = 32.0                       # plan's K=32 simple Elo update
ELO_SCALE = 400.0                  # standard Elo rating scale
ELO_P_CLAMP = 1e-6                 # win-rate clamp -> |ELO| <= ~2400


# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------

def elo_from_winrate(winrate: float) -> float:
    """Standard Elo rating difference from a win rate ``p``.

    ``ELO(p) = 400 * log10(p / (1 - p))`` (plan References: Wikipedia). Known
    values: ``0.5 -> 0``, ``0.55 -> ~34.9``, ``0.75 -> ~190.9``. ``p`` is
    clamped to ``[1e-6, 1 - 1e-6]`` so degenerate evaluations (0% / 100%) map
    to ``-2400`` / ``+2400`` instead of ``+/-inf``.
    """
    p = float(winrate)
    p = min(max(p, ELO_P_CLAMP), 1.0 - ELO_P_CLAMP)
    return ELO_SCALE * math.log10(p / (1.0 - p))


def expected_score(rating: float, opponent_rating: float = 0.0) -> float:
    """Expected score of ``rating`` vs ``opponent_rating`` under the Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((float(opponent_rating) - float(rating)) / ELO_SCALE))


def update_elo(
    rating: float,
    score: float,
    opponent_rating: float = 0.0,
    k: float = ELO_K,
) -> float:
    """One K-factor Elo update: ``R' = R + K * (score - expected)`` (initial 0)."""
    return float(rating) + float(k) * (
        float(score) - expected_score(float(rating), float(opponent_rating))
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def gate_decision(candidate_wins: int, games: int, threshold: float) -> bool:
    """Replace-the-best gate: candidate wins iff ``candidate_wins / games >= threshold``.

    The boundary is included (plan: 0.55 阈值含等于替换): ``12/21 = 0.5714``
    replaces, ``11/21 = 0.5238`` does not, and ``11/20 = 0.55`` replaces.
    """
    games = int(games)
    if games <= 0:
        return False
    return (int(candidate_wins) / games) >= float(threshold)


# ---------------------------------------------------------------------------
# one evaluation game (no noise, tau=0)
# ---------------------------------------------------------------------------

def play_eval_game(
    black_net: torch.nn.Module,
    white_net: torch.nn.Module,
    sims: int,
    *,
    size: int = 19,
    komi: float = 7.5,
    seed: int = 0,
    virtual_loss: int = 3,
    max_moves: int = DEFAULT_MAX_MOVES,
    leaf_batch: "int | None" = None,
) -> dict:
    """Play one evaluation game, ``black_net`` vs ``white_net``.

    Evaluation discipline (plan, Oracle #8): the search runs with **no
    Dirichlet root noise** (``run_search``'s ``dirichlet_alpha`` stays
    ``None``) and every move is ``sample_action(root, tau=0.0)`` -- argmax
    (ties resolved uniformly at random, AGZ ``tau -> 0``). Both networks are
    put in ``eval()`` mode (inference discipline, plan G3); the batched leaf
    evaluator wraps every forward in ``torch.no_grad()``. Two consecutive
    passes terminate (Tromp-Taylor), ``max_moves`` force-terminates and scores
    as a timeout protection.

    Returns a per-game record dict: ``seed``, ``winner`` ("B"/"W"/None),
    ``result``, ``forced_terminal``, ``moves``.
    """
    size = int(size)
    komi = float(komi)
    sims = int(sims)
    virtual_loss = int(virtual_loss)
    max_moves = int(max_moves)
    rng = np.random.default_rng(int(seed))
    black_net.eval()
    white_net.eval()
    black_eval = BatchedNetworkEvaluator(black_net)
    white_eval = BatchedNetworkEvaluator(white_net)
    board = Board(size)
    moves = 0
    while not board.is_terminal() and moves < max_moves:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        evaluator = black_eval if color == BLACK else white_eval
        root = make_root(board)
        run_search(
            root, None, sims, evaluator=evaluator, komi=komi,
            virtual_loss=virtual_loss, batch_size=leaf_batch,
        )  # dirichlet_alpha stays None -> no root noise
        action = sample_action(root, 0.0, rng=rng)  # tau=0 -> argmax
        if action == pass_index(size):
            board.pass_move(color)
        else:
            board.play((action // size, action % size), color)
        moves += 1
    winner = board.winner(komi)
    return {
        "seed": int(seed),
        "winner": winner,
        "result": board.result_string(komi),
        "forced_terminal": not board.is_terminal(),
        "moves": moves,
    }


# ---------------------------------------------------------------------------
# the evaluation match
# ---------------------------------------------------------------------------

def run_evaluation(
    candidate_net: torch.nn.Module,
    best_net: torch.nn.Module,
    cfg: dict,
    *,
    games: "int | None" = None,
    sims: "int | None" = None,
    size: "int | None" = None,
    komi: "float | None" = None,
    virtual_loss: "int | None" = None,
    threshold: "float | None" = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    seed: int = 0,
) -> dict:
    """Play the evaluation match: candidate vs best, colours alternating.

    ``games`` (default config ``eval_games`` = 21) full games at ``sims``
    (default config ``eval_sims`` = 200) simulations per move. The candidate
    is black on even game indices and white on odd indices (komi 7.5 favours
    white, so alternating keeps the comparison fair -- same protocol as todo
    12). Seeds are ``seed + game_index`` (deterministic per game).

    Returns a report dict: per-game records, candidate wins / draws, win rate,
    the standard ELO difference (:func:`elo_from_winrate`) and the gate
    decision vs ``replace_threshold`` (default config = 0.55).
    """
    games = int(games if games is not None else cfg.get("eval_games", DEFAULT_EVAL_GAMES))
    sims = int(sims if sims is not None else cfg.get("eval_sims", DEFAULT_EVAL_SIMS))
    size = int(size if size is not None else cfg.get("board_size", 19))
    komi = float(komi if komi is not None else cfg.get("komi", 7.5))
    virtual_loss = int(
        virtual_loss if virtual_loss is not None else cfg.get("virtual_loss", 3)
    )
    threshold = float(
        threshold if threshold is not None else cfg.get("replace_threshold", DEFAULT_THRESHOLD)
    )
    candidate_net.eval()
    best_net.eval()
    t0 = time.perf_counter()
    records: list[dict] = []
    candidate_wins = 0
    draws = 0
    for i in range(games):
        seed_i = int(seed) + i
        if i % 2 == 0:  # candidate plays black
            rec = play_eval_game(
                candidate_net, best_net, sims, size=size, komi=komi,
                seed=seed_i, virtual_loss=virtual_loss, max_moves=max_moves,
            )
            rec["candidate_color"] = "B"
            if rec["winner"] == "B":
                candidate_wins += 1
        else:  # candidate plays white
            rec = play_eval_game(
                best_net, candidate_net, sims, size=size, komi=komi,
                seed=seed_i, virtual_loss=virtual_loss, max_moves=max_moves,
            )
            rec["candidate_color"] = "W"
            if rec["winner"] == "W":
                candidate_wins += 1
        if rec["winner"] is None:
            draws += 1
        records.append(rec)
    wall = time.perf_counter() - t0
    winrate = candidate_wins / games
    return {
        "games": games,
        "sims": sims,
        "board_size": size,
        "komi": komi,
        "virtual_loss": virtual_loss,
        "threshold": threshold,
        "candidate_wins": candidate_wins,
        "draws": draws,
        "winrate": winrate,
        "elo_diff": round(elo_from_winrate(winrate), 3),
        "replaced": gate_decision(candidate_wins, games, threshold),
        "wall_time_s": wall,
        "games_detail": records,
    }


# ---------------------------------------------------------------------------
# best.pt bootstrap + history persistence
# ---------------------------------------------------------------------------

def best_checkpoint_path(checkpoint_dir: "str | Path" = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Path of ``models/best.pt`` (the evaluation-gate checkpoint)."""
    return Path(checkpoint_dir) / BEST_NAME


def ensure_best_model(
    best_path: "str | Path",
    arch: dict,
    cfg: dict,
    device: torch.device,
) -> "tuple[torch.nn.Module, bool]":
    """Load ``best.pt`` or bootstrap it with random-init weights (plan Oracle G1).

    When ``best_path`` does not exist (the training start), a fresh
    random-init network of the candidate's architecture is saved to
    ``best.pt`` -- with SGD optimizer state and ``global_step=0`` in the
    todo-14 checkpoint format -- and returned as the baseline opponent, so the
    very first gate decision is "candidate vs a random net".

    Returns ``(model, bootstrapped)``.
    """
    best_path = Path(best_path)
    if best_path.exists():
        ckpt = load_checkpoint(best_path)
        a = ckpt["arch"]
        model = create_model(
            int(a["blocks"]), int(a["channels"]), int(a["board_size"])
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, False
    model = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    optimizer = make_sgd_optimizer(
        model,
        lr=float(cfg.get("lr", 0.2)),
        momentum=float(cfg.get("momentum", 0.9)),
        l2=float(cfg.get("l2", 1e-4)),
    )
    save_checkpoint(best_path, model, optimizer, global_step=0, config=cfg)
    return model, True


def read_last_elo(history_path: "str | Path") -> float:
    """The last recorded running ELO (``0.0`` when the file is absent/empty)."""
    history_path = Path(history_path)
    if not history_path.exists():
        return 0.0
    last: "str | None" = None
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return 0.0
    try:
        return float(json.loads(last).get("elo", 0.0))
    except (ValueError, TypeError):
        return 0.0


def append_eval_history(entry: dict, history_path: "str | Path") -> str:
    """Append one JSONL entry to ``history_path`` (creates the parent dir)."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(history_path)


# ---------------------------------------------------------------------------
# full orchestration: load -> evaluate -> gate -> replace -> record
# ---------------------------------------------------------------------------

def evaluate_and_gate(
    candidate_path: "str | Path",
    best_path: "str | Path",
    cfg: dict,
    *,
    games: "int | None" = None,
    sims: "int | None" = None,
    size: "int | None" = None,
    komi: "float | None" = None,
    virtual_loss: "int | None" = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    seed: int = 0,
    device: "torch.device | None" = None,
    history_path: "str | Path | None" = None,
) -> dict:
    """Run the full evaluation gate: candidate vs best, record, replace.

    1. load the candidate checkpoint (``models/latest.pt`` by default) --
       weights + SGD optimizer state + ``global_step``;
    2. ensure ``best.pt`` (bootstrap a random-init baseline when missing);
    3. run the evaluation match (no noise, ``tau = 0``, colours alternating);
    4. decide the gate (win rate ``>= replace_threshold``);
    5. on replace, write the candidate's full checkpoint (weights + optimizer
       + ``global_step``) to ``best.pt``;
    6. append a JSONL history entry (win rate + K=32 ELO update) and return
       the full report dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(candidate_path)
    arch = ckpt["arch"]
    candidate = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    optimizer = make_sgd_optimizer(
        candidate,
        lr=float(cfg.get("lr", 0.2)),
        momentum=float(cfg.get("momentum", 0.9)),
        l2=float(cfg.get("l2", 1e-4)),
    )
    global_step = restore_from_checkpoint(ckpt, candidate, optimizer)

    best, bootstrapped = ensure_best_model(best_path, arch, cfg, device)
    report = run_evaluation(
        candidate, best, cfg,
        games=games, sims=sims, size=size, komi=komi,
        virtual_loss=virtual_loss, max_moves=max_moves, seed=seed,
    )

    best_written = None
    if report["replaced"]:
        best_written = save_checkpoint(
            best_path, candidate, optimizer, global_step=global_step, config=cfg,
        )

    elo_prev = read_last_elo(history_path) if history_path is not None else 0.0
    elo_new = update_elo(elo_prev, report["winrate"], 0.0)
    elo_update = {
        "elo_before": round(elo_prev, 3),
        "elo_delta": round(elo_new - elo_prev, 3),
        "elo": round(elo_new, 3),
        "elo_diff": report["elo_diff"],
    }
    entry = {
        "event": "evaluate_gate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": str(candidate_path),
        "best": str(best_path),
        "global_step": global_step,
        "bootstrapped_best": bootstrapped,
        "games": report["games"],
        "sims": report["sims"],
        "candidate_wins": report["candidate_wins"],
        "winrate": round(report["winrate"], 6),
        "threshold": report["threshold"],
        "replaced_best": bool(report["replaced"]),
        "best_written": best_written,
        **elo_update,
    }
    if history_path is not None:
        append_eval_history(entry, history_path)

    return {
        "todo": 15,
        "device": str(device),
        "protocol": {
            "candidate": str(candidate_path),
            "best": str(best_path),
            "games": report["games"],
            "sims": report["sims"],
            "board_size": report["board_size"],
            "komi": report["komi"],
            "virtual_loss": report["virtual_loss"],
            "replace_threshold": report["threshold"],
            "seed": seed,
            "max_moves": max_moves,
            "arch": arch,
            "candidate_global_step": global_step,
            "bootstrapped_best": bootstrapped,
        },
        "match": {k: v for k, v in report.items() if k != "games_detail"},
        "games_detail": report["games_detail"],
        "elo_update": elo_update,
        "history": None if history_path is None else str(history_path),
        "replaced_best": bool(report["replaced"]),
        "best_written": best_written,
        "accepted": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="omigamax todo-15 evaluation gate: candidate "
                    "(models/latest.pt) vs best.pt -- no Dirichlet noise, "
                    "tau=0 (argmax); win rate >= replace_threshold replaces "
                    "best.pt; win rate + K=32 ELO recorded to "
                    "logs/eval_history.jsonl."
    )
    parser.add_argument("--games", type=int, default=None,
                        help="eval games (default: config eval_games=21; "
                             "plan acceptance uses 9)")
    parser.add_argument("--sims", type=int, default=None,
                        help="MCTS simulations per move (default: config "
                             "eval_sims=200; pass a small value for fast "
                             "demos)")
    parser.add_argument("--board-size", type=int, default=None,
                        help="board edge (default: config board_size=19)")
    parser.add_argument("--komi", type=float, default=None,
                        help="komi on white (default: config komi=7.5)")
    parser.add_argument("--virtual-loss", type=int, default=None,
                        help="virtual loss (default: config virtual_loss=3)")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"move cap per game (default {DEFAULT_MAX_MOVES})")
    parser.add_argument("--candidate", type=str, default=None,
                        help="candidate checkpoint (default: models/latest.pt)")
    parser.add_argument("--best", type=str, default=None,
                        help="best checkpoint (default: models/best.pt; "
                             "random-init bootstrapped when absent)")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help=f"checkpoint dir (default {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--seed", type=int, default=0,
                        help="master random seed (games use seed + index)")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--history", type=str, default=DEFAULT_HISTORY_PATH,
                        help=f"eval-history JSONL (default {DEFAULT_HISTORY_PATH}; "
                             "pass '' to disable)")
    parser.add_argument("--evidence", type=str, default=DEFAULT_EVIDENCE,
                        help=f"write the result JSON here (default {DEFAULT_EVIDENCE})")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    candidate = args.candidate or str(latest_checkpoint_path(args.checkpoint_dir))
    best = args.best or str(best_checkpoint_path(args.checkpoint_dir))
    history = None if args.history == "" else args.history

    result = evaluate_and_gate(
        candidate, best, cfg,
        games=args.games, sims=args.sims, size=args.board_size, komi=args.komi,
        virtual_loss=args.virtual_loss, max_moves=args.max_moves, seed=args.seed,
        device=device, history_path=history,
    )

    _print_report(result)

    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0


def _print_report(result: dict) -> None:
    proto = result["protocol"]
    match = result["match"]
    print("=== omigamax evaluation gate (todo 15) ===", flush=True)
    print(f"device: {result['device']}", flush=True)
    print(
        f"protocol: candidate={proto['candidate']} best={proto['best']} "
        f"games={proto['games']} sims={proto['sims']} "
        f"board={proto['board_size']} komi={proto['komi']} "
        f"virtual_loss={proto['virtual_loss']} "
        f"replace_threshold={proto['replace_threshold']} seed={proto['seed']}",
        flush=True,
    )
    if proto["bootstrapped_best"]:
        print(
            f"bootstrap: best.pt did not exist -> random-init baseline written "
            f"to {proto['best']} (plan Oracle G1)", flush=True
        )
    else:
        print("best.pt: loaded existing checkpoint", flush=True)
    for rec in result["games_detail"]:
        forced = " (max-moves forced)" if rec["forced_terminal"] else ""
        print(
            f"  game seed={rec['seed']}: candidate={rec['candidate_color']} "
            f"winner={rec['winner']} moves={rec['moves']} "
            f"result={rec['result']}{forced}", flush=True
        )
    print(
        f"match: {match['candidate_wins']}/{match['games']} candidate wins "
        f"(win rate {match['winrate']:.3f}) in {match['wall_time_s']:.1f}s",
        flush=True,
    )
    elo = result["elo_update"]
    print(
        f"ELO: diff(win-rate) = {elo['elo_diff']}  K=32 update "
        f"{elo['elo_before']} -> {elo['elo']} (delta {elo['elo_delta']})",
        flush=True,
    )
    decision = "REPLACE best.pt" if result["replaced_best"] else "KEEP best.pt"
    print(
        f"gate: win rate {match['winrate']:.3f} "
        f"{'>=' if result['replaced_best'] else '<'} "
        f"{match['threshold']} -> {decision}", flush=True
    )
    print(f"history: {result['history']}", flush=True)
    print("RESULT: PASS (exit 0)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
