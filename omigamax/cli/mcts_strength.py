"""Todo 12 acceptance harness: MCTS strength is monotonic in simulations.

Plays full games (two-pass terminal, Tromp-Taylor area scoring, komi 7.5)
between a *fixed-weight* policy-value network run at different MCTS
simulation budgets -- the plan's 40 / 200 / 800 sims ladder -- using the
existing search (``omigamax.mcts.run_search`` + batched leaf evaluation,
todos 9-11) and the rules engine (todos 3-4). No search / rules code is
reimplemented here.

Protocol (plan todo 12, authoritative): every pair of the three sim levels
plays ``--games`` (default 60) full games, same network, same komi,
alternating colours (先后手各半 -- komi 7.5 favours white, so the higher-sim
side plays black in exactly half the games). The acceptance matrix reports
the win rate of the higher-sim side per pair and asserts monotonicity
``P(800>200) > 0.5 and P(200>40) > 0.5``.

Fixed weight (plan): *random* init by default (``--seed``-seeded). If the
plan's fallback is taken, ``--smoke-train-steps N`` smoke-trains the
network N steps on synthetic random data (the plan's "先用随机数据冒烟训练
~500 步的权重作为固定网络重测") and the weight source is recorded in the
report. ``--weights <path>`` loads a pre-saved fixed ``state_dict``.

Move selection: the MCTS agent samples from the visit-count policy at
``--tau`` (default ``1.0`` -- proportional to visit counts, the todo-10 AGZ
temperature selection); the random-legal opponent plays uniformly at random
over legal moves + pass.

Time (the plan's own escape hatch, Oracle round-3 item 2): the plan
estimates the full 60-game protocol at 11-22 h @ 300-600 sims/s. The
measured rate on the RTX 3060 Laptop here is ~90-160 sims/s, so the full
protocol projects to tens of hours. ``--quick`` reduces the two 800-sim
pairings to 30 games each (the plan's exact quick reduction);
``--max-time MIN`` time-boxes the whole run (an in-flight game finishes;
partial coverage is recorded and the projected full/quick wall-times are
reported -- never silently skipped).

Every game is seeded deterministically
(``master_seed + pairing_index * 10000 + game_index``, pairing indices
assigned by a fixed counter so seeds are stable across runs) and its
per-game record (seed, agents, moves, result, wall time, move list) is
written to the evidence JSON. SGFs of every game are written to
``--sgf-dir`` (default ``logs/matches/todo12``) -- the plan's failure path
(manual review) and a normal by-product otherwise.

Usage::

    uv run python -m omigamax.cli.mcts_strength --games 60
    uv run python -m omigamax.cli.mcts_strength --quick --max-time 120 \\
        --evidence .omo/evidence/omigamax-go/task-12-strength.json
    uv run python -m omigamax.cli.mcts_strength --games 60 \\
        --random-baseline 10 --smoke-train-steps 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts import BatchedNetworkEvaluator, make_root, run_search, sample_action
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board
from omigamax.rules.sgf import export_sgf

# The plan's locked ladder (todo 12: 40 vs 200 vs 800 sims, 60 局/配对).
DEFAULT_LEVELS = (40, 200, 800)
DEFAULT_GAMES = 60
QUICK_800_PAIRING_GAMES = 30  # plan's --quick reduction for 800-sim pairings
DEFAULT_TAU = 1.0
DEFAULT_MAX_MOVES = 1000  # todo-13 style timeout protection
# Acceptance pairings in priority order (cheap acceptance pairing first, so a
# time-boxed run covers it before the expensive 800-sim pairings).
DEFAULT_PAIRINGS = ("40v200", "200v800", "40v800")
SEED_STRIDE = 10000  # per-pairing seed spacing


class _MCTSAgent:
    """An MCTS agent with a fixed simulation budget and visit-count sampling.

    Uses one :class:`BatchedNetworkEvaluator` (todo 11, ``leaf_batch`` from
    config) shared with the fixed network; every move is a fresh
    ``run_search`` from the current position (no cross-move tree reuse --
    the simple, correct baseline) followed by ``sample_action(tau)``.
    """

    def __init__(self, network, sims: int, tau: float, komi: float,
                 virtual_loss: int, size: int, rng) -> None:
        self.sims = int(sims)
        self.tau = float(tau)
        self.komi = float(komi)
        self.virtual_loss = int(virtual_loss)
        self.size = int(size)
        self.rng = rng
        self.evaluator = BatchedNetworkEvaluator(network)

    def play(self, board, color) -> None:
        root = make_root(board)
        run_search(
            root, None, self.sims, evaluator=self.evaluator,
            komi=self.komi, virtual_loss=self.virtual_loss,
        )
        action = sample_action(root, self.tau, rng=self.rng)
        if action == self.size * self.size:
            board.pass_move(color)
        else:
            board.play((action // self.size, action % self.size), color)


class _RandomAgent:
    """Uniform-random-legal opponent (pass is a legal move, played with
    probability ``1 / (legal points + 1)``)."""

    def __init__(self, size: int, rng) -> None:
        self.size = int(size)
        self.rng = rng

    def play(self, board, color) -> None:
        points = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if board.is_legal((r, c), color)
        ]
        idx = int(self.rng.integers(0, len(points) + 1))  # last slot = pass
        board.play(None if idx == len(points) else points[idx], color)


def _mcts_agent(network, sims, tau, komi, virtual_loss, size, rng):
    return _MCTSAgent(network, sims, tau, komi, virtual_loss, size, rng)


def play_game(black_agent, white_agent, size: int, komi: float, seed: int,
              max_moves: int) -> dict:
    """Play one full game; returns a per-game record dict.

    Two-pass terminal (``Board.is_terminal``), Tromp-Taylor scoring with
    ``komi`` on white; ``max_moves`` force-terminates and scores (marked
    ``forced_terminal``) as a timeout protection. ``Board.play`` raises on
    any illegal move, so legality is guaranteed by construction. The move
    list is kept so the game can be exported to SGF.
    """
    rng = np.random.default_rng(seed)
    board = Board(size)
    t0 = time.perf_counter()
    move_list: list[tuple] = []
    moves = 0
    while not board.is_terminal() and moves < max_moves:
        color = BLACK if len(board.moves) % 2 == 0 else WHITE
        agent = black_agent if color == BLACK else white_agent
        before = len(board.moves)
        agent.play(board, color)
        move_list.append((board.moves[before][0], color))
        moves += 1
    wall_time_s = time.perf_counter() - t0
    forced = not board.is_terminal()
    winner = board.winner(komi)
    return {
        "seed": seed,
        "moves": moves,
        "wall_time_s": wall_time_s,
        "winner": winner,  # "B" / "W" (komi 7.5 => no jigo)
        "result": board.result_string(komi),
        "forced_terminal": forced,
        "move_list": move_list,
    }


def _sgf_from_record(record: dict, size: int, komi: float,
                     black_name: str, white_name: str) -> str:
    """Export the game's SGF from its recorded move list."""
    board = Board(size)
    for move, color in record["move_list"]:
        board.play(move, color)
    return export_sgf(board, komi=komi, result=record["result"],
                      player_black=black_name, player_white=white_name)


def _write_sgf(sgf_dir: Path, name: str, record: dict, size: int, komi: float,
               black_name: str, white_name: str) -> str:
    sgf_dir.mkdir(parents=True, exist_ok=True)
    path = sgf_dir / f"{name}.sgf"
    path.write_text(
        _sgf_from_record(record, size, komi, black_name, white_name),
        encoding="utf-8",
    )
    return str(path)


def run_pairing(lo_sims, hi_sims, games: int, size: int, komi: float,
                network, tau: float, virtual_loss: int, base_seed: int,
                max_moves: int, sgf_dir: Path | None,
                pairing_time_s: float | None = None,
                ) -> tuple[list[dict], dict]:
    """Play ``games`` games between two sim levels, alternating colours.

    Game ``i`` even: ``lo_sims`` plays black (white has komi, so this is a
    handicap for lo -- balanced by game ``i`` odd where ``hi_sims`` plays
    black). ``pairing_time_s`` time-boxes the pairing (an in-flight game
    finishes; truncation is recorded). Returns (game records, pairing
    summary)."""
    records: list[dict] = []
    t_pair = time.perf_counter()
    truncated = False
    for i in range(games):
        if pairing_time_s is not None and (time.perf_counter() - t_pair) >= pairing_time_s:
            truncated = True
            break
        seed = base_seed + i
        if i % 2 == 0:
            black_sims, white_sims = lo_sims, hi_sims
        else:
            black_sims, white_sims = hi_sims, lo_sims
        black_agent = _mcts_agent(network, black_sims, tau, komi, virtual_loss,
                                  size, np.random.default_rng(seed))
        white_agent = _mcts_agent(network, white_sims, tau, komi, virtual_loss,
                                  size, np.random.default_rng(seed + 1))
        rec = play_game(black_agent, white_agent, size, komi, seed, max_moves)
        rec.update({
            "black_sims": black_sims,
            "white_sims": white_sims,
            "black_agent": "mcts",
            "white_agent": "mcts",
        })
        if sgf_dir is not None:
            rec["sgf"] = _write_sgf(
                sgf_dir, f"pair-{lo_sims}v{hi_sims}-game-{i:03d}", rec,
                size, komi, f"mcts{black_sims}", f"mcts{white_sims}",
            )
        records.append(rec)
    summary = _summarize_pairing(records, hi_sims)
    summary["truncated"] = truncated
    summary["planned_games"] = games
    return records, summary


def run_vs_random(sims, games: int, size: int, komi: float, network,
                  tau: float, virtual_loss: int, base_seed: int,
                  max_moves: int, sgf_dir: Path | None) -> tuple[list[dict], dict]:
    """Play the sim level vs the random-legal opponent, alternating colours."""
    records: list[dict] = []
    for i in range(games):
        seed = base_seed + i
        mcts_black = (i % 2 == 0)
        black_sims = sims if mcts_black else None
        white_sims = None if mcts_black else sims
        if mcts_black:
            black_agent = _mcts_agent(network, sims, tau, komi, virtual_loss,
                                      size, np.random.default_rng(seed))
            white_agent = _RandomAgent(size, np.random.default_rng(seed + 1))
        else:
            black_agent = _RandomAgent(size, np.random.default_rng(seed))
            white_agent = _mcts_agent(network, sims, tau, komi, virtual_loss,
                                      size, np.random.default_rng(seed + 1))
        rec = play_game(black_agent, white_agent, size, komi, seed, max_moves)
        rec.update({
            "black_sims": black_sims,
            "white_sims": white_sims,
            "black_agent": "mcts" if mcts_black else "random",
            "white_agent": "random" if mcts_black else "mcts",
        })
        if sgf_dir is not None:
            rec["sgf"] = _write_sgf(
                sgf_dir, f"random{sims}-game-{i:03d}", rec, size, komi,
                f"mcts{sims}" if mcts_black else "random",
                "random" if mcts_black else f"mcts{sims}",
            )
        records.append(rec)
    mcts_wins = 0
    for rec in records:
        mcts_color = "B" if rec["black_sims"] == sims else "W"
        if rec["winner"] == mcts_color:
            mcts_wins += 1
    return records, {
        "games": games,
        "mcts_wins": mcts_wins,
        "mcts_winrate": mcts_wins / games,
    }


def _summarize_pairing(records, hi_sims) -> dict:
    """Win rate of the higher-sim side (``hi_sims``) across ``records``."""
    games = len(records)
    if not games:
        return {"games": 0, "hi_wins": 0, "hi_winrate": None}
    hi_wins = 0
    for rec in records:
        hi_color = "B" if rec["black_sims"] == hi_sims else "W"
        if rec["winner"] == hi_color:
            hi_wins += 1
    return {"games": games, "hi_wins": hi_wins,
            "hi_winrate": hi_wins / games}


def _project_hours(per_game_times: list[float], games_target: int) -> float | None:
    """Project wall-hours for ``games_target`` games from measured per-game
    times (median, robust to the rare very-long game)."""
    if not per_game_times:
        return None
    return float(np.median(per_game_times)) * games_target / 3600.0


def _smoke_train(network, cfg, steps: int, device, seed: int,
                 save_path: Path) -> torch.nn.Module:
    """Plan fallback: smoke-train on synthetic random data (AGZ CE+MSE, SGD
    momentum, L2 via weight decay -- the todo-8 recipe) and save the weight."""
    from omigamax.train.loss import make_sgd_optimizer, train_step

    batch = int(cfg["batch_size"])
    lr, momentum, l2 = float(cfg["lr"]), float(cfg["momentum"]), float(cfg["l2"])
    fp16 = bool(cfg.get("fp16", False))
    optimizer = make_sgd_optimizer(network, lr, momentum, l2)
    size = int(cfg["board_size"])
    n_logits = size * size + 1
    torch.manual_seed(seed)
    losses: list[float] = []
    inputs = torch.randn(batch, 17, size, size, device=device)
    target_idx = torch.randint(0, n_logits, (batch,), device=device)
    pi = torch.zeros(batch, n_logits, device=device)
    pi[torch.arange(batch), target_idx] = 1.0
    z = torch.randint(0, 2, (batch, 1), device=device).float() * 2.0 - 1.0
    for _ in range(steps):
        losses.append(train_step(network, optimizer, inputs, pi, z, use_fp16=fp16))
    network.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"[smoke-train] {steps} steps: loss {losses[0]:.6f} -> "
          f"{losses[-1]:.6f}; saved {save_path}", flush=True)
    return network


def _load_weights(network: torch.nn.Module, path: "str | Path", device) -> str:
    """Load network weights from ``path`` into ``network``; return a label.

    Accepts both a raw state dict (the smoke-trained weights this harness
    itself emits) and a full training checkpoint (``models/best.pt``, which
    wraps the state dict under ``model_state_dict`` together with
    global_step/arch/config). Regression fix: loading a trained checkpoint
    previously raised ``Missing key(s) in state_dict`` because the whole
    checkpoint dict was fed to ``load_state_dict``.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) \
        else ckpt
    network.load_state_dict(sd)
    return f"loaded from {path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="omigamax todo-12 MCTS strength ladder "
                    "(win rate monotonic in simulations)."
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"games per pairing (default {DEFAULT_GAMES}; plan: 60)")
    parser.add_argument("--quick", action="store_true",
                        help="plan quick mode: 800-sim pairings -> 30 games")
    parser.add_argument("--pairings", type=str, default=",".join(DEFAULT_PAIRINGS),
                        help="pairings to run, comma-separated (default "
                             f"{','.join(DEFAULT_PAIRINGS)})")
    parser.add_argument("--random-baseline", type=int, default=0,
                        help="also play each sim level vs the random-legal "
                             "opponent for this many games (default 0)")
    parser.add_argument("--random-levels", type=str, default=None,
                        help="which levels get the random baseline, "
                             "comma-separated (default: all levels)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help=f"MCTS move-selection temperature (default {DEFAULT_TAU})")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"move cap per game (default {DEFAULT_MAX_MOVES})")
    parser.add_argument("--max-time", type=float, default=None,
                        help="global wall-clock budget in minutes; stops "
                             "starting new games when exceeded")
    parser.add_argument("--pairing-time", type=float, default=None,
                        help="per-pairing wall-clock budget in minutes "
                             "(an in-flight game finishes; truncation recorded)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master random seed (network init + game seeds)")
    parser.add_argument("--weights", type=str, default=None,
                        help="load a fixed network state_dict from this path")
    parser.add_argument("--smoke-train-steps", type=int, default=0,
                        help="plan fallback: smoke-train the network this "
                             "many steps on synthetic random data and use it "
                             "as the fixed weight (recorded as weight source)")
    parser.add_argument("--config", type=str, default=None,
                        help="config YAML path (default: config/default.yaml)")
    parser.add_argument("--sgf-dir", type=str, default="logs/matches/todo12",
                        help="write per-game SGFs here (default logs/matches/todo12; "
                             "pass 'none' to disable)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="write the result JSON to this path (utf-8)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    size = int(cfg["board_size"])
    komi = float(cfg.get("komi", 7.5))
    c_puct = float(cfg.get("c_puct", 2.5))
    virtual_loss = int(cfg.get("virtual_loss", 3))

    levels = DEFAULT_LEVELS
    valid_pairs = {
        (levels[0], levels[1]): "40v200",
        (levels[1], levels[2]): "200v800",
        (levels[0], levels[2]): "40v800",
    }
    pairings = tuple(p.strip() for p in args.pairings.split(",") if p.strip())
    for p in pairings:
        if p not in valid_pairs.values():
            parser.error(f"pairing {p!r} not among the plan's {list(valid_pairs.values())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    network = create_model(int(cfg["blocks"]), int(cfg["channels"]), size).to(device)

    weight_source = f"random init (torch seed {args.seed})"
    if args.weights:
        weight_source = _load_weights(network, args.weights, device)
    elif args.smoke_train_steps > 0:
        network = _smoke_train(network, cfg, args.smoke_train_steps, device,
                               args.seed, Path("models/todo12-fixed-weights.pt"))
        weight_source = (
            f"smoke-trained {args.smoke_train_steps} steps on synthetic random "
            f"data (models/todo12-fixed-weights.pt)"
        )
    network.eval()

    sgf_dir = None if args.sgf_dir.lower() == "none" else Path(args.sgf_dir)
    max_time_s = None if args.max_time is None else args.max_time * 60.0
    t_start = time.perf_counter()

    effective_games = {
        p: (min(args.games, QUICK_800_PAIRING_GAMES)
            if args.quick and "800" in p else args.games)
        for p in pairings
    }
    random_levels = levels
    if args.random_levels is not None and args.random_baseline > 0:
        parsed = [int(v) for v in args.random_levels.split(",") if v.strip()]
        random_levels = tuple(v for v in levels if v in parsed)

    result: dict = {
        "todo": 12,
        "device": str(device),
        "weight_source": weight_source,
        "protocol": {
            "levels": list(levels),
            "pairings": list(pairings),
            "games_per_pairing_full": args.games,
            "quick_mode": args.quick,
            "quick_reduction": ("800-sim pairings -> 30 games" if args.quick else None),
            "games_per_pairing_effective": effective_games,
            "random_baseline_games": args.random_baseline,
            "random_baseline_levels": list(random_levels),
            "max_time_min": args.max_time,
            "pairing_time_min": args.pairing_time,
            "tau": args.tau,
            "komi": komi,
            "max_moves": args.max_moves,
            "c_puct": c_puct,
            "virtual_loss": virtual_loss,
            "master_seed": args.seed,
        },
        "pairings": {},
        "random_baseline": {},
        "projected_times": {},
        "games": [],
        "monotonic": {},
        "time_box": {"triggered": False, "reason": None},
        "wall_time_s": 0.0,
    }

    pairing_wall: dict[str, list[float]] = {}
    for idx, p in enumerate(pairings):
        lo_sims, hi_sims = (int(v) for v in p.split("v"))
        base_seed = args.seed + idx * SEED_STRIDE
        games = effective_games[p]
        recs, summary = run_pairing(
            lo_sims, hi_sims, games, size, komi, network, args.tau,
            virtual_loss, base_seed, args.max_moves, sgf_dir,
            pairing_time_s=None if args.pairing_time is None
            else args.pairing_time * 60.0,
        )
        result["pairings"][p] = summary
        result["games"].extend(recs)
        pairing_wall[p] = [r["wall_time_s"] for r in recs]
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        if max_time_s is not None and elapsed_min >= args.max_time:
            result["time_box"]["triggered"] = True
            result["time_box"]["reason"] = (
                f"time budget {args.max_time} min exceeded after pairing {p} "
                f"({elapsed_min:.1f} min elapsed)"
            )
            break

    # -- random baseline (after pairings; cheap at low sims) --
    if args.random_baseline > 0:
        for level_idx, sims in enumerate(random_levels):
            if max_time_s is not None and (time.perf_counter() - t_start) / 60.0 >= args.max_time:
                result["time_box"]["triggered"] = True
                result["time_box"]["reason"] = (
                    f"time budget {args.max_time} min exceeded before random "
                    f"baseline at {sims} sims"
                )
                break
            base_seed = args.seed + 900000 + level_idx * SEED_STRIDE
            recs, summary = run_vs_random(
                sims, args.random_baseline, size, komi, network, args.tau,
                virtual_loss, base_seed, args.max_moves, sgf_dir,
            )
            result["random_baseline"][f"{sims}_vs_random"] = summary
            result["games"].extend(recs)
            pairing_wall[f"random{sims}"] = [r["wall_time_s"] for r in recs]

    result["wall_time_s"] = time.perf_counter() - t_start

    # -- projected full / quick protocol wall-times from measured games --
    proj: dict = {
        "assumption": "median per-game wall time x target games (measured on this device)"
    }
    full_total, quick_total = 0.0, 0.0
    for p in pairings:
        target_quick = (min(args.games, QUICK_800_PAIRING_GAMES)
                        if "800" in p else args.games)
        proj[p] = {
            "full_60_hours": _project_hours(pairing_wall.get(p, []), 60),
            "quick_hours": _project_hours(pairing_wall.get(p, []), target_quick),
        }
        full_total += (proj[p]["full_60_hours"] or 0.0)
        quick_total += (proj[p]["quick_hours"] or 0.0)
    proj["full_protocol_total_hours"] = full_total
    proj["quick_protocol_total_hours"] = quick_total
    result["projected_times"] = proj

    # -- monotonic acceptance: P(800>200)>0.5 and P(200>40)>0.5 --
    mono: dict = {}
    for p in ("40v200", "200v800"):
        s = result["pairings"].get(p)
        if s and s["games"]:
            lo, hi = p.split("v")
            mono[f"P({hi}>{lo})"] = s["hi_winrate"]
    ok_200_40 = mono.get("P(200>40)") is not None and mono["P(200>40)"] > 0.5
    ok_800_200 = mono.get("P(800>200)") is not None and mono["P(800>200)"] > 0.5
    mono["P(200>40)>0.5"] = ok_200_40
    mono["P(800>200)>0.5"] = ok_800_200
    mono["ordering_ok"] = bool(ok_200_40 and ok_800_200)
    result["monotonic"] = mono
    result["accepted"] = mono["ordering_ok"]

    _print_report(result)
    if args.evidence:
        path = Path(args.evidence)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evidence written: {path}", flush=True)

    return 0 if result["accepted"] else 1


def _print_report(result: dict) -> None:
    proto = result["protocol"]
    print("=== omigamax MCTS strength ladder (todo 12) ===", flush=True)
    print(f"device: {result['device']}", flush=True)
    print(f"weight source: {result['weight_source']}", flush=True)
    print(
        f"protocol: levels={proto['levels']} pairings={proto['pairings']} "
        f"tau={proto['tau']} komi={proto['komi']} max_moves={proto['max_moves']} "
        f"master_seed={proto['master_seed']}", flush=True
    )
    if proto["quick_mode"]:
        print(f"quick mode: {proto['quick_reduction']}", flush=True)
    print(f"games per pairing (effective): {proto['games_per_pairing_effective']}",
          flush=True)
    if result["time_box"]["triggered"]:
        print(f"TIME BOX: {result['time_box']['reason']}", flush=True)
    print("pairings (higher-sim side win rate):", flush=True)
    for p, s in result["pairings"].items():
        wr = f"{s['hi_winrate']:.3f}" if s["hi_winrate"] is not None else "n/a"
        trunc = " (time-truncated)" if s.get("truncated") else ""
        print(f"  {p}: games={s['games']}/{s.get('planned_games', s['games'])} "
              f"hi_wins={s['hi_wins']} hi_winrate={wr}{trunc}", flush=True)
    for k, s in result["random_baseline"].items():
        print(
            f"  {k}: games={s['games']} mcts_wins={s['mcts_wins']} "
            f"mcts_winrate={s['mcts_winrate']:.3f}", flush=True
        )
    mono = result["monotonic"]
    print("monotonic:", flush=True)
    for k, v in mono.items():
        print(f"  {k}: {v}", flush=True)
    pt = result["projected_times"]
    print(f"projected full protocol (60 games/pairing): "
          f"{pt['full_protocol_total_hours']:.1f} h", flush=True)
    print(f"projected quick protocol: "
          f"{pt['quick_protocol_total_hours']:.1f} h", flush=True)
    print(f"total wall time: {result['wall_time_s'] / 60:.1f} min "
          f"({len(result['games'])} games)", flush=True)
    print(
        f"RESULT: {'PASS' if result['accepted'] else 'FAIL'} "
        f"(monotonic P(200>40)>0.5={mono['P(200>40)>0.5']}, "
        f"P(800>200)>0.5={mono['P(800>200)>0.5']})", flush=True
    )


if __name__ == "__main__":
    raise SystemExit(main())
