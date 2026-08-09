"""P12 experiment driver: parallel self-play GPU utilization (workers=1/2/3).

P10 methodology: serial runs, one config at a time, real b20c256 net from
models/pretrain.pt, 150 sims/move, 2 games per run, nvidia-smi sampled every
~1s in a background thread for the whole batch wall. Reports per config:
sims/s (parent batch wall incl. spawn), avg GPU util% (all samples + busy
window where startup's zero-util spawn phase is excluded), %samples>=90%,
peak GPU memory, wall time per game, and the per-worker generation-only
timings.

Usage: uv run python .omo/evidence/omigamax-go/task-P12-workers.py
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def _sample_gpu(samples: list, stop: threading.Event) -> None:
    """Append (wall_ts, util%, mem MiB) every ~1s until ``stop``."""
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            util, mem = (out.split(",")[:2] + ["0", "0"])[:2]
            samples.append((time.perf_counter(), float(util), float(mem)))
        except Exception:  # noqa: BLE001 - a missed sample is not fatal
            pass
        time.sleep(1.0)


if __name__ == "__main__":
    import torch
    from omigamax.config import load_config
    from omigamax.network.model import create_model
    from omigamax.train.train import load_checkpoint
    import omigamax.train.selfplay as sp

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    ckpt = load_checkpoint("models/pretrain.pt")
    arch = ckpt["arch"]
    print(f"arch: {arch}", flush=True)
    net = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])

    configs = (1, 2, 3)
    rows = {}
    for workers in configs:
        samples: list = []
        stop = threading.Event()
        sampler = threading.Thread(target=_sample_gpu, args=(samples, stop),
                                   daemon=True)
        data_dir = Path("data/experiment_p12") / f"w{workers}"
        data_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        sampler.start()
        try:
            report, records = sp.generate_games(
                net, cfg, games=2, data_dir=data_dir, keep=1000, seed=12345,
                size=19, simulations=150, max_moves=150, leaf_batch=16,
                fp16=True, workers=workers)
        finally:
            stop.set()
            sampler.join(timeout=10)
        batch_wall = time.perf_counter() - t0

        utils = [s[1] for s in samples]
        mems = [s[2] for s in samples]
        busy = [u for u in utils if u > 0]
        per_game = report["wall_time_s"] / max(1, report["games"])
        worker_gen_s = (
            [w["wall_time_s"] for w in report["per_worker"]]
            if "per_worker" in report else [report["wall_time_s"]])
        rows[workers] = {
            "sims": report["sims"],
            "sims_per_sec": report["sims_per_sec"],
            "batch_wall_s": round(batch_wall, 2),
            "wall_time_s": round(report["wall_time_s"], 2),
            "wall_per_game_s": round(per_game, 2),
            "worker_generation_s": [round(s, 2) for s in worker_gen_s],
            "avg_util_all_pct": round(sum(utils) / len(utils), 1) if utils else 0.0,
            "avg_util_busy_pct": round(sum(busy) / len(busy), 1) if busy else 0.0,
            "pct_samples_ge90": round(100.0 * sum(u >= 90 for u in utils) / len(utils), 1) if utils else 0.0,
            "n_samples": len(utils),
            "peak_mem_mib": int(max(mems)) if mems else 0,
            "games": report["games"],
        }
        print(json.dumps({"workers": workers, **rows[workers]}), flush=True)
        # cooldown between configs so memory pressure is comparable
        time.sleep(3.0)

    print("\n=== SUMMARY (workers | sims/s | avg_util_busy% | ge90% | peakMiB | wall/game) ===")
    for w in configs:
        r = rows[w]
        print(f"{w} | {r['sims_per_sec']:.1f} | {r['avg_util_busy_pct']} | "
              f"{r['pct_samples_ge90']} | {r['peak_mem_mib']} | {r['wall_per_game_s']}s",
              flush=True)
    print("DONE", flush=True)
