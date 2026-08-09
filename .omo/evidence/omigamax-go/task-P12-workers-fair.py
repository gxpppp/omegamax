"""P12 supplementary: fair 3-active-worker measurement (games=3, workers=3).

The plan's 2-game protocol leaves one worker idle at workers=3 (2 games /
3 workers). This run gives every worker exactly one game to answer whether a
3rd ACTIVE worker adds throughput beyond workers=2.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def _sample_gpu(samples: list, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            util, mem = (out.split(",")[:2] + ["0", "0"])[:2]
            samples.append((time.perf_counter(), float(util), float(mem)))
        except Exception:  # noqa: BLE001
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
    ckpt = load_checkpoint("models/pretrain.pt")
    arch = ckpt["arch"]
    net = create_model(
        int(arch["blocks"]), int(arch["channels"]), int(arch["board_size"])
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])

    rows = {}
    for workers, games in ((2, 4), (3, 3)):
        samples: list = []
        stop = threading.Event()
        sampler = threading.Thread(target=_sample_gpu, args=(samples, stop),
                                   daemon=True)
        data_dir = Path("data/experiment_p12") / f"fair_w{workers}"
        data_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        sampler.start()
        try:
            report, records = sp.generate_games(
                net, cfg, games=games, data_dir=data_dir, keep=1000, seed=12345,
                size=19, simulations=150, max_moves=150, leaf_batch=16,
                fp16=True, workers=workers)
        finally:
            stop.set()
            sampler.join(timeout=10)
        batch_wall = time.perf_counter() - t0
        utils = [s[1] for s in samples]
        mems = [s[2] for s in samples]
        busy = [u for u in utils if u > 0]
        rows[f"w{workers}g{games}"] = {
            "sims": report["sims"],
            "sims_per_sec": round(report["sims_per_sec"], 1),
            "batch_wall_s": round(batch_wall, 1),
            "wall_per_game_s": round(report["wall_time_s"] / report["games"], 1),
            "worker_generation_s": [round(w["wall_time_s"], 1)
                                    for w in report["per_worker"]],
            "avg_util_busy_pct": round(sum(busy) / len(busy), 1) if busy else 0.0,
            "peak_mem_mib": int(max(mems)) if mems else 0,
            "games": report["games"],
        }
        print(json.dumps({k: rows[f"w{workers}g{games}"][k]
                          for k in rows[f"w{workers}g{games}"]}), flush=True)
        time.sleep(3.0)

    print("\n=== FAIR COMPARISON ===")
    for k, r in rows.items():
        print(f"{k}: {r['sims_per_sec']} sims/s | busy_util {r['avg_util_busy_pct']}% | "
              f"peak {r['peak_mem_mib']}MiB | wall/game {r['wall_per_game_s']}s", flush=True)
    print("DONE", flush=True)
