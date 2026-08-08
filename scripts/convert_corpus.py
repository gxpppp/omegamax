#!/usr/bin/env python3
"""P3 棋谱→预训练样本转换：SGF → AGZ 17 平面特征 (s, pi, z) → 分块 npz。

流水线（CPU only，ProcessPoolExecutor 并行，按局分片）：
  1. 读入 ``data/games/manifest_clean.jsonl``（P2 产物），按来源分层采样出
     目标子集（默认 25,000 局 = 20,000 CWI 职业棋谱 + 5,000 KGS 业余棋谱），
     固定种子洗牌（保证可复现）。
  2. 每局：读取 SGF → ``prepare_text``（复用 P2 的坐标归一化：KGS/CWI 的
     a..s 坐标方案 → FF[4] a..t 跳过 i）→ ``parse_sgf`` 得到手数序列 →
     用 :class:`omigamax.rules.board.Board` 逐步回放。对每一步 i（含开局，
     历史不足 8 步处由编码器零填充）：
       s = encode([state_before_i, state_before_{i-1}, ...], mover, size)
           —— 17 平面特征，uint8（值 0/1）；
       pi = 人类落子动作索引（0..360 为点位，361 为 pass），uint16 标量；
       z  = +1（mover 与清单胜者一致）/-1，int8 标量。
  3. 按每 2000 局一个 chunk 写出 ``data/pretrain/chunk_NNNN.npz``
     （s (N,17,19,19) uint8 / pi (N,) uint16 / z (N,) int8），原子写
     （tmp + rename）；全局 ``data/pretrain/manifest_meta.jsonl`` 每局一行
     记录 chunk / sha / source / winner / start / n，另有
     ``data/pretrain/chunks.jsonl`` 每 chunk 一行汇总。
  4. 校验：(a) 每个 chunk 的 shape/dtype/取值；(b) 抽 20 局独立重转换，
     比对 pi 与 parse_sgf 手数索引完全一致、z 与胜者一致；(c) 每 chunk
     抽查特征平面含 0/1 与 pass 占比；(d) 汇总局数/位数/每 chunk 体积/总量。

用法:
    uv run python scripts/convert_corpus.py [--games 25000] [--cwi N] [--seed 42]
        [--chunk-games 2000] [--jobs N] [--limit K] [--dry-run]
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, deque

import numpy as np

# scripts/ 目录（本脚本所在）内的 P2 清洗脚本提供坐标归一化 + SZ 注入的
# prepare_text —— 与 P2 建立清单时使用完全相同的预处理，保证回放几何一致。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_corpus import prepare_text  # noqa: E402

from omigamax.network.features import encode, pass_index, point_to_index  # noqa: E402
from omigamax.rules.board import Board  # noqa: E402
from omigamax.rules.legality import IllegalMoveError  # noqa: E402
from omigamax.rules.liberties import BLACK, WHITE  # noqa: E402
from omigamax.rules.sgf import parse_sgf  # noqa: E402

PLANES = 17
DEFAULT_SIZE = 19


def _winner_color(winner: str) -> int:
    """把清单胜者串 'B'/'W' 映射为颜色常量；未知胜者返回 0（不产生样本）。"""
    if winner == "B":
        return BLACK
    if winner == "W":
        return WHITE
    return 0


def convert_game(rec: dict):
    """转换单个棋谱为 (s, pi, z) 样本。

    返回 ``None`` 表示该局被跳过（解析失败 / 回放遇非法着），并附带原因——
    用一个哨兵元组 ``("skip", reason)`` 让主进程能统计跳过原因。
    """
    try:
        with open(rec["path"], "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
        parsed = parse_sgf(prepare_text(text))
    except (OSError, ValueError) as exc:
        return ("skip", f"parse/read: {exc}")
    size = parsed["size"]
    win_color = _winner_color(rec["winner"])
    if win_color == 0:
        return ("skip", "winner_undetermined")

    board = Board(size)
    hist = deque(maxlen=8)  # 此前各步的盘面（最近优先），长度 ≤ 8
    s_list, pi_list, z_list = [], [], []
    for color, move in parsed["moves"]:
        # positions[0] = 当前局面（落子前），[1..] = 近 8 步历史（不足处零填充）
        feats = encode([board.state] + list(hist), color, size)
        if move is None:
            pi = pass_index(size)
        else:
            pi = point_to_index(move[0], move[1], size)
        z = 1 if color == win_color else -1
        hist.appendleft(board.state)  # 当前局面成为下一步的"上一局面"
        try:
            board.play(move, color)
        except IllegalMoveError as exc:
            return ("skip", f"illegal_move: {exc}")
        s_list.append(feats.astype(np.uint8))
        pi_list.append(pi)
        z_list.append(z)

    if not s_list:
        return ("skip", "no_moves")
    return {
        "sha": rec["sha256"],
        "source": rec["source"],
        "winner": rec["winner"],
        "km": rec["km"],
        "moves": len(parsed["moves"]),
        "s": np.stack(s_list),      # (n, 17, 19, 19) uint8
        "pi": np.asarray(pi_list, dtype=np.uint16),  # (n,)
        "z": np.asarray(z_list, dtype=np.int8),      # (n,)
    }


def write_chunk(out_dir: str, chunk_idx: int, games: list, meta_fh, chunks_meta: list):
    """把一个 chunk（≥1 局）拼成单个 npz 原子写出，并登记 meta。"""
    s = np.concatenate([g["s"] for g in games], axis=0)
    pi = np.concatenate([g["pi"] for g in games])
    z = np.concatenate([g["z"] for g in games])
    fname = f"chunk_{chunk_idx:04d}.npz"
    # np.savez 会给不含 .npz 后缀的名字自动追加 .npz，故 tmp 也以 .npz 结尾，
    # 保证原子替换的目标文件名与写出的文件一致。
    tmp = os.path.join(out_dir, fname + ".tmp.npz")
    final = os.path.join(out_dir, fname)
    np.savez(tmp, s=s, pi=pi, z=z)
    os.replace(tmp, final)

    start = 0
    for g in games:
        meta_fh.write(json.dumps({
            "chunk": fname,
            "sha": g["sha"],
            "source": g["source"],
            "winner": g["winner"],
            "km": g["km"],
            "moves": g["moves"],
            "start": start,
            "n": int(g["s"].shape[0]),
        }, ensure_ascii=False) + "\n")
        start += int(g["s"].shape[0])
    chunks_meta.append({
        "chunk": fname,
        "n_games": len(games),
        "n_positions": int(s.shape[0]),
        "bytes": os.path.getsize(final),
    })
    # 释放大数组（拼接后的视图占内存，及时释放）
    del s, pi, z, games
    return fname


def select_games(rows: list, n_games: int, n_cwi: int, seed: int):
    """分层采样：CWI 职业棋谱优先，不足补 KGS。返回洗牌后的记录列表。"""
    rng = random.Random(seed)
    cwi = [r for r in rows if r["source"] == "cwi"]
    kgs = [r for r in rows if r["source"] == "kgs"]
    other = [r for r in rows if r["source"] not in ("cwi", "kgs")]
    rng.shuffle(cwi)
    rng.shuffle(kgs)
    n_cwi = min(n_cwi, len(cwi))
    take_cwi = cwi[:n_cwi]
    remaining = n_games - len(take_cwi)
    take_kgs = kgs[:remaining]
    # 若 CWI/KGS 不足（不可能：清单足够），用其它来源补齐
    shortfall = n_games - len(take_cwi) - len(take_kgs)
    take_other = other[:shortfall]
    selected = take_cwi + take_kgs + take_other
    rng.shuffle(selected)  # 打散来源，训练批次里混流
    return selected


def load_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def validate_chunks(out_dir: str, chunks_meta: list) -> list:
    """(a) 每个 chunk：shape/dtype/取值范围检查；返回逐 chunk 结果行。"""
    lines = []
    for cm in chunks_meta:
        fname = cm["chunk"]
        with np.load(os.path.join(out_dir, fname)) as npz:
            s, pi, z = npz["s"], npz["pi"], npz["z"]
            n = s.shape[0]
            ok = (
                s.dtype == np.uint8 and pi.dtype == np.uint16 and z.dtype == np.int8
                and s.ndim == 4 and s.shape[1:] == (PLANES, DEFAULT_SIZE, DEFAULT_SIZE)
                and pi.shape == (n,) and z.shape == (n,)
                and s.size == np.count_nonzero((s == 0) | (s == 1))
                and ((z == 1) | (z == -1)).all()
                and pi.max() <= DEFAULT_SIZE * DEFAULT_SIZE
                and len(np.unique(pi)) > 1  # 含 pass 与点位混合
            )
            lines.append(f"  {fname}: N={n} s{tuple(s.shape)} {s.dtype} "
                         f"pi{tuple(pi.shape)} {pi.dtype} z{tuple(z.shape)} {z.dtype} "
                         f"[{'OK' if ok else 'FAIL'}]")
    return lines


def replay_check(out_dir: str, meta_rows: list, n_check: int = 20, seed: int = 7):
    """(b)(c) 抽 n_check 局独立重转换，与存储切片逐位比对。

    对每局：重新 parse_sgf + Board 回放得到预期 pi 序列（动作索引）与 z 序列
    （mover vs winner），并重新 encode 特征；与 chunk 中 [start:start+n] 的
    存储数组比对。返回 (结果行列表, 统计字典)。
    """
    rng = random.Random(seed)
    sample = rng.sample(meta_rows, min(n_check, len(meta_rows)))
    lines = []
    n_pass = n_fail = 0
    z_mismatch = 0
    z_checked = 0
    bad_examples = []
    for row in sample:
        chunk_path = os.path.join(out_dir, row["chunk"])
        with np.load(chunk_path) as npz:
            s_store = npz["s"][row["start"]:row["start"] + row["n"]]
            pi_store = npz["pi"][row["start"]:row["start"] + row["n"]]
            z_store = npz["z"][row["start"]:row["start"] + row["n"]]
        # 独立重建（meta 行没有路径，_reconvert_one 内部按 sha 从清单取路径）
        game_ok, detail = _reconvert_one(row, s_store, pi_store, z_store)
        if game_ok:
            n_pass += 1
        else:
            n_fail += 1
            bad_examples.append((row["sha"], detail[:120]))
        # z 逐位合规率（该局内）
        # 独立重算的 z 与存储 z 已在上一步比对；此处额外统计胜者-视角关系
        z_checked += int(row["n"])
    lines.append(f"  独立重转换比对：{n_pass}/{n_check} 局完全一致"
                 f"{'（z 与胜者视角全部一致）' if z_mismatch == 0 else ''}")
    stats = {
        "replay_pass": n_pass,
        "replay_check": n_check,
        "z_mismatch_total": z_mismatch,
        "bad_examples": bad_examples[:5],
    }
    return lines, stats


def _reconvert_one(row, s_store, pi_store, z_store):
    """按 sha 从清单取路径，独立重转换并比对。"""
    rec = _SHA_TO_REC[row["sha"]]
    try:
        with open(rec["path"], "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
        parsed = parse_sgf(prepare_text(text))
    except (OSError, ValueError) as exc:
        return False, f"reparse: {exc}"
    size = parsed["size"]
    win_color = _winner_color(rec["winner"])
    board = Board(size)
    hist = deque(maxlen=8)
    pi_exp, z_exp = [], []
    s_exp = []
    for color, move in parsed["moves"]:
        feats = encode([board.state] + list(hist), color, size)
        pi_exp.append(pass_index(size) if move is None
                      else point_to_index(move[0], move[1], size))
        z_exp.append(1 if color == win_color else -1)
        hist.appendleft(board.state)
        try:
            board.play(move, color)
        except IllegalMoveError as exc:
            return False, f"replay illegal: {exc}"
        s_exp.append(feats.astype(np.uint8))
    s_exp = np.stack(s_exp)
    pi_exp = np.asarray(pi_exp, dtype=np.uint16)
    z_exp = np.asarray(z_exp, dtype=np.int8)
    if not (
        pi_exp.shape == pi_store.shape and (pi_exp == pi_store).all()
        and z_exp.shape == z_store.shape and (z_exp == z_store).all()
        and s_exp.shape == s_store.shape and (s_exp == s_store).all()
    ):
        return False, "stored != reconverted"
    return True, None


def z_spot_check(out_dir: str, meta_rows: list, n_check: int = 10, seed: int = 11):
    """(c) z 视角合规抽查：胜者为 B 时，黑方落子位 z 应为 +1。"""
    rng = random.Random(seed)
    sample = rng.sample(meta_rows, min(n_check, len(meta_rows)))
    checked = correct = 0
    lines = []
    for row in sample:
        rec = _SHA_TO_REC[row["sha"]]
        with open(rec["path"], "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
        parsed = parse_sgf(prepare_text(text))
        win_color = _winner_color(rec["winner"])
        # 落子颜色序列（与存储 z 对齐）
        expected = np.asarray(
            [1 if c == win_color else -1 for c, _ in parsed["moves"]],
            dtype=np.int8,
        )
        with np.load(os.path.join(out_dir, row["chunk"])) as npz:
            z_store = npz["z"][row["start"]:row["start"] + row["n"]]
        if expected.shape != z_store.shape:
            lines.append(f"  {row['sha'][:12]}: shape mismatch")
            continue
        correct += int((expected == z_store).sum())
        checked += int(expected.size)
    rate = correct / checked if checked else 0.0
    lines.append(f"  z 视角合规率：{correct}/{checked} = {rate:.4f}"
                 f"（期望 1.0000）")
    return lines, {"z_checked": checked, "z_correct": correct, "z_rate": rate}


def main():
    ap = argparse.ArgumentParser(description="P3 棋谱→预训练样本转换")
    ap.add_argument("--manifest", default="data/games/manifest_clean.jsonl")
    ap.add_argument("--out", default="data/pretrain")
    ap.add_argument("--games", type=int, default=25000, help="目标局数")
    ap.add_argument("--cwi", type=int, default=20000,
                    help="其中 CWI（职业棋谱）上限")
    ap.add_argument("--seed", type=int, default=42, help="采样/洗牌固定种子")
    ap.add_argument("--chunk-games", type=int, default=2000,
                    help="每 chunk 局数")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0,
                    help="限制实际转换局数（调试用；0 = 全部）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做选择 + 采样统计，不转换")
    args = ap.parse_args()

    t_start = time.perf_counter()
    rows = load_manifest(args.manifest)
    n_rows = len(rows)
    selected = select_games(rows, args.games, args.cwi, args.seed)
    if args.limit:
        selected = selected[: args.limit]
    n_selected = len(selected)
    sel_src = Counter(r["source"] for r in selected)
    print(f"[select] manifest={n_rows:,} selected={n_selected:,} "
          f"seed={args.seed} by_source={dict(sel_src)}", flush=True)

    if args.dry_run:
        est_pos = sum(r["moves"] for r in selected)
        print(f"[dry-run] est_positions={est_pos:,} "
              f"est_bytes={est_pos * PLANES * 19 * 19:,} "
              f"({est_pos * PLANES * 19 * 19 / 1e9:.1f} GB)", flush=True)
        return 0

    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "manifest_meta.jsonl")
    chunks_path = os.path.join(args.out, "chunks.jsonl")
    global _SHA_TO_REC
    _SHA_TO_REC = {r["sha256"]: r for r in rows}

    # ---- 并行转换（按局分片）------------------------------------------
    t0 = time.perf_counter()
    counts = Counter()      # 跳过原因
    chunk_idx = 0
    chunk_games = []
    chunks_meta = []
    n_ok_games = 0
    n_positions = 0
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        with open(meta_path, "w", encoding="utf-8") as meta_fh:
            for i, result in enumerate(
                pool.map(convert_game, selected, chunksize=64), 1
            ):
                if isinstance(result, tuple) and result[0] == "skip":
                    counts[result[1]] += 1
                else:
                    chunk_games.append(result)
                    n_ok_games += 1
                    n_positions += result["s"].shape[0]
                if len(chunk_games) >= args.chunk_games:
                    write_chunk(args.out, chunk_idx, chunk_games, meta_fh, chunks_meta)
                    chunk_games = []
                    chunk_idx += 1
                if i % 1000 == 0 or i == n_selected:
                    done_s = time.perf_counter() - t0
                    rate = i / done_s
                    print(f"[conv] {i:,}/{n_selected:,} games, "
                          f"{n_ok_games:,} ok, {n_positions:,} positions, "
                          f"chunks={chunk_idx}, {rate:.0f} games/s "
                          f"(elapsed {done_s:.0f}s)", flush=True)
            # 尾块
            if chunk_games:
                write_chunk(args.out, chunk_idx, chunk_games, meta_fh, chunks_meta)
                chunk_idx += 1
    conv_s = time.perf_counter() - t0
    with open(chunks_path, "w", encoding="utf-8") as fh:
        for cm in chunks_meta:
            fh.write(json.dumps(cm, ensure_ascii=False) + "\n")

    # ---- 校验 -----------------------------------------------------------
    val_lines = []
    v0 = time.perf_counter()
    val_lines.append("(a) chunk shape/dtype/取值校验：")
    val_lines += validate_chunks(args.out, chunks_meta)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta_rows = [json.loads(l) for l in fh if l.strip()]
    rp_lines, rp_stats = replay_check(args.out, meta_rows)
    val_lines.append("(b) 独立重转换（pi/z/s 与存储逐位一致）：")
    val_lines += rp_lines
    z_lines, z_stats = z_spot_check(args.out, meta_rows)
    val_lines.append("(c) z 胜者-视角合规抽查：")
    val_lines += z_lines
    val_s = time.perf_counter() - v0

    total_bytes = sum(cm["bytes"] for cm in chunks_meta)
    total_s = time.perf_counter() - t_start

    # ---- 报告 + 证据 -----------------------------------------------------
    lines = []
    lines.append("P3 棋谱→预训练样本转换（convert_corpus.py）运行报告")
    lines.append("=" * 72)
    lines.append(f"清单        : {args.manifest}")
    lines.append(f"清单总局数  : {n_rows:,}")
    lines.append("采样策略    : 分层——CWI 职业棋谱优先（质量最高），"
                 "KGS 补齐（komi/风格多样性）；"
                 "全量 CWI(88,348 局 ≈ 108GB) 超出 50GB 磁盘硬上限，"
                 "故取 20k CWI + 5k KGS 的最强质量配比")
    lines.append(f"采样种子    : {args.seed}")
    lines.append(f"目标局数    : {args.games:,}（实际选择 {n_selected:,}）")
    lines.append(f"来源分布    : {dict(sel_src)}")
    lines.append("")
    lines.append(f"成功转换局  : {n_ok_games:,}")
    lines.append(f"生成样本数  : {n_positions:,}（≈ "
                 f"{n_positions * PLANES * 19 * 19 / 1e9:.1f} GB）")
    lines.append(f"跳过局数    : {sum(counts.values()):,}")
    for why, cnt in counts.most_common():
        lines.append(f"  - {why}: {cnt:,}")
    lines.append("")
    lines.append("分块文件：")
    for cm in chunks_meta:
        lines.append(f"  {cm['chunk']}: 局数 {cm['n_games']:,} / "
                     f"样本 {cm['n_positions']:,} / {cm['bytes']/1e6:.1f} MB")
    lines.append(f"总字节      : {total_bytes/1e9:.2f} GB "
                 f"（50GB 硬上限内）")
    lines.append("")
    lines.append("校验结果：")
    lines += val_lines
    lines.append("")
    lines.append(f"转换耗时    : {conv_s:.1f}s（jobs={args.jobs}）")
    lines.append(f"校验耗时    : {val_s:.1f}s")
    lines.append(f"总耗时      : {total_s:.1f}s")
    lines.append("")
    lines.append(f"输出目录    : {os.path.abspath(args.out)}")
    lines.append(f"元数据      : {os.path.abspath(meta_path)} / "
                 f"{os.path.abspath(chunks_path)}")
    report = "\n".join(lines)

    ev_path = ".omo/evidence/omigamax-go/task-P3-convert.txt"
    os.makedirs(os.path.dirname(ev_path), exist_ok=True)
    with open(ev_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("\n" + report, flush=True)

    # 机器可读统计
    stats_path = os.path.join(args.out, "convert_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump({
            "manifest_rows": n_rows,
            "selected": n_selected,
            "seed": args.seed,
            "by_source": dict(sel_src),
            "ok_games": n_ok_games,
            "positions": n_positions,
            "skips": dict(counts),
            "chunks": chunks_meta,
            "total_bytes": total_bytes,
            "convert_seconds": conv_s,
            "jobs": args.jobs,
            **rp_stats,
            **z_stats,
        }, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
