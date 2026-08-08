#!/usr/bin/env python3
"""P2 棋谱清洗：扫描 data/games 下的 *.sgf，过滤 + 内容去重，产出清洗后的语料清单。

流水线（CPU only，ProcessPoolExecutor 并行）：
  1. 递归扫描 ``--root``（默认 data/games）下的所有 ``*.sgf``。
  2. 轻量正则预筛（廉价，不触发真解析）：
       (a) SZ 缺省视为 19，显式 SZ != 19 剔除；
       (b) HA 缺省或无碍，HA 值非 "0"（让子棋）剔除；
       (c) RE 缺失 / 首字符非 B|W（无法判定胜者）剔除；
       (d) KM 缺失或非数值剔除；
       (e) 正则统计手数，非 [2, 500) 剔除。
  3. 通过预筛者走真解析器 ``omigamax.rules.sgf.parse_sgf``（缺 SZ 时注入
     ``SZ[19]`` 以适配解析器），解析失败计入 parse_error 并跳过。
  4. 按内容 sha256 去重（KGS 三份同内容归档塌缩为一份）。
  5. 产出清单 ``data/games/manifest_clean.jsonl``（每行一个游戏：
     source / path / sz / km / ha / re / winner / moves / sha256），
     并写统计数据 + 随机 20 条解析/回放健全性检查到
     ``.omo/evidence/omigamax-go/task-P2-clean.txt``。

用法:
    uv run python scripts/clean_corpus.py [--root data/games] [--out data/games/manifest_clean.jsonl] [--jobs N]
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from omigamax.rules.board import Board
from omigamax.rules.legality import IllegalMoveError
from omigamax.rules.sgf import parse_sgf

# -- 轻量正则（预筛用） -------------------------------------------------------
RE_SZ = re.compile(r"SZ\[(\d+)\]")
RE_KM = re.compile(r"KM\[([^\]]*)\]")
RE_HA = re.compile(r"HA\[([^\]]*)\]")
RE_RE = re.compile(r"RE\[([^\]]*)\]")
RE_MOVE = re.compile(r";(?:[BW])\[")
RE_OPEN = re.compile(r"\(\s*;")  # 首个节点起点，用于注入 SZ[19]

# KGS / CWI 归档的坐标方案与 FF[4] 不同：用 a..s 共 19 个字母（不跳过 'i'，
# 无 't'，'ss'=右下角）；而 omigamax 解析器按 FF[4]（a..t 跳过 'i'）。二者
# 语义索引一致但字母偏移不同，清洗时仅在落子/摆子节点内做归一化：
#   i->j, j->k, ..., s->t（a..h 不变），再交给 parse_sgf 保证几何正确 + 可解析。
_MOVE_NODE = re.compile(r";([BW])\[([^\]]*)\]")
_SETUP_NODE = re.compile(r";(AB|AW|AE)\[([^\]]*)\]")
_CORPUS_LETTERS = "abcdefghijklmnopqrs"  # 语料坐标字母表（a..s）
_TRANS = str.maketrans({chr(i): chr(i + 1) for i in range(ord("i"), ord("t"))})


def normalize_coords(text):
    """把语料坐标方案 (a..s) 归一化为 FF[4] (a..t 跳过 i)。

    仅改写落子/摆子节点（;B/W/AB/AW/AE）内的两字母坐标值，绝不动其它属性
    （PB/PW 等文本）——故不能全局 translate。已符合 FF[4] 的文件不受影响
    （含 't' 的坐标不在 a..s 内，原样保留）。
    """
    def _fix(inner):
        if len(inner) == 2 and all(ch in _CORPUS_LETTERS for ch in inner):
            return inner.translate(_TRANS)
        return inner

    text = _MOVE_NODE.sub(
        lambda m: ";%s[%s]" % (m.group(1), _fix(m.group(2))), text
    )
    text = _SETUP_NODE.sub(
        lambda m: ";%s[%s]" % (m.group(1), _fix(m.group(2))), text
    )
    return text


def prepare_text(raw_text):
    """缺失 SZ 时注入 SZ[19]（解析器要求显式 SZ），并归一化坐标。"""
    if RE_SZ.search(raw_text) is None:
        m = RE_OPEN.search(raw_text)
        if m is not None:
            raw_text = raw_text[: m.end()] + "SZ[19]" + raw_text[m.end():]
    return normalize_coords(raw_text)


MIN_MOVES, MAX_MOVES = 2, 500


def classify_source(path):
    """按目录树判断来源：cwi / kgs。"""
    norm = path.replace("\\", "/")
    if "/kgs/" in norm:
        return "kgs"
    if "/cwi/" in norm:
        return "cwi"
    return "other"


def process_one(path):
    """处理单个 SGF 文件。返回 ``(status, reason, payload)``。

    status: ``"ok"``（通过全部过滤，payload=清单记录）/ ``"reject"``
    （被某条规则剔除，reason 为规则名）/ ``"error"``（读盘失败）。
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return ("error", "read", str(exc)[:80])
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:  # 理论不发生（replace 兜底），防御性保留
        text = raw.decode("latin-1", "replace")

    sha = hashlib.sha256(raw).hexdigest()
    src = classify_source(path)

    # (a) 盘面尺寸：缺省 SZ = 19，显式非 19 剔除
    sz_m = RE_SZ.search(text)
    if sz_m is not None and int(sz_m.group(1)) != 19:
        return ("reject", "sz_not_19", None)
    sz = 19

    # (b) 让子棋：HA 存在且值非 "0" 剔除
    ha_m = RE_HA.search(text)
    ha = 0
    if ha_m is not None and ha_m.group(1) != "0":
        return ("reject", "handicap", None)

    # (d) 结果可解析 + 胜者可判定
    re_m = RE_RE.search(text)
    result = re_m.group(1).strip() if re_m else None
    if result is None:
        return ("reject", "result_missing", None)
    winner = result[0] if result and result[0] in "BW" else None
    if winner is None:
        return ("reject", "winner_undetermined", None)

    # (c) KM 缺失/非法 → 无法真解析
    km_m = RE_KM.search(text)
    if km_m is None:
        return ("reject", "missing_km", None)
    try:
        komi = float(km_m.group(1))
    except ValueError:
        return ("reject", "bad_km", None)

    # (c) 手数范围（正则粗统计，[2, 500)）
    n_moves = len(RE_MOVE.findall(text))
    if not (MIN_MOVES <= n_moves < MAX_MOVES):
        return ("reject", "moves_out_of_range", n_moves)

    # 真解析器：缺 SZ 注入 SZ[19]，坐标归一化到 FF[4]，再交给 parse_sgf
    try:
        parsed = parse_sgf(prepare_text(text))
    except ValueError as exc:
        return ("reject", "parse_error", str(exc)[:120])

    n_moves = len(parsed["moves"])
    record = {
        "source": src,
        "path": path,
        "sz": parsed["size"],
        "km": parsed["komi"],
        "ha": ha,
        "re": result,
        "winner": winner,
        "moves": n_moves,
        "sha256": sha,
    }
    return ("ok", None, record)


def replay(record):
    """按清单记录重放整局，返回 (ok?, 失败原因)。"""
    try:
        with open(record["path"], "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
    except OSError as exc:
        return False, f"read: {exc}"
    try:
        parsed = parse_sgf(prepare_text(text))
    except ValueError as exc:
        return False, f"parse: {exc}"
    board = Board(parsed["size"])
    for color, move in parsed["moves"]:
        try:
            board.play(move, color)
        except IllegalMoveError as exc:
            return False, f"illegal_move: {exc}"
    return True, None


def main():
    ap = argparse.ArgumentParser(description="P2 棋谱清洗流水线")
    ap.add_argument("--root", default="data/games")
    ap.add_argument("--out", default="data/games/manifest_clean.jsonl")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument("--evidence",
                    default=".omo/evidence/omigamax-go/task-P2-clean.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root
    files = sorted(glob.glob(os.path.join(root, "**", "*.sgf"), recursive=True))
    print(f"[scan] {len(files)} SGF files under {root}", flush=True)

    t0 = time.perf_counter()
    rejects = Counter()
    errors = Counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (status, reason, payload) in enumerate(
            pool.map(process_one, files, chunksize=64), 1
        ):
            if status == "ok":
                records.append(payload)
            elif status == "reject":
                rejects[reason] += 1
            else:
                errors[reason] += 1
    scan_s = time.perf_counter() - t0
    print(f"[scan] done in {scan_s:.1f}s; ok={len(records)} "
          f"reject={sum(rejects.values())} error={sum(errors.values())}",
          flush=True)

    # (e) 内容去重（KGS 3x 同内容归档 → 保留首个）
    t0 = time.perf_counter()
    seen = set()
    unique = []
    for rec in records:
        if rec["sha256"] not in seen:
            seen.add(rec["sha256"])
            unique.append(rec)
    dedup_s = time.perf_counter() - t0
    dup_count = len(records) - len(unique)

    # 排序输出（source / path 稳定顺序）
    unique.sort(key=lambda r: (r["source"], r["path"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in unique:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 统计 ---------------------------------------------------------------
    by_source = Counter(r["source"] for r in unique)
    wins = Counter(r["winner"] for r in unique)
    n = len(unique)
    avg_moves = (sum(r["moves"] for r in unique) / n) if n else 0.0
    km_dist = Counter(r["km"] for r in unique)

    # ---- 健全性：随机 20 条解析 + 回放 --------------------------------------
    rng = random.Random(args.seed)
    sample = rng.sample(unique, min(20, len(unique)))
    parsed_ok = 0
    replay_ok = 0
    replay_fail_reasons = []
    for rec in sample:
        try:
            with open(rec["path"], "rb") as fh:
                text = fh.read().decode("utf-8", "replace")
            # 与流水线同一预处理（坐标归一化 + 缺 SZ 注入）后再真解析，
            # 否则 KGS/CWI 的 a..s 坐标（含 'i'）会被 FF[4] 解析器误拒。
            parse_sgf(prepare_text(text))
            parsed_ok += 1
        except ValueError as exc:
            replay_fail_reasons.append(f"parse_fail({rec['source']}): {exc}")
            continue
        ok, why = replay(rec)
        if ok:
            replay_ok += 1
        else:
            replay_fail_reasons.append(f"replay_fail({rec['source']}): {why}")

    # ---- 统计字典（机器可读）-------------------------------------------------
    stats = {
        "scanned": len(files),
        "scan_seconds": scan_s,
        "dedup_seconds": dedup_s,
        "jobs": args.jobs,
        "reject_counts": dict(rejects),
        "read_errors": sum(errors.values()),
        "passed_prefilter": len(records),
        "duplicate_count": dup_count,
        "unique_count": n,
        "by_source": dict(by_source),
        "wins": dict(wins),
        "avg_moves": avg_moves,
        "km_top": dict(km_dist.most_common(8)),
        "sanity_parsed": parsed_ok,
        "sanity_replay": replay_ok,
        "sanity_sample": len(sample),
    }
    stats_path = os.path.splitext(args.out)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    # ---- 证据文件 -----------------------------------------------------------
    lines = []
    lines.append("P2 棋谱清洗（clean_corpus.py）运行报告")
    lines.append("=" * 72)
    lines.append(f"扫描目录     : {os.path.abspath(root)}")
    lines.append(f"扫描 SGF 数  : {len(files)}")
    lines.append(f"扫描耗时     : {scan_s:.1f}s（去重 {dedup_s:.1f}s，jobs={args.jobs}）")
    lines.append("")
    lines.append("逐规则过滤计数（每条棋谱按首个命中规则计一次）：")
    order = [
        ("sz_not_19", "SZ 显式且 != 19（小棋盘）"),
        ("handicap", "让子棋（HA 非 0）"),
        ("result_missing", "无 RE"),
        ("winner_undetermined", "RE 无法判定胜者（非 B/W 开头）"),
        ("missing_km", "无 KM"),
        ("bad_km", "KM 非数值"),
        ("moves_out_of_range", f"手数不在 [{MIN_MOVES}, {MAX_MOVES})"),
        ("parse_error", "真解析器 parse_sgf 失败"),
    ]
    for key, label in order:
        lines.append(f"  {label:<32}: {rejects.get(key, 0):,}")
    lines.append(f"  {'读盘失败':<32}: {sum(errors.values()):,}")
    lines.append("")
    lines.append(f"预筛通过进入真解析   : {len(records):,}")
    lines.append(f"内容重复（sha256 去重）: {dup_count:,}")
    lines.append(f"最终唯一棋谱数       : {n:,}")
    lines.append("")
    lines.append("按来源分布：")
    for src in ("kgs", "cwi", "other"):
        lines.append(f"  {src:<6}: {by_source.get(src, 0):,}")
    lines.append("")
    lines.append(f"胜负分布   : 黑 {wins.get('B', 0):,} / 白 {wins.get('W', 0):,}")
    lines.append(f"平均手数   : {avg_moves:.1f}")
    lines.append("KM 分布（Top 8）：")
    for km, cnt in km_dist.most_common(8):
        lines.append(f"  KM[{km}] : {cnt:,}")
    lines.append("")
    lines.append("健全性检查（随机 20 条）：")
    lines.append(f"  解析成功    : {parsed_ok}/20")
    lines.append(f"  完整回放成功（无 IllegalMoveError）: {replay_ok}/20")
    if replay_fail_reasons:
        lines.append("  失败明细：")
        for why in replay_fail_reasons[:6]:
            lines.append(f"    - {why}")
    lines.append("")
    lines.append(f"清单输出     : {os.path.abspath(args.out)}")
    report = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(args.evidence)), exist_ok=True)
    with open(args.evidence, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    sys.exit(main())
