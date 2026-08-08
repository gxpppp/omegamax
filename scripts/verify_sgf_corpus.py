#!/usr/bin/env python3
"""Verify downloaded Go SGF corpora: counts + random parse samples."""
import glob
import random
import re
import sys

COLS = "abcdefghijklmnopqrs"  # 19 cols => max index 's'

def count_sgf(root):
    return glob.glob(root + "/**/*.sgf", recursive=True)

def board_size(path):
    with open(path, "rb") as fh:
        raw = fh.read(20000)
    txt = raw.decode("utf-8", "replace")
    m = re.search(r"SZ\[(\d+)\]", txt)
    if m:
        return int(m.group(1))
    # No SZ property -> SGF default is 19.
    # Cross-check with move coordinates if any are present.
    coords = re.findall(r";(?:[BW])\[([a-s])([a-s])\]", txt)
    if coords:
        maxc = max(max(ord(a), ord(b)) for a, b in coords)
        return maxc - ord("a") + 1
    return 19  # no moves / no SZ: default


def sample_check(root, label, k=5, seed=42):
    files = count_sgf(root)
    if not files:
        print(f"{label}: NO SGF FILES")
        return
    print(f"--- {label}: total={len(files)}")
    rng = random.Random(seed)
    for f in rng.sample(files, min(k, len(files))):
        with open(f, "rb") as fh:
            head = fh.read(64)
        starts = head.startswith(b"(;")
        bs = board_size(f)
        name = f.replace("\\", "/").split("/")[-1]
        print(f"  {name}: starts_with_semicolon={starts} inferred_board={bs}")


if __name__ == "__main__":
    roots = {
        "CWI": "data/games/cwi",
        "KGS-zip": "data/games/kgs/zip",
        "KGS-targz": "data/games/kgs/targz",
        "KGS-tarbz2": "data/games/kgs/tarbz2",
    }
    for label, root in roots.items():
        sample_check(root, label)
