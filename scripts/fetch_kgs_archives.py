#!/usr/bin/env python3
"""Fetch all KGS 7d+ game record archives from u-go.net.

Parses every archive href (KGS-*.zip / *.tar.gz / *.tar.bz2) on the
index page https://www.u-go.net/gamerecords/, downloads each archive
into the given destination dir, skipping files that already exist with
size > 0 (resume-friendly). Downloads are sequential; each archive gets
up to RETRIES attempts with RETRY_BACKOFF seconds between failures.
Archives that still fail after all retries are logged and skipped.

Usage:
    python fetch_kgs_archives.py [--out DIR] [--index-url URL]
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error

INDEX_URL = "https://www.u-go.net/gamerecords/"
ARCHIVE_PATTERN = r'href="(https://dl\.u-go\.net/gamerecords/KGS-.*?\.(?:zip|tar\.gz|tar\.bz2))"'
RETRIES = 3
RETRY_BACKOFF = 5  # seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) omigamax-go-downloader/1.0"
ARCHIVE_DIR = "data/games/kgs/archives"


def get_archive_urls(index_url):
    req = urllib.request.Request(index_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    import re

    urls = sorted(set(re.findall(ARCHIVE_PATTERN, html)))
    if not urls:
        sys.stderr.write(f"ERROR: no KGS archive hrefs found at {index_url}\n")
        sys.exit(2)
    return urls


def download_one(url, out_path):
    """Download url to out_path. Returns True on success."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(out_path, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
            if os.path.getsize(out_path) > 0:
                return True
        except (urllib.error.URLError, OSError) as exc:
            sys.stderr.write(f"  attempt {attempt}/{RETRIES} failed for {url}: {exc}\n")
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF)
    return False


def main():
    ap = argparse.ArgumentParser(description="Fetch all KGS 7d+ archives from u-go.net")
    ap.add_argument("--out", default=ARCHIVE_DIR, help="destination dir for archives")
    ap.add_argument("--index-url", default=INDEX_URL, help="index page URL")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    urls = get_archive_urls(args.index_url)
    print(f"Found {len(urls)} KGS archive links on {args.index_url}")

    failed = []
    downloaded = 0
    skipped = 0
    for url in urls:
        fname = url.rsplit("/", 1)[-1]
        out_path = os.path.join(args.out, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            print(f"[skip ] {fname}")
            continue
        print(f"[dl   ] {fname}")
        if download_one(url, out_path):
            downloaded += 1
        else:
            failed.append(url)
            print(f"[FAIL ] {fname}")

    print(f"\nSummary: {len(urls)} archives, {downloaded} downloaded, {skipped} skipped (already present), {len(failed)} failed")
    if failed:
        print("Failed archives:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
