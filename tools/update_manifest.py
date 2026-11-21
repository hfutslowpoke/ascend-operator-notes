#!/usr/bin/env python3
"""Regenerate MANIFEST-SHA256.json from git-tracked regular files.

The manifest is an integrity snapshot, not a source of truth.  Regenerate it
after intentional content changes so ``tools/verify_archive.py`` can detect
unexpected edits without reporting stale historical files.

Usage:
  python3 tools/update_manifest.py
  python3 tools/update_manifest.py --check
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST-SHA256.json"
SELF = "MANIFEST-SHA256.json"


def tracked_files() -> list[str]:
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        stderr=subprocess.STDOUT,
    )
    names = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    result = []
    for name in names:
        if name == SELF:
            continue
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            continue
        result.append(name)
    return sorted(result)


def build() -> dict:
    existing = {}
    if MANIFEST.exists():
        try:
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    files = {}
    for name in tracked_files():
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {
        "created_utc": existing.get("created_utc", now),
        "updated_utc": now,
        "algorithm": "sha256",
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the manifest is out of date",
    )
    args = parser.parse_args()
    document = build()
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not MANIFEST.exists():
            print("MANIFEST-SHA256.json is missing", file=sys.stderr)
            return 1
        current = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if current.get("algorithm") != document["algorithm"] or current.get("files") != document["files"]:
            print("MANIFEST-SHA256.json is out of date", file=sys.stderr)
            return 1
        print(f"manifest ok: {len(document['files'])} files")
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(document['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
