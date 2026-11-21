#!/usr/bin/env python3
"""Build a reproducible submission archive from a CannJudge template.

This tool never POSTs.  It fetches the template, copies the editable files
into a ZIP using the template paths, and writes a manifest with hashes.
Optionally it also writes the exact JSON payload used by cann_submit.py.

Example:
  python3 cann_package.py --problem-id <PID> --account A \
      --source-dir ./src --out submission.zip --payload-json payload.json
"""
import argparse
import datetime
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cann_client  # noqa: E402
from cann_submit import read_sources, source_digest  # noqa: E402


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out", required=True, help="output zip")
    ap.add_argument("--payload-json", help="also write the POST payload JSON")
    args = ap.parse_args()

    row, template = cann_client.template_files(args.problem_id, args.account)
    template = cann_client.unwrap(template)
    files = template.get("files", [])
    if not files:
        raise SystemExit("template returned no files")
    selected = read_sources(files, args.source_dir, {})
    digest = source_digest(selected)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": now_utc(),
        "problem_id": args.problem_id,
        "account": args.account,
        "uid": row["uid"],
        "source_digest": digest,
        "files": [
            {"path": x["path"], "bytes": x["bytes"], "sha256": x["sha256"]}
            for x in selected
        ],
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in selected:
            zf.writestr(item["path"], item["content"])
        zf.writestr("SUBMISSION-MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    if args.payload_json:
        payload = {
            "problemId": args.problem_id,
            "userId": row["uid"],
            "files": [{"path": x["path"], "content": x["content"]} for x in selected],
        }
        Path(args.payload_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({**manifest, "zip": str(out), "zip_bytes": out.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
