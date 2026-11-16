#!/usr/bin/env python3
"""Offline integrity checks. Does not execute archived source or contact a server."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent.parent
errors = []
counts = {"public_files": 0, "own_copied_files": 0, "python_syntax": 0, "json": 0}


def check_file(rel, sha256, blob=None):
    path = ROOT / rel
    if not path.is_file():
        errors.append(f"missing: {rel}")
        return
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        errors.append(f"SHA256 mismatch: {rel}")
    if blob and hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != blob:
        errors.append(f"Git blob mismatch: {rel}")


catalog = json.loads((ROOT / "catalog/public-final-sources.json").read_text())
for record in catalog["records"]:
    for item in record["files"]:
        check_file(item["local_path"], item["sha256"], item["git_blob_sha1"])
        counts["public_files"] += 1
if catalog.get("errors") or counts["public_files"] != catalog["file_count"]:
    errors.append("public source acquisition is incomplete")
own = json.loads((ROOT / "catalog/our-work.json").read_text())
for item in own["copied_files"]:
    check_file(item["path"], item["sha256"])
    counts["own_copied_files"] += 1

patterns = {
    "GitHub token": rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{30,})",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "literal bearer credential": rb"(?i)Bearer\s+[A-Za-z0-9_.-]{30,}",
}
for path in ROOT.rglob("*"):
    rel = path.relative_to(ROOT)
    if ".git" in rel.parts or not path.is_file():
        continue
    if path.is_symlink():
        errors.append(f"unexpected symlink: {rel}")
    if path.name in {"accounts.json", "hosts.yml", ".env"} or path.suffix in {".pem", ".key"}:
        errors.append(f"credential filename: {rel}")
    data = path.read_bytes()
    for name, pattern in patterns.items():
        if re.search(pattern, data):
            errors.append(f"{name} pattern in {rel}")  # Never print matched secret.
    if path.suffix == ".json":
        try:
            json.loads(data)
            counts["json"] += 1
        except (ValueError, UnicodeDecodeError) as exc:
            errors.append(f"invalid JSON {rel}: {type(exc).__name__}")
    if path.suffix == ".py" and rel.parts[0] == "tools":
        try:
            compile(data, str(rel), "exec")
            counts["python_syntax"] += 1
        except SyntaxError as exc:
            errors.append(f"Python syntax {rel}:{exc.lineno}")

# Newly written navigation/docs only. Historical reports may reference old workspaces.
docs = [ROOT / "README.md", ROOT / "THIRD_PARTY.md", ROOT / "catalog/INDEX.md"]
docs += list((ROOT / "docs").glob("*.md"))
for path in docs:
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            continue
        target = unquote(target.split("#", 1)[0])
        if not (path.parent / target).is_file() and not (path.parent / target).is_dir():
            errors.append(f"broken link in {path.relative_to(ROOT)}: {target}")

manifest = ROOT / "MANIFEST-SHA256.json"
if manifest.exists():
    for rel, digest in json.loads(manifest.read_text())["files"].items():
        check_file(rel, digest)

print(json.dumps({"ok": not errors, "checks": counts, "errors": errors}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))

