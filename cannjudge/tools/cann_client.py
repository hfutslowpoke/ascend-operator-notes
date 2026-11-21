#!/usr/bin/env python3
"""CannJudge public discovery + authenticated read-only queries.

Credentials live in ~/.config/cannjudge/accounts.json (mode 0600) or the
file named by CANNJUDGE_ACCOUNT_FILE.  This tool never prints cookies.

Examples:
  python3 cann_client.py accounts
  python3 cann_client.py contests
  python3 cann_client.py resolve <contest-slug> [problem-slug]
  python3 cann_client.py problems <contest-slug-or-id>
  python3 cann_client.py template <problem-id> --account A
  python3 cann_client.py download-template <problem-id> template.zip --account A
  python3 cann_client.py history <problem-id> --account A
  python3 cann_client.py detail <submission-id> --account A
  python3 cann_client.py set-cookie A
"""
import argparse
import base64
import getpass
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = os.environ.get("CANNJUDGE_API", "https://cannjudge.cn/api").rstrip("/")
ACCOUNT_FILE = Path(
    os.environ.get(
        "CANNJUDGE_ACCOUNT_FILE",
        str(Path.home() / ".config/cannjudge/accounts.json"),
    )
)
if os.environ.get("CANNJUDGE_USE_PROXY") == "1":
    OPENER = urllib.request.build_opener()
else:
    OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class CannHttpError(RuntimeError):
    def __init__(self, method, path, status, metadata):
        self.status = status
        self.metadata = metadata
        suffix = ""
        if "retry_after_seconds" in metadata:
            suffix = f"; retry_after_seconds={metadata['retry_after_seconds']}"
        super().__init__(f"{method} {path}: HTTP {status}{suffix}")


def token_metadata(token):
    """Read uid/exp from the signed cookie payload without validating it."""
    segment = token.split(".")[0]
    payload = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    expiry = payload.get("exp")
    if expiry is not None:
        expiry = float(expiry)
        if expiry > 100_000_000_000:
            expiry /= 1000
    return {"uid": str(payload["uid"]), "expires_epoch": expiry}


def _load_rows():
    if not ACCOUNT_FILE.exists():
        return []
    return json.loads(ACCOUNT_FILE.read_text())


def accounts():
    saved = _load_rows()
    by_label = {row["label"]: row for row in saved if row.get("label")}
    order = []
    for row in saved:
        label = row.get("label")
        if label and label not in order:
            order.append(label)
    for label in "ABCDE":
        if label not in order:
            order.append(label)
    result = []
    for label in order:
        row = dict(by_label.get(label, {"label": label}))
        token = os.environ.get("CANNJUDGE_COOKIE_" + label) or row.get("cookie")
        if token:
            try:
                row.update(token_metadata(token))
            except Exception:
                pass
            row["cookie"] = token
            result.append(row)
    return result


def account(label):
    for row in accounts():
        if row.get("label") == label:
            if not row.get("cookie"):
                raise RuntimeError(f"account {label} has no cookie; run set-cookie")
            return row
    raise RuntimeError(f"missing credential for account {label}")


def request(method, path, cookie=None, body=None, binary=False):
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("API path must begin with one /, without hostname")
    headers = {"Accept": "application/json"}
    if cookie:
        headers["Cookie"] = "cannjudge_auth=" + cookie
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with OPENER.open(req, timeout=30) as response:
            raw = response.read()
            return raw if binary else json.loads(raw)
    except urllib.error.HTTPError as error:
        metadata = {}
        raw = error.read()
        try:
            payload = json.loads(raw)
            for obj in [payload, payload.get("data", {})]:
                if isinstance(obj, dict):
                    for key in (
                        "retry_after_seconds",
                        "retry_after",
                        "limit_type",
                        "quota_reset_at",
                        "quota_remaining",
                        "msg",
                        "message",
                    ):
                        value = obj.get(key)
                        if isinstance(value, (str, int, float, bool)):
                            if isinstance(value, str):
                                value = value[:500]
                                if cookie:
                                    value = value.replace(cookie, "[REDACTED]")
                            metadata[key] = value
        except (ValueError, AttributeError):
            payload = None
        if raw:
            body = raw.decode("utf-8", errors="replace")[:2000]
            if cookie:
                body = body.replace(cookie, "[REDACTED]")
            metadata.setdefault("body", body)
        headers = {}
        for key, value in (error.headers or {}).items():
            lower = key.lower()
            if lower == "retry-after" or lower.startswith("x-ratelimit-"):
                headers[key] = value
        if headers:
            metadata["headers"] = headers
        raise CannHttpError(method, path, error.code, metadata) from None


def unwrap(value):
    if isinstance(value, dict) and "data" in value and (
        "code" in value or "msg" in value or "success" in value
    ):
        return value["data"]
    return value


def save_cookie(label):
    token = getpass.getpass(f"Paste account {label} cannjudge_auth value (hidden): ").strip()
    if token.startswith("cannjudge_auth="):
        token = token.split("=", 1)[1].split(";", 1)[0]
    meta = token_metadata(token)
    rows = _load_rows()
    previous = next((r for r in rows if r.get("label") == label), None)
    if previous and previous.get("uid") and previous["uid"] != meta["uid"]:
        raise RuntimeError("UID differs from saved account; check the browser account")
    row = dict(previous or {"label": label})
    row.update(meta, cookie=token, usable_for_submit=True)
    rows = [r for r in rows if r.get("label") != label] + [row]
    rows.sort(key=lambda r: r.get("label", ""))
    ACCOUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="accounts-", dir=ACCOUNT_FILE.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, ACCOUNT_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"label": label, "uid": meta["uid"], "saved_to": str(ACCOUNT_FILE)}


def public_group_id():
    group = unwrap(request("GET", "/groups/public"))
    if isinstance(group, list):
        group = group[0]
    return group["_id"]


def list_contests():
    gid = public_group_id()
    return unwrap(request("GET", "/contests/group/" + gid))


def find_contest(contest):
    contests = list_contests()
    if isinstance(contests, dict):
        contests = contests.get("contests", contests.get("rows", []))
    for row in contests:
        if row.get("_id") == contest or row.get("name") == contest:
            return row
    raise RuntimeError(f"contest not found: {contest}")


def list_problems(contest):
    row = find_contest(contest)
    problems = unwrap(request("GET", "/problems/contest/" + row["_id"]))
    if isinstance(problems, dict):
        problems = problems.get("problems", problems.get("rows", []))
    return row, problems


def find_problem(contest, problem_slug):
    row, problems = list_problems(contest)
    for problem in problems:
        if problem.get("name") == problem_slug or problem.get("_id") == problem_slug:
            return row, problem
    raise RuntimeError(f"problem not found: {contest}/{problem_slug}")


def template_files(problem_id, account_label):
    row = account(account_label)
    value = unwrap(
        request(
            "GET",
            f"/problems/{problem_id}/template?userId={row['uid']}",
            row["cookie"],
        )
    )
    return row, value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("accounts", help="show labels, UIDs, expiry; never cookies")

    sub.add_parser("contests", help="discover public contests")

    p = sub.add_parser("problems", help="list problems in a contest")
    p.add_argument("contest")

    p = sub.add_parser("resolve", help="resolve contest/problem slug to ObjectId")
    p.add_argument("contest")
    p.add_argument("problem", nargs="?")

    p = sub.add_parser("problem", help="get problem detail")
    p.add_argument("problem_id")

    p = sub.add_parser("template", help="list template files (no full content)")
    p.add_argument("problem_id")
    p.add_argument("--account", required=True)
    p.add_argument("--write-dir", help="write editable files to this directory")
    p.add_argument("--show-content", action="store_true")

    p = sub.add_parser("download-template", help="download template ZIP")
    p.add_argument("problem_id")
    p.add_argument("output")
    p.add_argument("--account", required=True)

    p = sub.add_parser("history", help="this account's submissions for a problem")
    p.add_argument("problem_id")
    p.add_argument("--account", required=True)

    p = sub.add_parser("detail", help="submission detail")
    p.add_argument("submission_id")
    p.add_argument("--account", required=True)

    p = sub.add_parser("ranking", help="problem ranking")
    p.add_argument("problem_id")
    p.add_argument("--account")
    p.add_argument("--q", default="")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--size", type=int, default=100)

    p = sub.add_parser("check-auth", help="check restricted history for all accounts")
    p.add_argument("--problem", required=True)

    p = sub.add_parser("get", help="raw GET path under /api")
    p.add_argument("path")
    p.add_argument("--account")

    p = sub.add_parser("download", help="raw binary GET path under /api")
    p.add_argument("path")
    p.add_argument("output")
    p.add_argument("--account")

    p = sub.add_parser("set-cookie", help="save a hidden cannjudge_auth cookie")
    p.add_argument("account")

    args = parser.parse_args()

    if args.command == "accounts":
        value = [
            {
                "label": row["label"],
                "uid": row.get("uid"),
                "expires_utc": datetime.fromtimestamp(
                    row["expires_epoch"], timezone.utc
                ).isoformat()
                if row.get("expires_epoch")
                else None,
                "usable_for_submit": row.get("usable_for_submit", True),
            }
            for row in accounts()
        ]
    elif args.command == "contests":
        value = list_contests()
    elif args.command == "problems":
        value = list_problems(args.contest)[1]
    elif args.command == "resolve":
        c, problems = list_problems(args.contest)
        value = {"contest": c}
        if args.problem:
            value["problem"] = find_problem(args.contest, args.problem)[1]
    elif args.command == "problem":
        value = unwrap(request("GET", "/problems/" + args.problem_id))
    elif args.command == "template":
        row, value = template_files(args.problem_id, args.account)
        files = value.get("files", [])
        summary = []
        for f in files:
            content = f.get("content", "")
            item = {
                "path": f.get("path"),
                "editable": f.get("editable"),
                "bytes": len(content.encode()),
            }
            if args.show_content:
                item["content"] = content
            summary.append(item)
        if args.write_dir:
            root = Path(args.write_dir)
            for f in files:
                if not f.get("editable"):
                    continue
                target = root / f["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f.get("content", ""))
            value = {"account": args.account, "uid": row["uid"], "written_to": str(root), "files": summary}
        else:
            value = {"files": summary}
    elif args.command == "download-template":
        row = account(args.account)
        raw = request(
            "GET",
            f"/problems/{args.problem_id}/package?userId={row['uid']}",
            row["cookie"],
            binary=True,
        )
        Path(args.output).write_bytes(raw)
        value = {"saved_to": args.output, "bytes": len(raw)}
    elif args.command == "history":
        row = account(args.account)
        value = request(
            "GET",
            f"/submissions/user/{row['uid']}/problem/{args.problem_id}",
            row["cookie"],
        )
    elif args.command == "detail":
        row = account(args.account)
        value = request(
            "GET",
            f"/submissions/{args.submission_id}?userId={row['uid']}",
            row["cookie"],
        )
    elif args.command == "ranking":
        row = account(args.account) if args.account else None
        qs = f"?page={args.page}&size={args.size}&q={urllib.parse.quote(args.q)}"
        value = request(
            "GET",
            f"/problems/{args.problem_id}/ranking{qs}",
            row["cookie"] if row else None,
        )
    elif args.command == "check-auth":
        value = []
        for row in accounts():
            if not row.get("cookie"):
                continue
            try:
                history = request(
                    "GET",
                    f"/submissions/user/{row['uid']}/problem/{args.problem}",
                    row["cookie"],
                )
                value.append(
                    {
                        "label": row["label"],
                        "uid": row["uid"],
                        "authenticated_history_http": 200,
                        "history_count": len(history) if isinstance(history, list) else None,
                    }
                )
            except RuntimeError as error:
                value.append({"label": row["label"], "uid": row.get("uid"), "error": str(error)})
    elif args.command == "set-cookie":
        value = save_cookie(args.account)
    else:
        row = account(args.account) if args.account else None
        if args.command == "download":
            raw = request("GET", args.path, row["cookie"] if row else None, binary=True)
            Path(args.output).write_bytes(raw)
            value = {"saved_to": args.output, "bytes": len(raw)}
        else:
            value = request("GET", args.path, row["cookie"] if row else None)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
