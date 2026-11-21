#!/usr/bin/env python3
"""Generic CannJudge submitter with template checks, global ledger and SID polling.

No credentials are printed or stored in the ledger. Cookies stay in the
private account file managed by cann_client.py.

The ledger is global for this user (default:
~/.local/state/cannjudge/submit-ledger), so every session sees the same
submissions. The local quota mirror counts every problem for the account in
the platform's 12:00->12:00 Beijing window. A per-ledger flock only prevents
two sessions from POSTing concurrently; it is not pacing or a cooldown.

Default is dry-run. Pass --submit to POST once, after explicit authorization.

Examples:
  python3 cann_submit.py --problem-id <PID> --account A \
      --source-dir ./src --tag my-r1

  CANNJUDGE_ALLOW_SUBMIT=1 python3 cann_submit.py --problem-id <PID> --account A \
      --source-dir ./src --tag my-r1 --submit

  # old host/kernel templates only; verify the returned detail afterwards
  python3 cann_submit.py ... --compat auto --submit
"""
import argparse
from contextlib import contextmanager, nullcontext
import datetime
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cann_client  # noqa: E402
import cann_quota  # noqa: E402

ACTIVE_STATUS = {
    "Queued",
    "Queuing",
    "Pending",
    "Running",
    "Judging",
    "Compiling",
    "Wait",
    "Waiting",
}
PAYLOAD_LIMIT = 609980
CST = datetime.timezone(datetime.timedelta(hours=8))


def default_ledger_dir() -> Path:
    """One ledger shared by every session/account for this user.

    The public toolkit is read-only; the ledger is local state, not repo state.
    Override with CANNJUDGE_SUBMIT_LEDGER or ACW_SUBMIT_LEDGER.
    """
    value = (os.environ.get("CANNJUDGE_SUBMIT_LEDGER")
             or os.environ.get("ACW_SUBMIT_LEDGER"))
    return Path(value) if value else Path.home() / ".local/state/cannjudge/submit-ledger"


@contextmanager
def submission_lock(ledger: Path):
    """Serialize POST/ledger decisions for one global ledger.

    This is not pacing or a cooldown; it only prevents two sessions from
    double-POSTing or racing on the same digest.
    """
    path = Path(os.environ.get("CANNJUDGE_SUBMIT_LOCK") or ledger / ".submit.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_digest(files):
    h = hashlib.sha256()
    for item in sorted(files, key=lambda x: x["path"]):
        h.update(item["path"].encode() + b"\0" + item["content"].encode() + b"\0")
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".record-", delete=False
    ) as stream:
        temp = Path(stream.name)
        try:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def load_template(problem_id, account_label):
    row, value = cann_client.template_files(problem_id, account_label)
    value = cann_client.unwrap(value)
    files = value.get("files", [])
    if not files:
        raise SystemExit("template returned no files; check problem id/account")
    return row, files


def read_sources(template, source_dir, overrides):
    root = Path(source_dir).resolve()
    selected = []
    missing = []
    for f in template:
        if f.get("editable") is not True:
            continue
        path = f.get("path")
        local = overrides.get(path)
        if local is None:
            local = root / path
        else:
            local = Path(local)
            if not local.is_absolute():
                local = root / local
        if not local.is_file():
            missing.append((path, str(local)))
            continue
        raw = local.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise SystemExit(f"{local} is not UTF-8 text; templates expect source text")
        if "\x00" in text:
            raise SystemExit(f"{local} contains NUL bytes; not a source file")
        selected.append(
            {
                "path": path,
                "content": text,
                "bytes": len(raw),
                "sha256": sha_file(raw),
                "local": str(local),
            }
        )
    if missing:
        lines = "\n".join(f"  template path: {p}\n  looked for:   {l}" for p, l in missing)
        raise SystemExit("missing editable files:\n" + lines + "\nuse --file path=local to map")
    if not selected:
        raise SystemExit("template has no editable files; use --all-files only for local inspection")
    return selected


def all_sources(template, source_dir):
    root = Path(source_dir).resolve()
    selected = []
    for f in template:
        path = f.get("path")
        local = root / path
        if not local.is_file():
            raise SystemExit(f"missing template file: {local}")
        raw = local.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise SystemExit(f"{local} is not UTF-8 text")
        selected.append(
            {
                "path": path,
                "content": text,
                "bytes": len(raw),
                "sha256": sha_file(raw),
                "local": str(local),
            }
        )
    return selected


def compat_fields(selected):
    fields = {}
    mapping = [
        ("host_cpp", lambda p: p.startswith("op_host/") and p.endswith(".cpp")),
        ("kernel_cpp", lambda p: p.startswith("op_kernel/") and p.endswith(".cpp")),
        ("tiling_key_h", lambda p: "tiling_key" in p and p.endswith(".h")),
        ("tiling_h", lambda p: "tiling" in p and p.endswith(".h") and "tiling_key" not in p),
    ]
    used = set()
    for key, pred in mapping:
        candidates = [x for x in selected if pred(x["path"]) and x["path"] not in used]
        if candidates:
            item = sorted(candidates, key=lambda x: x["path"])[0]
            fields[key] = item["content"]
            used.add(item["path"])
    return fields


def count_problem_window(history, reset_hour=12):
    if not isinstance(history, list):
        return 0
    start, end = cann_quota.quota_window(reset_hour=reset_hour)
    return len(cann_quota.count_window(history, start, end))


def count_today(history):
    """Legacy helper for submit_background; uses the platform 12:00 window."""
    return count_problem_window(history, reset_hour=12)


def resolve_contest(problem_id):
    detail = cann_client.unwrap(
        cann_client.request("GET", "/problems/" + problem_id)
    )
    if isinstance(detail, dict):
        value = detail.get("contest_id")
        if value:
            return value
        contest = detail.get("contest")
        if isinstance(contest, dict):
            return contest.get("_id")
    return None


def count_account_window(row, contest, reset_hour):
    """Count every problem for this account in the current 12:00->12:00 window."""
    start, end = cann_quota.quota_window(reset_hour=reset_hour)
    _, problems = cann_client.list_problems(contest)
    total = 0
    per_problem = {}
    for problem in problems:
        history = cann_client.unwrap(
            cann_client.request(
                "GET",
                f"/submissions/user/{row['uid']}/problem/{problem['_id']}",
                row["cookie"],
            )
        )
        selected = cann_quota.count_window(history or [], start, end)
        if selected:
            per_problem[problem.get("name") or problem["_id"]] = len(selected)
            total += len(selected)
    return total, per_problem, start, end


def find_prior(ledger_dir: Path, problem_id: str, digest: str):
    base = ledger_dir / problem_id
    for name in (f"{digest}.accepted.json", f"{digest}.json"):
        path = base / name
        if path.exists():
            try:
                return path, json.loads(path.read_text())
            except ValueError:
                return path, {}
    return None, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--submit", action="store_true", help="actually POST (default: dry-run)")
    ap.add_argument("--no-poll", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=3)
    ap.add_argument("--poll-timeout", type=int, default=3600)
    ap.add_argument(
        "--ledger-dir",
        default=str(default_ledger_dir()),
        help="global ledger shared by all sessions; default: ~/.local/state/cannjudge/submit-ledger",
    )
    ap.add_argument(
        "--max-per-day",
        type=int,
        default=50,
        help="local mirror of the platform limit (50/account/12:00 window); 0 disables",
    )
    ap.add_argument(
        "--quota-scope",
        choices=["account", "problem"],
        default="account",
        help="account counts every problem in the 12:00->12:00 Beijing window; problem is the legacy per-problem count",
    )
    ap.add_argument("--contest", help="contest name/id for account-wide quota; inferred from the problem when omitted")
    ap.add_argument("--reset-hour", type=int, default=12, help="Beijing-time quota reset hour")
    ap.add_argument("--compat", choices=["none", "auto"], default="none")
    ap.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="local safety override; platform may allow reposting the same source",
    )
    ap.add_argument("--all-files", action="store_true", help="use all template files for inspection")
    ap.add_argument("--file", action="append", default=[], metavar="PATH=LOCAL",
                    help="override a template path with a local file; repeatable")
    args = ap.parse_args()

    allowed = (os.environ.get("CANNJUDGE_ALLOW_SUBMIT") == "1"
               or os.environ.get("ACW_ALLOW_SUBMIT") == "1")
    if args.submit and not allowed:
        raise SystemExit(
            "refusing POST: set CANNJUDGE_ALLOW_SUBMIT=1 or ACW_ALLOW_SUBMIT=1 "
            "only after dry-run and explicit authorization"
        )

    overrides = {}
    for item in args.file:
        if "=" not in item:
            raise SystemExit(f"--file expects PATH=LOCAL, got {item!r}")
        path, local = item.split("=", 1)
        overrides[path] = local

    if args.all_files and args.submit:
        raise SystemExit("--all-files is inspection-only; refusing to submit non-editable files")
    row, template = load_template(args.problem_id, args.account)
    if args.all_files:
        selected = all_sources(template, args.source_dir)
        print("[warn] --all-files includes non-editable files; use only for inspection", file=sys.stderr)
    else:
        selected = read_sources(template, args.source_dir, overrides)

    with submission_lock(Path(args.ledger_dir)) if args.submit else nullcontext():
        acceptance = prepare_submission(args, row, selected)
    if acceptance is None or args.no_poll:
        return 0
    return poll_submission(args, row, *acceptance)


def prepare_submission(args, row, selected):
    digest = source_digest(selected)
    ledger = Path(args.ledger_dir)
    prior_path, prior = find_prior(ledger, args.problem_id, digest)
    if prior and not args.allow_duplicate:
        status = prior.get("status")
        if status not in ("rate_limited_no_post", "paused_no_post"):
            sid = prior.get("submission_id")
            raise SystemExit(
                f"duplicate source digest already recorded: {prior_path}\n"
                f"  status={status} sid={sid}\n"
                "use --allow-duplicate only if you really intend to repost"
            )

    history = cann_client.request(
        "GET",
        f"/submissions/user/{row['uid']}/problem/{args.problem_id}",
        row["cookie"],
    )
    history = cann_client.unwrap(history)
    quota_contest = args.contest
    quota_start = quota_end = None
    per_problem = {}
    if args.quota_scope == "account":
        quota_contest = quota_contest or resolve_contest(args.problem_id)
        if quota_contest:
            try:
                used, per_problem, quota_start, quota_end = count_account_window(
                    row, quota_contest, args.reset_hour
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "[warn] account-wide quota lookup failed; falling back to "
                    f"per-problem count: {exc}",
                    file=sys.stderr,
                )
                used = count_problem_window(history, args.reset_hour)
        else:
            print(
                "[warn] cannot infer contest for account-wide quota; "
                "falling back to per-problem count",
                file=sys.stderr,
            )
            used = count_problem_window(history, args.reset_hour)
    else:
        used = count_problem_window(history, args.reset_hour)
    if args.max_per_day and used >= args.max_per_day:
        raise SystemExit(
            f"local quota guard: {used}/{args.max_per_day} submissions in the current "
            f"{args.reset_hour}:00->{args.reset_hour}:00 Beijing window for {args.account}; "
            "check the platform history before overriding"
        )

    payload_files = [{"path": x["path"], "content": x["content"]} for x in selected]
    payload = {
        "problemId": args.problem_id,
        "userId": row["uid"],
        "files": payload_files,
    }
    compat = {}
    if args.compat == "auto":
        compat = compat_fields(selected)
        payload.update(compat)
    size = len(json.dumps(payload, ensure_ascii=False).encode())
    if size >= PAYLOAD_LIMIT:
        raise SystemExit(f"payload too large: {size} bytes >= {PAYLOAD_LIMIT}")

    summary = {
        "problem_id": args.problem_id,
        "account": args.account,
        "uid": row["uid"],
        "tag": args.tag,
        "source_digest": digest,
        "payload_bytes": size,
        "compat_fields": sorted(compat),
        "files": [
            {"path": x["path"], "bytes": x["bytes"], "sha256": x["sha256"]}
            for x in selected
        ],
        "posts_today_before": used,
        "max_per_day": args.max_per_day,
        "quota_scope": args.quota_scope,
        "quota_contest": quota_contest,
        "quota_window_start": quota_start.isoformat() if quota_start else None,
        "quota_window_end": quota_end.isoformat() if quota_end else None,
        "quota_per_problem": per_problem,
        "submit": args.submit,
    }

    plan_path = ledger / args.problem_id / f"{digest}.plan.json"
    dump(plan_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.submit:
        print(f"\n[dry-run] no POST. plan: {plan_path}", file=sys.stderr)
        return None

    accepted_path = ledger / args.problem_id / f"{digest}.accepted.json"
    if accepted_path.exists():
        try:
            old = json.loads(accepted_path.read_text())
        except ValueError:
            old = {}
        if old.get("submission_id") or old.get("status") not in ("rate_limited_no_post", "paused_no_post"):
            raise SystemExit(f"acceptance record already exists: {accepted_path}; refusing to overwrite")

    record = dict(summary, created_utc=now_utc(), status="post_started")
    dump(accepted_path, record)
    try:
        response = cann_client.request(
            "POST", "/submissions/submit", row["cookie"], body=payload
        )
    except cann_client.CannHttpError as error:
        if error.status == 429:
            record["status"] = "rate_limited_no_post"
            record["retry_after_seconds"] = error.metadata.get("retry_after_seconds")
            dump(accepted_path, record)
            raise SystemExit(
                f"rate limited (429); no acceptance recorded. "
                f"retry after {record['retry_after_seconds']}s"
            )
        record["status"] = "post_error_check_history"
        record["post_error"] = f"HTTP {error.status}"
        record["post_error_metadata"] = error.metadata
        dump(accepted_path, record)
        raise SystemExit(
            f"POST failed HTTP {error.status}; check account history before any retry"
        )

    sid = (response.get("data", response) or {}).get("submissionId")
    record["post_response"] = response
    if not sid:
        record["status"] = "ambiguous_do_not_retry"
        dump(accepted_path, record)
        raise SystemExit("ambiguous POST: no submissionId; check account history manually")
    record.update(submission_id=sid, status="accepted", accepted_utc=now_utc())
    dump(accepted_path, record)
    with (ledger / args.problem_id / "ledger.jsonl").open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "tag": args.tag,
                    "problem_id": args.problem_id,
                    "account": args.account,
                    "submission_id": sid,
                    "source_digest": digest,
                    "accepted_utc": record["accepted_utc"],
                }
            )
            + "\n"
        )
    print(f"accepted sid={sid}", flush=True)
    return accepted_path, record


def poll_submission(args, row, accepted_path, record):
    sid = record["submission_id"]
    detail_path = accepted_path.with_name(record["source_digest"] + ".detail.json")
    deadline = time.time() + args.poll_timeout
    while True:
        try:
            detail = cann_client.request(
                "GET", f"/submissions/{sid}?userId={row['uid']}", row["cookie"]
            )
        except Exception as error:  # keep polling on transient GET errors
            record["last_get_error"] = str(error)
            dump(accepted_path, record)
            if time.time() > deadline:
                raise SystemExit("GET polling deadline")
            time.sleep(max(3, args.poll_interval))
            continue

        dump(detail_path, detail)
        results = detail.get("result", []) if isinstance(detail, dict) else []
        times = [row.get("time") for row in results[:20]] if isinstance(results, list) else []
        print(
            json.dumps(
                {
                    "sid": sid,
                    "status": detail.get("status") if isinstance(detail, dict) else None,
                    "valid": detail.get("valid") if isinstance(detail, dict) else None,
                    "times": times,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        status = detail.get("status") if isinstance(detail, dict) else None
        if status not in ACTIVE_STATUS:
            record.update(
                status="platform_terminal",
                platform_status=status,
                valid=detail.get("valid") if isinstance(detail, dict) else None,
                finished_utc=now_utc(),
                detail_path=str(detail_path),
            )
            dump(accepted_path, record)
            return 0
        if time.time() > deadline:
            record.update(status="poll_deadline", detail_path=str(detail_path))
            dump(accepted_path, record)
            return 1
        time.sleep(max(1, args.poll_interval))


if __name__ == "__main__":
    canonical = Path(__file__).resolve().parents[3] / "cannjudge/tools/cann_submit.py"
    os.execv(sys.executable, [sys.executable, str(canonical), *sys.argv[1:]])
