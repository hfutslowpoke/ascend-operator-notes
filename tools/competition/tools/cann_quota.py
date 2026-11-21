#!/usr/bin/env python3
"""Count CannJudge submissions in the Beijing-time quota window.

Rules implemented here:

* one quota day runs from 12:00 to 12:00 Beijing time (UTC+8);
* each account has its own 50-submission limit and its own cooldown;
* counts must include every problem in the target contest, not just the
  problem currently being optimized.

The platform is authoritative.  This helper only predicts the local limit and
prints the evidence (per-problem counts, last submission, next reset).  It
never POSTs and never prints cookies.

Examples:
  python3 cann_quota.py --contest gcc_2026_mfg
  python3 cann_quota.py --contest gcc_2026_mfg --account B --details
  python3 cann_quota.py --contest gcc_2026_mfg --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cann_client  # noqa: E402

BEIJING = datetime.timezone(datetime.timedelta(hours=8))
DEFAULT_RESET_HOUR = 12
DEFAULT_LIMIT = 50
INTERRUPTED = {"Judge Interrupted", "judge interrupted"}


def parse_time(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(BEIJING)


def quota_window(
    now: datetime.datetime | None = None, reset_hour: int = DEFAULT_RESET_HOUR
) -> tuple[datetime.datetime, datetime.datetime]:
    """Current [start, end) quota window in Beijing time."""
    current = (now or datetime.datetime.now(BEIJING)).astimezone(BEIJING)
    start = current.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
    if current < start:
        start -= datetime.timedelta(days=1)
    return start, start + datetime.timedelta(days=1)


def count_window(
    rows: list[dict], start: datetime.datetime, end: datetime.datetime
) -> list[dict]:
    selected = []
    for row in rows or []:
        created = parse_time(row.get("create_time"))
        if created is not None and start <= created < end:
            selected.append(row)
    return selected


def account_quota(
    label: str,
    contest: str,
    *,
    reset_hour: int = DEFAULT_RESET_HOUR,
    limit: int = DEFAULT_LIMIT,
    details: bool = False,
    at: str | None = None,
) -> dict:
    row = cann_client.account(label)
    contest_row, problems = cann_client.list_problems(contest)
    reference = parse_time(at) if at else datetime.datetime.now(BEIJING)
    if reference is None:
        raise ValueError(f"invalid --at timestamp: {at}")
    start, end = quota_window(now=reference, reset_hour=reset_hour)
    per_problem = []
    all_rows: list[dict] = []
    for problem in problems:
        history = cann_client.unwrap(
            cann_client.request(
                "GET",
                f"/submissions/user/{row['uid']}/problem/{problem['_id']}",
                row["cookie"],
            )
        )
        selected = count_window(history or [], start, end)
        for item in selected:
            item["_problem"] = problem.get("name")
        if selected:
            per_problem.append(
                {
                    "problem": problem.get("name"),
                    "problem_id": problem.get("_id"),
                    "count": len(selected),
                    "last": max(
                        (parse_time(x.get("create_time")) for x in selected),
                        default=None,
                    ),
                }
            )
        all_rows.extend(selected)
    per_problem.sort(key=lambda x: (-x["count"], x["problem"] or ""))
    last = max((parse_time(x.get("create_time")) for x in all_rows), default=None)
    effective = [x for x in all_rows if x.get("status") not in INTERRUPTED]
    result = {
        "account": label,
        "uid": row.get("uid"),
        "contest": contest_row.get("name") or contest,
        "contest_id": contest_row.get("_id"),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "next_reset": end.isoformat(),
        "reference": reference.isoformat(),
        "limit": limit,
        "total": len(all_rows),
        "effective": len(effective),
        "remaining": max(0, limit - len(effective)),
        "last_submission": last.isoformat() if last else None,
        "per_problem": per_problem,
    }
    if last:
        result["minutes_since_last"] = round(
            (reference - last).total_seconds() / 60, 1
        )
    if details:
        result["submissions"] = [
            {
                "time": parse_time(x.get("create_time")).isoformat(),
                "status": x.get("status"),
                "sid": x.get("_id") or x.get("id"),
                "problem": x.get("_problem"),
            }
            for x in all_rows
        ]
    return result


def print_table(results: list[dict], show_details: bool) -> None:
    header = (
        f"{'account':7s} {'total':>5s} {'eff':>5s} {'limit':>5s} "
        f"{'remain':>6s} {'last':16s} {'reset':16s}"
    )
    print(header)
    for item in results:
        if item.get("error"):
            print(f"{item['account']:7s} ERROR {item['error']}")
            continue
        last = item.get("last_submission")
        last_text = last[5:16].replace("T", " ") if last else "-"
        reset = item.get("next_reset", "")[5:16].replace("T", " ")
        print(
            f"{item['account']:7s} {item['total']:5d} {item['effective']:5d} "
            f"{item['limit']:5d} {item['remaining']:6d} {last_text:16s} {reset:16s}"
        )
        if show_details:
            for row in item.get("per_problem", []):
                print(
                    f"  - {row['problem']}: {row['count']}  "
                    f"(last {row['last'].strftime('%m-%d %H:%M') if row.get('last') else '-'})"
                )
            if item.get("minutes_since_last") is not None:
                print(f"  last submission {item['minutes_since_last']} minutes ago")
            print(f"  window: {item['window_start']} -> {item['window_end']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--contest", default="gcc_2026_mfg")
    parser.add_argument(
        "--account",
        action="append",
        default=[],
        help="account label; repeatable; default: every configured account with a cookie",
    )
    parser.add_argument("--reset-hour", type=int, default=DEFAULT_RESET_HOUR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--at",
        help="reference time for a historical window, e.g. 2026-11-18T16:37:00+08:00",
    )
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    labels = args.account
    if not labels:
        labels = [
            row["label"]
            for row in cann_client.accounts()
            if row.get("label") and row.get("cookie")
        ]
    results = []
    for label in labels:
        try:
            results.append(
                account_quota(
                    label,
                    args.contest,
                    reset_hour=args.reset_hour,
                    limit=args.limit,
                    details=args.details,
                    at=args.at,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "account": label,
                    "error": str(exc),
                    "total": 0,
                    "effective": 0,
                    "limit": args.limit,
                    "remaining": 0,
                    "per_problem": [],
                }
            )
    if args.json:
        print(json.dumps({"quota": results}, ensure_ascii=False, indent=2))
    else:
        print_table(results, args.details)
        print(
            "\nquota day: 12:00 -> 12:00 Beijing time; "
            "each account is independent; 429 is not proof of quota exhaustion."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
