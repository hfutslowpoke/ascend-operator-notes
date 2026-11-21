#!/usr/bin/env python3
"""Score a CannJudge submission and show per-case headroom.

Inputs are offline JSON files:

  * a platform detail JSON (contains ``result``), and/or
  * a ranking JSON (contains ``testcases`` with ``tbest`` and ``rows``).

No network and no credentials are used.  The platform is authoritative; this
tool only re-computes the published per-case formula so the next experiment can
be chosen by expected gain rather than by the largest raw ratio.

Examples:
  python3 cann_score.py --detail detail.json --ranking ranking.json
  python3 cann_score.py --ranking ranking.json --sid <SID> --target 60
  python3 cann_score.py --detail detail.json --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

LOG_1_5 = math.log(1.5)


def case_score(time: float | None, tbest: float | None) -> float | None:
    """CannJudge per-case score: 100 / (1 + log_1.5(time / tbest))."""
    if not time or not tbest or time <= 0 or tbest <= 0:
        return None
    ratio = float(time) / float(tbest)
    if ratio <= 1.0:
        return 100.0
    return 100.0 / (1.0 + math.log(ratio) / LOG_1_5)


def ratio_for_score(score: float) -> float:
    """Inverse of case_score: allowed time/tbest ratio for a target score."""
    score = max(0.01, min(100.0, float(score)))
    if score >= 100.0:
        return 1.0
    return math.exp(LOG_1_5 * (100.0 / score - 1.0))


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def load_json(path: str | None) -> dict:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    # Some endpoints wrap the payload in {"data": {...}}.
    if "data" in obj and isinstance(obj["data"], dict) and not (
        "result" in obj or "rows" in obj or "testcases" in obj
    ):
        return obj["data"]
    return obj


def detail_rows(detail: dict) -> list[dict]:
    rows = detail.get("result")
    if isinstance(rows, list):
        return rows
    data = detail.get("data")
    if isinstance(data, dict) and isinstance(data.get("result"), list):
        return data["result"]
    return []


def ranking_parts(ranking: dict) -> tuple[list[dict], list[dict]]:
    rows = ranking.get("rows")
    testcases = ranking.get("testcases")
    data = ranking.get("data")
    if isinstance(data, dict):
        rows = rows if isinstance(rows, list) else data.get("rows")
        testcases = testcases if isinstance(testcases, list) else data.get("testcases")
    return (
        rows if isinstance(rows, list) else [],
        testcases if isinstance(testcases, list) else [],
    )


def pick_row(rows: list[dict], sid: str | None) -> dict | None:
    if not rows:
        return None
    if sid:
        for row in rows:
            candidates = {
                row.get("submission_id"),
                row.get("_id"),
                (row.get("submission") or {}).get("_id"),
            }
            if sid in candidates:
                return row
        raise SystemExit(f"ranking does not contain submission_id={sid}")
    if len(rows) == 1:
        return rows[0]
    raise SystemExit(
        "ranking contains multiple rows (all teams); pass --sid to select your submission"
    )


def testcase_best(testcases: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in testcases:
        key = item.get("id") or item.get("_id") or item.get("testcase_id")
        value = item.get("tbest", item.get("best_time"))
        if key and isinstance(value, (int, float)) and value > 0:
            result[str(key)] = float(value)
    return result


def analyze(
    results: list[dict],
    testcases: list[dict],
    target: float | None = None,
) -> dict:
    best = testcase_best(testcases)
    cases: list[dict] = []
    for index, row in enumerate(results):
        testcase_id = (
            row.get("testcase_id")
            or row.get("testcaseId")
            or row.get("id")
            or row.get("_id")
        )
        time = row.get("time")
        if time is None:
            time = row.get("cost") or row.get("duration")
        tbest = best.get(str(testcase_id)) if testcase_id is not None else None
        if tbest is None:
            tbest = row.get("best_time") or row.get("tbest")
        time = float(time) if isinstance(time, (int, float)) and time > 0 else None
        tbest = float(tbest) if isinstance(tbest, (int, float)) and tbest > 0 else None
        status = row.get("testcase_status") or row.get("status") or "unknown"
        score = case_score(time, tbest)
        ratio = (time / tbest) if time and tbest else None
        case = {
            "case": index,
            "testcase_id": testcase_id,
            "status": status,
            "time": _round(time, 6),
            "tbest": _round(tbest, 6),
            "ratio": _round(ratio, 3),
            "score": _round(score, 2),
            "gain_2x": _round(
                (case_score(time / 2.0, tbest) or 0.0) - score, 2
            )
            if score is not None
            else None,
            "max_gain": _round(100.0 - score, 2) if score is not None else None,
            "factor_to_50": _round(max(1.0, ratio / 1.5), 2)
            if ratio is not None
            else None,
        }
        cases.append(case)

    valid = [c for c in cases if c["score"] is not None]
    total = sum(c["score"] for c in valid) / len(valid) if valid else None
    summary: dict = {
        "cases": cases,
        "valid_cases": len(valid),
        "total": _round(total, 2),
        "target": _round(target, 2),
        "gap": _round(target - total, 2) if target is not None and total is not None else None,
        "all_at_1_5x": _round(
            sum(max(c["score"], 50.0) for c in valid) / len(valid), 2
        )
        if valid
        else None,
        "all_at_1_3x": _round(
            sum(
                max(c["score"], case_score(1.3, 1.0) or 0.0) for c in valid
            )
            / len(valid),
            2,
        )
        if valid
        else None,
    }

    if target is not None and valid:
        current_sum = sum(c["score"] for c in valid)
        needed_sum = target * len(valid)
        for case in valid:
            required = needed_sum - (current_sum - case["score"])
            case["required_score_for_target"] = _round(required, 2)
            if required <= case["score"]:
                case["target_state"] = "already_enough"
                case["required_ratio_for_target"] = None
            elif required > 100.0:
                case["target_state"] = "impossible_alone"
                case["required_ratio_for_target"] = None
            else:
                case["target_state"] = "needs_improvement"
                case["required_ratio_for_target"] = _round(
                    ratio_for_score(required), 3
                )
    return summary


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(summary: dict) -> None:
    print("# CannScore")
    print()
    print(f"- 当前总分: {fmt(summary['total'])}")
    if summary.get("target") is not None:
        print(f"- 目标总分: {fmt(summary['target'])}")
        print(f"- 差距: {fmt(summary['gap'])}")
    print(f"- 有效 case: {summary['valid_cases']}")
    print(f"- 全部到 1.5×tbest: {fmt(summary.get('all_at_1_5x'))}")
    print(f"- 全部到 1.3×tbest: {fmt(summary.get('all_at_1_3x'))}")
    print()
    header = (
        f"{'case':>4} {'time':>12} {'tbest':>12} {'ratio':>9} "
        f"{'score':>7} {'gain_2x':>8} {'max_gain':>9} {'to_50':>7} "
        f"{'target_ratio':>13} {'state':>18}"
    )
    print(header)
    for case in summary["cases"]:
        print(
            f"{case['case']:>4} {fmt(case['time'], 3):>12} {fmt(case['tbest'], 3):>12} "
            f"{fmt(case['ratio'], 2):>9} {fmt(case['score']):>7} "
            f"{fmt(case['gain_2x']):>8} {fmt(case['max_gain']):>9} "
            f"{fmt(case['factor_to_50']):>7} "
            f"{fmt(case.get('required_ratio_for_target'), 2):>13} "
            f"{case.get('target_state', '-'):>18}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detail", help="platform detail JSON (result array)")
    parser.add_argument("--ranking", help="platform ranking JSON (testcases/rows)")
    parser.add_argument("--sid", help="submission_id to select from ranking rows")
    parser.add_argument("--target", type=float, help="target total score")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    detail = load_json(args.detail)
    ranking = load_json(args.ranking)
    results = detail_rows(detail)
    rows, testcases = ranking_parts(ranking)

    if not results and rows:
        row = pick_row(rows, args.sid)
        results = list(row.get("result") or []) if row else []
        if row and not args.sid:
            print(
                f"[info] single ranking row; score={row.get('score')} "
                f"submission_id={row.get('submission_id')}",
                file=sys.stderr,
            )

    if not results:
        raise SystemExit("no result rows: pass --detail or a ranking with rows/result")
    if not testcases:
        print(
            "[warn] no ranking testcases; falling back to per-row best_time",
            file=sys.stderr,
        )

    summary = analyze(results, testcases, args.target)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
