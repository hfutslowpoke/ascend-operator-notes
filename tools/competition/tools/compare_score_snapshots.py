#!/usr/bin/env python3
"""Compare public TBest and unchanged submissions in saved ranking snapshots."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from scoring_snapshot import point_score


def load_snapshot(path):
    raw = json.loads(path.read_text())
    pages = raw['pages']
    cases = {str(case.get('_id') or case.get('id')): case
             for page in pages for case in page.get('testcases', [])}
    rows = {}
    public_ids = set()
    for page in pages:
        for row in page.get('rows', []):
            sid = row.get('submission_id') or row.get('submission', {}).get('_id')
            if not sid or row.get('status') != 'Pass':
                continue
            public = {}
            for result in row.get('result', []):
                case_id = str(result.get('testcase_id'))
                case = cases.get(case_id, {})
                if case.get('hidden') is True or result.get('testcase_status') == 'Hidden':
                    continue
                elapsed, best = result.get('time'), case.get('tbest')
                if result.get('testcase_status') != 'Pass' or elapsed is None or best is None:
                    continue
                if not (math.isfinite(float(elapsed)) and float(elapsed) > 0):
                    continue
                public[case_id] = dict(time_us=float(elapsed), tbest_us=float(best),
                                       score=point_score(elapsed, best))
                public_ids.add(case_id)
            total = float(row['score'])
            count = len(cases)
            rows[str(sid)] = dict(score=total, public=public,
                public_sum=sum(case['score'] for case in public.values()),
                hidden_sum=total * count - sum(case['score'] for case in public.values()))
    return dict(path=str(path), queried_utc=raw['queried_utc'], rows=rows, count=len(cases),
                public_tbest={case_id: cases[case_id].get('tbest') for case_id in sorted(public_ids)})


def compare(left, right):
    assert left['count'] == right['count'], 'Different testcase sets need a separate analysis'
    same_public_tbest = left['public_tbest'] == right['public_tbest']
    fixed = []
    excluded = []
    for sid in sorted(set(left['rows']) & set(right['rows'])):
        a, b = left['rows'][sid], right['rows'][sid]
        times = lambda row: {key: case['time_us'] for key, case in row['public'].items()}
        if not a['public'] or times(a) != times(b):
            excluded.append(sid)
            continue
        count = left['count']
        hidden_count = count - len(a['public'])
        residual = b['hidden_sum'] - a['hidden_sum']
        # Each displayed total was rounded to 0.01; both endpoints contribute.
        uncertainty = count * 0.01
        fixed.append(dict(submission_id=sid, old_score=a['score'], new_score=b['score'],
            total_score_delta=b['score'] - a['score'], hidden_sum_delta=residual,
            hidden_sum_rounding_interval=[residual - uncertainty, residual + uncertainty],
            hidden_mean_delta=residual / hidden_count if hidden_count else None))
    deltas = [row['total_score_delta'] for row in fixed]
    drops = [row for row in fixed if row['hidden_sum_rounding_interval'][1] < -1e-9]
    rises = [row for row in fixed if row['hidden_sum_rounding_interval'][0] > 1e-9]
    return dict(reference=left['path'], candidate=right['path'],
        reference_queried_utc=left['queried_utc'], candidate_queried_utc=right['queried_utc'],
        same_public_tbest=same_public_tbest,
        reference_public_tbest=left['public_tbest'], candidate_public_tbest=right['public_tbest'],
        common_fixed_sid_count=len(fixed), fixed_sid_records=fixed,
        excluded_changed_public_timings=excluded,
        fixed_sid_score_delta_min=min(deltas) if deltas else None,
        fixed_sid_score_delta_max=max(deltas) if deltas else None,
        fixed_sid_score_delta_median=statistics.median(deltas) if deltas else None,
        hidden_scale_decrease_detected=bool(drops), hidden_scale_increase_detected=bool(rises),
        hidden_scale_drop_sids=[row['submission_id'] for row in drops],
        assessment=('A hidden scoring-scale decrease is consistent with unchanged-SID evidence; historical totals need caution.'
                    if drops else 'No hidden scoring-scale decrease resolved by the available unchanged-SID snapshots.'),
        limitation='Unchanged scores do not prove identical hidden TBest: totals are rounded and visible fixed submissions are a limited sample. Do not apply a common score-offset correction or infer an independent target-point time.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Keep the original comparison immutable'
    result = compare(load_snapshot(args.reference), load_snapshot(args.candidate))
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['fixed_sid_records', 'reference_public_tbest', 'candidate_public_tbest']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
