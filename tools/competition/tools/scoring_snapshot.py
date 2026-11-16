#!/usr/bin/env python3
"""Capture total and public contributions from the same live ranking snapshot."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cann_client


def point_score(elapsed, best):
    elapsed, best = float(elapsed), float(best)
    if not (math.isfinite(elapsed) and math.isfinite(best) and elapsed > 0 and best > 0):
        raise ValueError('Scoring requires finite positive times')
    return 100.0 / (1.0 + math.log(max(1.0, elapsed / best)) / math.log(1.5))


def parse_server_time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def positive_time(value):
    """A disclosed numeric time is evidence of visibility, even with stale flags."""
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def observe_testcases(testcases, results):
    """Keep disclosed values; only score cases with consistent same-snapshot data.

    Metadata without a hidden flag is the existing public API representation.
    A positive row time takes precedence for *display*, not for trusting a
    conflicting hidden flag or assigning a point score.
    """
    metadata, result_rows, order, problems = {}, {}, [], []
    for case in testcases:
        case_id = case.get('_id') or case.get('id')
        if case_id is None:
            problems.append(dict(testcase_id=None, issues=['metadata_missing_id']))
            continue
        key = str(case_id)
        if key in metadata:
            problems.append(dict(testcase_id=case_id, issues=['duplicate_metadata_id']))
        metadata[key] = case
    for row in results:
        case_id = row.get('testcase_id')
        if case_id is None:
            problems.append(dict(testcase_id=None, issues=['result_missing_testcase_id']))
            continue
        key = str(case_id)
        if key in result_rows:
            problems.append(dict(testcase_id=case_id, issues=['duplicate_result_id']))
        else:
            order.append(key)
        result_rows[key] = row
    order += [key for key in metadata if key not in result_rows]
    observations = []
    failure_statuses = {'Wrong Answer', 'Runtime Error', 'Time Limit Exceeded',
                        'Memory Limit Exceeded', 'Compilation Error', 'Failed', 'Fail', 'Error'}
    for key in order:
        case, row = metadata.get(key), result_rows.get(key)
        status = row.get('testcase_status') if row else None
        elapsed = row.get('time') if row else None
        disclosed = positive_time(elapsed)
        hidden = case.get('hidden') is True if case is not None else None
        issues = []
        if case is None:
            issues.append('missing_metadata')
        if row is None:
            issues.append('missing_result')
        # Preserve an actual time even if metadata or row status is lagging.
        if disclosed:
            visibility = 'public'
        elif status == 'Hidden' or hidden is True:
            visibility = 'hidden'
        elif case is not None or status is not None:
            visibility = 'public'
        else:
            visibility = 'unknown'
        if hidden is True and row is not None and (disclosed or status != 'Hidden'):
            issues.append('metadata_hidden_conflicts_with_visible_result')
        if status == 'Hidden' and (disclosed or hidden is False):
            issues.append('row_hidden_conflicts_with_visible_data')
        metadata_best = case.get('tbest') if case else None
        row_best = row.get('tbest') if row else None
        best = metadata_best if metadata_best is not None else row_best
        if (positive_time(metadata_best) and positive_time(row_best)
                and float(metadata_best) != float(row_best)):
            issues.append('conflicting_tbest')
        score = None
        if visibility == 'public':
            if not positive_time(best):
                issues.append('missing_or_invalid_tbest')
            if status == 'Pass' and not disclosed:
                issues.append('missing_or_invalid_pass_time')
            elif status != 'Pass' and status not in failure_statuses:
                issues.append('nonterminal_or_unknown_case_status')
            if not issues:
                score = point_score(elapsed, best) if status == 'Pass' else 0.0
        item = dict(testcase_id=(row['testcase_id'] if row else case.get('_id') or case.get('id')),
                    status=status, time_us=elapsed, tbest_us=best, point_score=score,
                    visibility=visibility, metadata_hidden=hidden,
                    metadata_tbest_us=metadata_best, row_tbest_us=row_best,
                    coverage_issues=issues)
        observations.append(item)
        if issues:
            problems.append(dict(testcase_id=item['testcase_id'], issues=issues))
    return observations, problems


def capture(problem_id, sid, cookie, raw_output=None, expected_create_time=None,
            expected_user_ids=None):
    stamp = datetime.now(timezone.utc).isoformat()
    first = cann_client.request('GET', f'/problems/{problem_id}/ranking?size=100&page=1', cookie)
    fetched_pages = [first]
    payload = first
    match = next((r for r in first.get('rows', [])
                  if r.get('submission_id') == sid or r.get('submission', {}).get('_id') == sid), None)
    for page in range(2, int(first.get('pages', 1)) + 1):
        if match is not None:
            break
        payload = cann_client.request('GET', f'/problems/{problem_id}/ranking?size=100&page={page}', cookie)
        fetched_pages.append(payload)
        match = next((r for r in payload.get('rows', [])
                      if r.get('submission_id') == sid or r.get('submission', {}).get('_id') == sid), None)
    if raw_output is not None:
        # Keep the board already fetched for this score, including unchanged
        # submissions. Later comparisons can distinguish scoring-scale drift
        # from the target's execution noise without another network request.
        destination = Path(raw_output)
        temporary = destination.with_suffix('.ranking.tmp')
        temporary.write_text(json.dumps({'queried_utc': stamp, 'submission_id': sid,
                                         'pages': fetched_pages},
                                        ensure_ascii=False, indent=2) + '\n')
        temporary.replace(destination)
    return from_pages(fetched_pages, sid, expected_create_time, expected_user_ids, stamp)


def from_pages(fetched_pages, sid, expected_create_time=None, expected_user_ids=None, stamp=None):
    """Compute a score from ONE already fetched observation, without network I/O."""
    stamp = stamp or datetime.now(timezone.utc).isoformat()
    first = fetched_pages[0]
    payload = first
    match = None
    for payload in fetched_pages:
        match = next((r for r in payload.get('rows', [])
                      if r.get('submission_id') == sid or r.get('submission', {}).get('_id') == sid), None)
        if match is not None:
            break
    identity = dict(expected_submission_time=expected_create_time)
    if match is None:
        user_ids = set(expected_user_ids or [])
        representative = next((r for page in fetched_pages for r in page.get('rows', [])
                               if r.get('user_id') in user_ids or
                               r.get('submitter', {}).get('_id') in user_ids), None)
        expected = parse_server_time(expected_create_time)
        actual = parse_server_time((representative or {}).get('create_time'))
        identity.update(identity_verified=False, superseded=bool(expected and actual and actual > expected),
                        ranking_representative={k: representative.get(k) for k in
                                                ['submission_id', 'create_time', 'status', 'rank']}
                        if representative else None)
        return {'queried_utc': stamp, 'submission_id': sid, 'available': False,
                'reason': 'SID is not a visible representative submission; no score inferred', **identity}
    ranking_time = match.get('create_time') or match.get('submission', {}).get('create_time')
    identity.update(ranking_submission_time=ranking_time)
    if expected_create_time is not None:
        expected = parse_server_time(expected_create_time)
        actual = parse_server_time(ranking_time)
        if expected is None or actual is None or expected != actual:
            return {'queried_utc': stamp, 'submission_id': sid, 'available': False,
                    'identity_verified': False, 'timestamp_mismatch': True,
                    'reason': 'Ranking SID matched but server creation time did not; score not attributed', **identity}
        identity['identity_verified'] = True
    else:
        identity['identity_verified'] = 'SID-only; no expected server creation time supplied'
    testcases = payload.get('testcases', first.get('testcases', []))
    observations, issues = observe_testcases(testcases, match.get('result', []))
    public = [case for case in observations if case['visibility'] == 'public']
    hidden_count = sum(case['visibility'] == 'hidden' for case in observations)
    total = float(match['score'])
    count = len(observations)
    reported_count = payload.get('total_testcase_count', first.get('total_testcase_count'))
    if reported_count is not None:
        try:
            reported = int(reported_count)
            if reported != float(reported_count) or reported < 0:
                raise ValueError('not a nonnegative integer')
            if reported != count:
                issues.append(dict(testcase_id=None, issues=['reported_testcase_count_mismatch']))
            count = max(count, reported)
        except (TypeError, ValueError, OverflowError):
            issues.append(dict(testcase_id=None, issues=['invalid_reported_testcase_count']))
    unknown_count = count - len(public) - hidden_count
    covered = bool(count) and not issues and unknown_count == 0
    known_public_sum = sum(r['point_score'] for r in public if r['point_score'] is not None)
    public_sum = known_public_sum if covered else None
    residual = total * count - public_sum if covered and hidden_count else None
    additive = match.get('status') == 'Pass' or total > 0
    gate_observed = total == 0 and any(r['status'] == 'Pass' for r in public)
    return {'queried_utc': stamp, 'submission_id': sid, 'available': True, **identity,
            'status': match.get('status'), 'rank': match.get('rank'),
            'package_score': total, 'testcase_count': count,
            'public': public, 'public_sum': public_sum,
            'known_public_sum': known_public_sum, 'testcases': observations,
            'visible_count': len(public), 'unknown_count': unknown_count,
            'reported_testcase_count': reported_count,
            'scoring_coverage_complete': covered,
            'coverage': dict(complete=covered, visible_time_count=sum(positive_time(r['time_us']) for r in public),
                             scored_visible_count=sum(r['point_score'] is not None for r in public),
                             issues=issues),
            'public_formula_package_contribution': public_sum / count if covered else None,
            'public_package_contribution': (public_sum / count if additive else 0.0) if covered else None,
            'scoring_gate': 'all-pass-required-observed' if gate_observed else
                            ('additive-score-visible' if additive else 'not-identifiable-from-zero-score'),
            'hidden_count': hidden_count, 'hidden_sum_inferred': residual if additive else None,
            'hidden_mean_inferred': residual / hidden_count if additive and residual is not None else None,
            'sum_rounding_uncertainty': count * 0.005,
            'note': 'Single-point interpretation requires a verified isolation probe. '
                    'TBest and total are from this ranking snapshot; they can change.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem-id', default='6ae3fad11dac5d235b148620')
    ap.add_argument('--sid', required=True)
    ap.add_argument('--account', default='A')
    ap.add_argument('--out')
    ap.add_argument('--raw-out', help='Save all ranking pages fetched for this same score snapshot')
    args = ap.parse_args()
    report = capture(args.problem_id, args.sid, cann_client.account(args.account)['cookie'],
                     raw_output=args.raw_out)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
