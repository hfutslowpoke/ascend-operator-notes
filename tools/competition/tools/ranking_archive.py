"""Timestamp-addressed observations written only by the independent board monitor."""
import hashlib
import json
import math
import os
from pathlib import Path

from scoring_snapshot import from_pages, parse_server_time
from cann_submit import ACTIVE_STATUS

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'test_harness/results/platform-watch'


def submission_key(server_time):
    parsed = parse_server_time(server_time)
    if parsed is None:
        raise ValueError('A timezone-aware server submission time is required')
    canonical = parsed.isoformat(timespec='microseconds')
    return hashlib.sha256(canonical.encode()).hexdigest(), canonical


def record(output, pages, row, observed_utc):
    """Keep every own-team poll, plus first usable score and latest by timestamp.

    Timestamp alone is the root key. SID guards collisions; it is never a fallback
    that could attach a different submission's score to that root.
    """
    sid = row.get('submission_id') or row.get('submission', {}).get('_id')
    server_time = row.get('create_time') or row.get('submission', {}).get('create_time')
    key, canonical = submission_key(server_time)
    if not sid:
        raise ValueError('Ranking observation has no submission ID')
    directory = Path(output) / 'by-submission-time' / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'record.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and (previous['submission_id'] != sid or previous['canonical_submission_time'] != canonical):
        raise ValueError('Server submission timestamp collision; refusing to overwrite another SID')
    try:
        score = from_pages(pages, sid, expected_create_time=server_time, stamp=observed_utc)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        score = dict(available=False, identity_verified=False, submission_id=sid,
                     reason='Ranking row has no complete numeric score: ' + str(error))
    # Preserve a terminal numeric package score even during a testcase rollout.
    # Missing testcase metadata limits decomposition, not this SID's raw score.
    usable = (score.get('available') and score.get('identity_verified') is True
              and row.get('status') not in ACTIVE_STATUS | {None, 'Unjudged', 'In Queue'}
              and math.isfinite(score['package_score']))
    observation = dict(observed_utc=observed_utc, server_submission_time=server_time,
                       canonical_submission_time=canonical, submission_time_key=key,
                       submission_id=sid, score=score, usable=bool(usable),
                       ranking_row=row, testcases=pages[0].get('testcases', []))
    with (directory / 'observations.jsonl').open('a') as stream:
        stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
    state = dict(previous, submission_time_key=key, canonical_submission_time=canonical,
                 server_submission_time=server_time, submission_id=sid,
                 first_observed_utc=previous.get('first_observed_utc', observed_utc),
                 last_observed_utc=observed_utc,
                 observation_count=previous.get('observation_count', 0) + 1,
                 latest_observation=observation)
    if usable and not previous.get('first_usable_observation'):
        raw = directory / 'first-score-ranking-snapshot.json'
        temporary = raw.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(queried_utc=observed_utc, submission_id=sid,
                                             pages=pages), ensure_ascii=False, indent=2) + '\n')
        temporary.replace(raw)
        observation['score']['raw_ranking_snapshot'] = str(raw)
        state['first_usable_observation'] = observation
    temporary = path.with_name(f'.record.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    return path


def lookup(output, sid, server_time):
    """Only use this timestamp's first independently archived terminal score."""
    key, canonical = submission_key(server_time)
    path = Path(output) / 'by-submission-time' / key / 'record.json'
    if not path.exists():
        return dict(available=False, identity_verified=False, submission_id=sid,
                    expected_submission_time=server_time, submission_time_key=key,
                    reason='Independent ranking monitor has no record for this submission time')
    record = json.loads(path.read_text())
    if record.get('canonical_submission_time') != canonical or record.get('submission_id') != sid:
        return dict(available=False, identity_verified=False, timestamp_mismatch=True,
                    submission_id=sid, reason='Archive timestamp/SID identity mismatch')
    observation = record.get('first_usable_observation')
    if not observation:
        return dict(available=False, identity_verified=False, submission_id=sid,
                    reason='Monitor has not yet archived a terminal numeric score')
    score = dict(observation['score'], archive_record=str(path), submission_time_key=key,
                 score_source='independent-ranking-monitor:first-usable-observation',
                 monitor_observed_utc=observation['observed_utc'],
                 expected_submission_time=server_time)
    if (score.get('submission_id') != sid or score.get('identity_verified') is not True
            or parse_server_time(score.get('ranking_submission_time')) != parse_server_time(server_time)):
        raise ValueError('Archived score failed timestamp/SID verification')
    return score
