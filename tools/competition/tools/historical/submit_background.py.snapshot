#!/usr/bin/env python3
"""LEGACY competition-specific background submitter.

This is the 2026 GroupedMatmulGelu account-rotation workflow, not the general
submission path. It adds account cooldowns and a job watcher on top of the
platform rules; use `cann_submit.py` for the normal one-shot submitter.

Example:
  python3 tools/submit_background.py --submit \
    --source-dir /path/to/package --tag gmm-rNN --job-dir /path/to/job

Use --sid SID without --submit to watch an already accepted submission.
Accounts default to A/B/C round robin; each has 50 posts per Beijing day.
The user removed the per-source attempt cap on 2026-11-13; each invocation
still submits only once and records its accepted SID before polling.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
from datetime import datetime, timezone, timedelta

import cann_client
import dispatch_policy
import scoring_snapshot
import ranking_archive
from cann_submit import ACTIVE_STATUS, CST, count_today

ACCOUNTS = ('A', 'B', 'C')
# Opt-in pause marker only; no hidden repository/probes path blocks submissions.
GLOBAL_PAUSE_FILE = (Path(os.environ['CANNJUDGE_SUBMIT_PAUSE_FILE'])
                     if os.environ.get('CANNJUDGE_SUBMIT_PAUSE_FILE') else None)


POLICY = 'cooldown-v10: user 2026-11-14 no global pacing; platform account cooldown; 60s/20s detail'
LOCK_SECONDS = 0
ACCOUNT_COOLDOWN_SECONDS = 180
FIRST_POLL_SECONDS = 60
DETAIL_POLL_SECONDS = 20
RANKING_INTERVAL = 10
RANKING_WAIT_SECONDS = 60
PUBLIC_TEST_POLICY = (Path(__file__).resolve().parents[2] /
                      'coordination/PUBLIC-TEST-COOLDOWN-OVERRIDE.json')


def public_test_override(args):
    """Legacy metadata field: all cases now bypass global pacing."""
    return True


class AccountQueryUnavailable(RuntimeError):
    """Only read-only account checks failed; no POST has been attempted."""

    def __init__(self, message, retry_after_seconds=0):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def update_submission_lock(path, **fields):
    """The release timer and result observer update separate fields safely."""
    with path.with_suffix('.metadata.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        record = json.loads(path.read_text())
        record.update(fields)
        save(path, record)


def record_frontier(ledger, frontier=None, completed_job=None, job_path=None):
    """An old result must never overwrite a newer acceptance frontier."""
    path = ledger / 'background-active.json'
    with (ledger / 'background-active.metadata.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if frontier is not None:
            current = json.loads(path.read_text()) if path.exists() else {}
            if float(current.get('accepted_epoch', 0)) > float(frontier['accepted_epoch']):
                return
            save(path, frontier)
        elif path.exists():
            current = json.loads(path.read_text())
            if current.get('submission_id') == completed_job['submission_id']:
                current.update(settled=result_is_complete(completed_job), result_observation_finished=True,
                               result_job_file=str(job_path), observed_terminal_utc=now())
                save(path, current)


def sleep_until(epoch):
    while True:
        remaining = epoch - time.time()
        if remaining <= 0:
            return
        time.sleep(remaining)


def wait_while_paused(ledger, job_path, job, interval):
    """Only an explicit user resume clears the durable shared pause marker."""
    while True:
        pause = None
        paths = {ledger / 'submission-pause.json'}
        if GLOBAL_PAUSE_FILE is not None:
            paths.add(GLOBAL_PAUSE_FILE)
        for path in paths:
            if path.exists():
                record = json.loads(path.read_text())
                if record.get('paused', True):
                    pause = dict(record, marker=str(path))
                    break
        if pause is None:
            return
        job.update(status='submission_paused', submission_pause=pause)
        save(job_path, job)
        time.sleep(max(5, interval))


def result_event(ledger, job_path, job):
    ledger.mkdir(parents=True, exist_ok=True)
    score = job.get('ranking_score', {})
    event = {'recorded_utc': now(), 'job_file': str(job_path),
             'submission_id': job['submission_id'], 'source_sha256': job.get('source_sha256'),
             'server_submission_time': job.get('server_submission_time'),
             'platform_status': job['last_result']['status'],
             'ranking_measurement_valid': job['ranking_measurement_valid'],
             'point_measurement_valid': job.get('point_measurement_valid', False),
             'point_measurement_file': job.get('point_measurement_file'),
             'package_score': score.get('package_score') if job['ranking_measurement_valid'] else None,
             'reason': job.get('ranking_invalid_reason'),
             'public': job['last_result'].get('public', [])}
    with (ledger / 'result-events.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')


def receipt_records(ledger):
    for path in ledger.rglob('*.accepted.json'):
        try:
            record = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        yield path, record


def posts_for_sha(ledger, source_sha):
    count = 0
    for path, record in receipt_records(ledger):
        if record.get('status') in ('rate_limited_no_post', 'paused_no_post'):
            continue
        if any(f.get('sha256') == source_sha for f in record.get('files', [])):
            count += 1
    return count


def last_submitted_account(ledger):
    """Bootstrap rotation from accepted receipts, including older direct submits."""
    latest = None
    for path, record in receipt_records(ledger):
        if not record.get('submission_id') or record.get('account') not in ACCOUNTS:
            continue
        stamp = record.get('accepted_utc') or record.get('created_utc')
        if not stamp:
            continue
        epoch = datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
        if latest is None or epoch > latest[0]:
            latest = (epoch, record['account'])
    state_path = ledger / 'background-rotation.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        epoch = state.get('accepted_epoch', 0)
        if latest is None or epoch > latest[0]:
            return state.get('last_account')
    return latest[1] if latest else None


class AccountSlotBusy(RuntimeError):
    def __init__(self, next_available_epoch, accounts):
        super().__init__('Account slots are cooling down, busy, or at daily quota')
        self.next_available_epoch = next_available_epoch
        self.accounts = accounts


def account_deadline(ledger, label):
    """Only this account's durable activity contributes to its cooldown."""
    deadline = 0.0
    state = ledger / f'background-account-{label}.json'
    if state.exists():
        deadline = float(json.loads(state.read_text()).get('next_post_not_before_epoch', 0))
    for path in ledger.rglob('*.accepted.json'):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A legacy submitter may still be writing its receipt.
        if record.get('account') != label or not record.get('submission_id'):
            continue
        stamp = record.get('accepted_utc') or record.get('created_utc')
        if stamp:
            deadline = max(deadline, datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
                           + ACCOUNT_COOLDOWN_SECONDS)
    return deadline


def next_quota_reset():
    local = datetime.now(CST)
    return (local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()


def choose_account(ledger, problem_id, requested='auto'):
    """Return an acquired account lock, never a lock shared by A/B/C.

    Nonblocking acquisition skips a concurrent POST. No sleeps, result waits,
    quota GET, or POST run inside a cross-account metadata mutex.
    """
    last = last_submitted_account(ledger)
    start = (ACCOUNTS.index(last) + 1) % len(ACCOUNTS) if last in ACCOUNTS else 0
    labels = [ACCOUNTS[(start + i) % len(ACCOUNTS)] for i in range(len(ACCOUNTS))]
    if requested != 'auto':
        labels = [requested]
    unavailable = []
    deadlines = []
    transient_errors = False
    retry_after = 0
    for label in labels:
        guard = (ledger / f'background-account-{label}.lock').open('a')
        acquired = False
        keep = False
        try:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                deadline = max(time.time() + 1, account_deadline(ledger, label))
                unavailable.append(f'{label}: POST/preflight in progress')
                deadlines.append(deadline)
                continue
            deadline = account_deadline(ledger, label)
            if deadline > time.time():
                unavailable.append(f'{label}: account cooldown until {deadline}')
                deadlines.append(deadline)
                continue
            account = cann_client.account(label)
            history = cann_client.unwrap(cann_client.request(
                'GET', f"/submissions/user/{account['uid']}/problem/{problem_id}", account['cookie']))
            used = count_today(history)
            if used >= 50:
                unavailable.append(f'{label}: {used}/50 today')
                deadlines.append(next_quota_reset())
                continue
            # Server history also covers old/direct workers that do not use our slots.
            for row in history if isinstance(history, list) else []:
                stamp = row.get('create_time')
                if stamp:
                    deadline = max(deadline, datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
                                   + ACCOUNT_COOLDOWN_SECONDS)
            if deadline > time.time():
                unavailable.append(f'{label}: server account cooldown until {deadline}')
                deadlines.append(deadline)
                continue
            keep = True
            return label, used, guard
        except Exception as error:
            unavailable.append(f'{label}: {error}')
            if isinstance(error, cann_client.CannHttpError):
                if error.status in (408, 425, 429) or error.status >= 500:
                    transient_errors = True
                    retry_after = max(retry_after,
                                      float(error.metadata.get('retry_after_seconds', 0) or 0),
                                      30 if error.status == 429 else 0)
            elif isinstance(error, (urllib.error.URLError, TimeoutError,
                                    ConnectionError, json.JSONDecodeError)):
                transient_errors = True
        finally:
            if not keep:
                if acquired:
                    fcntl.flock(guard, fcntl.LOCK_UN)
                guard.close()
    if transient_errors:
        raise AccountQueryUnavailable('Account quota GET temporarily unavailable: ' +
                                      '; '.join(unavailable), retry_after)
    if deadlines:
        raise AccountSlotBusy(min(deadlines), unavailable)
    raise RuntimeError('No account available after credential checks: ' + '; '.join(unavailable))


def record_rotation(ledger, selected_account, accepted_epoch, sid):
    with (ledger / 'background-rotation.metadata.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = ledger / 'background-rotation.json'
        current = json.loads(path.read_text()) if path.exists() else {}
        if float(current.get('accepted_epoch', 0)) > accepted_epoch:
            return
        save(path, dict(last_account=selected_account,
                       next_account=ACCOUNTS[(ACCOUNTS.index(selected_account) + 1) % len(ACCOUNTS)],
                       accepted_epoch=accepted_epoch, submission_id=sid,
                       quota_reset='00:00 Asia/Shanghai', daily_limit_per_account=50))


def account_for_sid(ledger, sid):
    for path, record in receipt_records(ledger):
        if record.get('submission_id') == sid:
            return record['account']
    return 'A'


def accepted_job_paths(ledger):
    root = Path(__file__).resolve().parents[2]
    paths = set()
    for workspace in (root, root.with_name(root.name + '-regular-worker'),
                      root.with_name(root.name + '-oddstride-worker')):
        paths.update((workspace / 'test_harness/results/jobs').glob('*/job.json'))
    active = ledger / 'background-active.json'
    if active.exists():
        record = json.loads(active.read_text())
        for key in ('job_file', 'result_job_file'):
            if record.get(key):
                paths.add(Path(record[key]))
    return paths


def latest_accepted_submission(ledger, problem_id):
    """Receipts close the crash window between POST acceptance and job publication."""
    active_path = ledger / 'background-active.json'
    latest = json.loads(active_path.read_text()) if active_path.exists() else None
    if latest and latest.get('problem_id', problem_id) != problem_id:
        latest = None
    for path, receipt in receipt_records(ledger):
        if not receipt.get('submission_id') or receipt.get('problem_id') != problem_id:
            continue
        stamp = receipt.get('accepted_utc') or receipt.get('created_utc')
        if not stamp:
            raise RuntimeError('Accepted receipt has no timestamp; inspect ledger before another POST')
        epoch = datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
        prior_epoch = (latest or {}).get('accepted_epoch')
        if prior_epoch is None and latest and latest.get('accepted_utc'):
            prior_epoch = datetime.fromisoformat(latest['accepted_utc'].replace('Z', '+00:00')).timestamp()
        if latest is None or epoch > float(prior_epoch or 0):
            if latest is None or latest.get('submission_id') != receipt['submission_id']:
                latest = {}
            latest.update(submission_id=receipt['submission_id'], account=receipt['account'],
                          accepted_epoch=epoch, accepted_utc=stamp, problem_id=problem_id,
                          receipt=str(path))
    if latest and not latest.get('account'):
        latest['account'] = account_for_sid(ledger, latest['submission_id'])
    if latest and latest.get('accepted_epoch') is None:
        latest['accepted_epoch'] = datetime.fromisoformat(
            latest['accepted_utc'].replace('Z', '+00:00')).timestamp()
    return latest


def result_is_complete(job):
    status = job.get('last_result', {}).get('status')
    return bool(job.get('status') == 'finished' and status and status not in ACTIVE_STATUS
                and (status != 'Pass' or job.get('ranking_measurement_valid') is True))


def work(args):
    # Enforce the canonical schedule even for imported/programmatic workers.
    args.initial_delay = FIRST_POLL_SECONDS
    args.interval = DETAIL_POLL_SECONDS
    job_dir = Path(args.job_dir).resolve()
    job_path = job_dir / 'job.json'
    ledger = Path(args.ledger_dir).resolve()
    ledger.mkdir(parents=True, exist_ok=True)
    # Same-job dedupe is independent of account slots. Other jobs/accounts pass.
    with (job_dir / 'post-attempt.lock').open('a') as job_guard:
        if args.submit:
            try:
                fcntl.flock(job_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('This job already has a submitting worker; no duplicate POST')
        job = json.loads(job_path.read_text())
        if args.submit and (job.get('submission_id') or job.get('post_attempt_started_utc')):
            raise RuntimeError('Job already accepted or POST attempted; inspect receipt/history and use --sid, never automatically POST again')
        job.update(pid=os.getpid(), started_utc=now(), status='starting',
                   initial_delay_seconds=args.initial_delay, poll_interval_seconds=args.interval,
                   ranking_archive_dir=args.ranking_archive_dir,
                   ranking_match_interval_seconds=RANKING_INTERVAL,
                   ranking_match_timeout_seconds=RANKING_WAIT_SECONDS,
                   submission_policy=POLICY, submission_lock_seconds=0,
                   global_cooldown_disabled=True, account_cooldown_seconds=ACCOUNT_COOLDOWN_SECONDS,
                   public_test_points=getattr(args, 'public_test', None) or [],
                   public_test_cooldown_bypassed=True)
        save(job_path, job)
        sid = args.sid
        selected_account = args.account
        accepted_epoch = None
        if args.submit:
            source = Path(args.source_dir).resolve()
            actual_sha = hashlib.sha256((source / 'kernel.asc').read_bytes()).hexdigest()
            if actual_sha != job['source_sha256']:
                raise RuntimeError('Source changed after background job was created')
            dispatch_policy.validate_package(source, args.diagnostic_probe)
            while True:
                while True:
                    wait_while_paused(ledger, job_path, job, args.interval)
                    try:
                        selected_account, used_today, account_guard = choose_account(ledger, args.problem_id, args.account)
                        break
                    except AccountSlotBusy as error:
                        job.update(status='waiting_for_account_slot', account_slot_status=error.accounts,
                                   next_account_available_epoch=error.next_available_epoch)
                        save(job_path, job)
                        sleep_until(error.next_available_epoch)
                    except AccountQueryUnavailable as error:
                        attempts = job.get('pre_post_query_errors', 0) + 1
                        retry_delay = max(error.retry_after_seconds,
                                          min(300, max(5, args.interval) * 2 ** min(attempts - 1, 6)))
                        job.update(status='waiting_for_account_query', pre_post_query_errors=attempts,
                                   last_account_query_error=str(error),
                                   next_account_query_retry_epoch=time.time() + retry_delay)
                        save(job_path, job)
                        print(f'Pre-POST quota GET failed; retrying without POST: {error}', flush=True)
                        time.sleep(retry_delay)
                try:
                    job.pop('next_account_query_retry_epoch', None)
                    job.update(account=selected_account, posts_today_before=used_today,
                               quota_date_beijing=datetime.now(CST).date().isoformat())
                    save(job_path, job)
                    if hashlib.sha256((source / 'kernel.asc').read_bytes()).hexdigest() != actual_sha:
                        raise RuntimeError('Source changed while waiting for an account slot')
                    used = posts_for_sha(ledger, actual_sha)
                    # Unique job path avoids same-SHA attempt-number collisions across accounts.
                    job_key = hashlib.sha256(str(job_path).encode()).hexdigest()[:24]
                    attempt_ledger = (ledger / 'background' / actual_sha / ('job-' + job_key) /
                                      ('attempt-' + str(int(job.get('post_attempt_count', 0)) + 1)))
                    command = [sys.executable, str(Path(__file__).with_name('cann_submit.py')),
                               '--problem-id', args.problem_id, '--account', selected_account,
                               '--source-dir', str(source), '--tag', args.tag,
                               '--ledger-dir', str(attempt_ledger), '--max-per-day', '50']
                    wait_while_paused(ledger, job_path, job, args.interval)
                    proc = subprocess.run(command, text=True, capture_output=True)
                    print(proc.stdout, end='', flush=True)
                    print(proc.stderr, end='', file=sys.stderr, flush=True)
                    if proc.returncode:
                        raise RuntimeError('Preflight failed before POST; inspect the command log')
                    wait_while_paused(ledger, job_path, job, args.interval)
                    if hashlib.sha256((source / 'kernel.asc').read_bytes()).hexdigest() != actual_sha:
                        raise RuntimeError('Source changed during submission preflight')
                    job.update(status='posting_once', post_attempt_started_utc=now(), attempt_ledger=str(attempt_ledger),
                               post_attempt_count=int(job.get('post_attempt_count', 0)) + 1)
                    save(job_path, job)
                    save(ledger / f'background-account-{selected_account}.json',
                         dict(account=selected_account, state='post_inflight', job_file=str(job_path),
                              post_started_epoch=time.time(),
                              next_post_not_before_epoch=time.time() + ACCOUNT_COOLDOWN_SECONDS))
                    # Exactly one POST-capable child invocation. Errors are never retried.
                    proc = subprocess.run(command + ['--submit', '--no-poll'], text=True, capture_output=True)
                    print(proc.stdout, end='', flush=True)
                    print(proc.stderr, end='', file=sys.stderr, flush=True)
                    if proc.returncode:
                        rejected = list(attempt_ledger.rglob('*.accepted.json'))
                        rejection = json.loads(rejected[0].read_text()) if len(rejected) == 1 else {}
                        if rejection.get('status') == 'rate_limited_no_post' and not rejection.get('submission_id'):
                            retry_seconds = max(1.0, float(rejection.get('retry_after_seconds') or ACCOUNT_COOLDOWN_SECONDS))
                            deadline = time.time() + retry_seconds
                            save(ledger / f'background-account-{selected_account}.json',
                                 dict(account=selected_account, state='rate_limited_no_post', job_file=str(job_path),
                                      rejected_utc=now(), retry_after_seconds=retry_seconds,
                                      next_post_not_before_epoch=deadline, receipt=str(rejected[0])))
                            job.pop('post_attempt_started_utc', None)
                            job.setdefault('definitive_post_rejections', []).append(dict(
                                account=selected_account, receipt=str(rejected[0]), status='rate_limited_no_post',
                                retry_after_seconds=retry_seconds, next_account_post_epoch=deadline))
                            job.update(status='retrying_after_definitive_429')
                            save(job_path, job)
                            continue  # Explicit no-acceptance only; finally releases this account first.
                        raise RuntimeError('Submission command failed; inspect receipt/history before any new POST')
                    receipts = list(attempt_ledger.rglob('*.accepted.json'))
                    if len(receipts) != 1:
                        raise RuntimeError('Expected one acceptance record; inspect ledger before any new POST')
                    receipt = json.loads(receipts[0].read_text())
                    sid = receipt['submission_id']
                    accepted_epoch = datetime.fromisoformat(receipt['accepted_utc'].replace('Z', '+00:00')).timestamp()
                    save(ledger / f'background-account-{selected_account}.json',
                         dict(account=selected_account, state='accepted', job_file=str(job_path), submission_id=sid,
                              accepted_epoch=accepted_epoch,
                              next_post_not_before_epoch=accepted_epoch + ACCOUNT_COOLDOWN_SECONDS))
                    record_rotation(ledger, selected_account, accepted_epoch, sid)
                    job.update(submission_id=sid, package_posts_used=used + 1, package_posts_authorized=None,
                               package_post_policy='user-authorized-no-per-source-cap-20261113',
                               accepted_utc=receipt['accepted_utc'], accepted_epoch=accepted_epoch,
                               submission_lock_seconds=0, submission_policy=POLICY)
                    save(job_path, job)
                    record_frontier(ledger, frontier=dict(job_file=str(job_path), submission_id=sid,
                        problem_id=args.problem_id, account=selected_account, source_sha256=actual_sha,
                        accepted_utc=job['accepted_utc'], accepted_epoch=accepted_epoch, settled=False,
                        submission_policy='cooldown-v10', release_not_before_epoch=accepted_epoch))
                    save(job_dir / 'submission-lock.json', dict(state='released', submission_id=sid,
                         result_complete=False, released_utc=now(), global_lock_used=False,
                         account=selected_account, account_next_post_not_before_epoch=accepted_epoch + ACCOUNT_COOLDOWN_SECONDS))
                finally:
                    fcntl.flock(account_guard, fcntl.LOCK_UN)
                    account_guard.close()
                break
        elif selected_account == 'auto':
            selected_account = account_for_sid(ledger, sid)
            job.update(account=selected_account)
        if accepted_epoch is None:
            previous = latest_accepted_submission(ledger, args.problem_id)
            accepted_epoch = (job.get('accepted_epoch') or
                              (previous.get('accepted_epoch') if previous and previous['submission_id'] == sid else None))
        # Only an accepted SID observer remains; no account/global POST lock is held.
        return poll_result(args, job, sid, selected_account, accepted_epoch)


def wait_for_archived_score(args, job_path, job, sid, detail):
    deadline = time.time() + RANKING_WAIT_SECONDS
    polls = 0
    score = {}
    while True:
        polls += 1
        try:
            score = ranking_archive.lookup(Path(args.ranking_archive_dir), sid, detail.get('create_time'))
            if score.get('available') and score.get('identity_verified') is True:
                # Match terminal status too, so a transient pre-result board row
                # cannot be accepted for a different final evaluation outcome.
                if score.get('status') == detail.get('status'):
                    raw = score.get('raw_ranking_snapshot')
                    if raw:
                        save(job_path.parent / 'score-ranking-snapshot.json', json.loads(Path(raw).read_text()))
                    return score
                score = dict(available=False, identity_verified=False, submission_id=sid,
                             reason='Archived ranking status differs from terminal detail')
        except Exception as error:
            score = dict(available=False, identity_verified=False, submission_id=sid,
                         reason='Independent ranking archive unavailable: ' + str(error))
        job.update(status='waiting_for_ranking_snapshot', ranking_wait_polls=polls,
                   ranking_match_deadline_epoch=deadline, ranking_score=score)
        save(job_path, job)
        remaining = deadline - time.time()
        if remaining <= 0:
            score.update(archive_match_timed_out=True, ranking_wait_seconds=RANKING_WAIT_SECONDS)
            return score
        time.sleep(min(RANKING_INTERVAL, remaining))


def poll_result(args, job, sid, selected_account, accepted_epoch=None):
    job_dir = Path(args.job_dir).resolve()
    job_path = job_dir / 'job.json'
    ledger = Path(args.ledger_dir).resolve()
    job.update(submission_id=sid, status='waiting_for_first_poll',
               first_poll_not_before_epoch=(accepted_epoch if accepted_epoch is not None else time.time()) + args.initial_delay)
    save(job_path, job)
    sleep_until(job['first_poll_not_before_epoch'])
    account = cann_client.account(selected_account)
    consecutive_get_errors = 0
    while True:
        retry_delay = args.interval
        try:
            detail = cann_client.request(
                'GET', f"/submissions/{sid}?userId={account['uid']}", account['cookie'])
            save(job_dir / 'detail.json', detail)
            if detail.get('_id') and detail['_id'] != sid:
                raise RuntimeError('Detail response SID differs from the requested submission')
            consecutive_get_errors = 0
            status = detail.get('status')
            job['server_submission_time'] = detail.get('create_time')
            summary = {'queried_utc': now(), 'submission_id': sid,
                       'status': status, 'valid': detail.get('valid'),
                       'public': [{key: row.get(key) for key in
                                   ('testcase_id', 'testcase_status', 'time', 'precision_ratio', 'best_time')}
                                  for row in detail.get('result', [])]}
            print(json.dumps(summary), flush=True)
            job.update(last_result=summary, status='polling')
            # All cases are now visible in the immutable SID detail. Their
            # timings remain usable even if a newer POST has replaced the row
            # before the independent ranking monitor sampled it.
            points = detail.get('result', [])
            point_valid = bool(detail.get('_id') == sid and status == 'Pass' and points and
                               all(row.get('testcase_id') and row.get('testcase_status') == 'Pass' and
                                   isinstance(row.get('time'), (int, float)) and row['time'] > 0
                                   for row in points))
            if points:
                point_file = job_dir / 'point-times.json'
                save(point_file, {'queried_utc': summary['queried_utc'], 'submission_id': sid,
                                  'source_sha256': job.get('source_sha256'),
                                  'server_submission_time': detail.get('create_time'),
                                  'status': status, 'point_measurement_valid': point_valid,
                                  'case_count': len(points), 'result': points})
                job.update(point_measurement_valid=point_valid, point_measurement_file=str(point_file))
            if status and status not in ACTIVE_STATUS:
                score = wait_for_archived_score(args, job_path, job, sid, detail)
                if args.diagnostic_probe and score.get('available'):
                    score['diagnostic_target'] = job['probe']['target_shape']
                    public = score.get('public', [])
                    controls = job['probe'].get('keep_public_controls', False)
                    public_ok = bool(public) and all(r['status'] == 'Pass' for r in public)
                    public_failed = bool(public) and all(r['status'] == 'Wrong Answer' for r in public)
                    value = score['package_score'] * score['testcase_count']
                    if controls:
                        value -= score['public_sum']
                    isolated = (status == 'Wrong Answer' and (public_ok if controls else public_failed)
                                and 0 < value <= 100.075)
                    score['isolation_consistent'] = isolated
                    if isolated:
                        score['isolated_point_score'] = value
                    else:
                        score['isolation_note'] = ('No single-point score inferred: inspect public failures, '
                                                   'target correctness and partial-scoring support.')
                    save(job_dir / 'score.json', score)
                job['ranking_score'] = score
                save(job_dir / 'score.json', score)
                job['ranking_measurement_valid'] = bool(score.get('available') and score.get('identity_verified') is True)
                if not job['ranking_measurement_valid']:
                    job['ranking_invalid_reason'] = score.get('reason') or job.get('ranking_score_error') or 'Matching ranking snapshot was not obtained'
                    print(json.dumps({'ranking_measurement_valid': False, 'submission_id': sid,
                                      'reason': job['ranking_invalid_reason'],
                                      'note': 'This POST still consumes quota; no automatic resubmission.'}), flush=True)
                job.update(status='finished', finished_utc=now())
                save(job_path, job)
                lock_record = job_dir / 'submission-lock.json'
                if lock_record.exists():
                    update_submission_lock(lock_record, result_complete=result_is_complete(job),
                                           result_observed_utc=now(),
                                           ranking_measurement_valid=job['ranking_measurement_valid'])
                record_frontier(ledger, completed_job=job, job_path=job_path)
                result_event(ledger, job_path, job)
                return
        except Exception as error:
            consecutive_get_errors += 1
            retry_delay = min(300, args.interval * 2 ** min(consecutive_get_errors, 6))
            if isinstance(error, cann_client.CannHttpError):
                retry_delay = max(retry_delay, float(error.metadata.get('retry_after_seconds', 0) or 0))
            job.update(last_poll_error=str(error), last_poll_error_utc=now())
            print(f'GET failed: {error}', flush=True)
        save(job_path, job)
        time.sleep(retry_delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem-id', default='6ae3fad11dac5d235b148620')
    ap.add_argument('--account', choices=('auto', *ACCOUNTS), default='auto',
                    help='auto rotates independent A/B/C slots, skipping account cooldowns and exhausted quotas')
    ap.add_argument('--source-dir')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--job-dir', required=True)
    ap.add_argument('--ledger-dir',
                    default=str(Path.home() / '.local/state/cannjudge/submit-ledger'),
                    help='global ledger shared by all sessions (legacy script; general use cann_submit.py)')
    ap.add_argument('--initial-delay', type=int, default=FIRST_POLL_SECONDS,
                    help='compatibility option; first detail GET is fixed at acceptance +60 seconds')
    ap.add_argument('--lock-seconds', type=int, default=0,
                    help='legacy compatibility option; ignored: latest user removed global pacing')
    ap.add_argument('--interval', type=int, default=DETAIL_POLL_SECONDS,
                    help='compatibility option; normal detail polling is fixed at 20 seconds')
    ap.add_argument('--ranking-retries', type=int, default=48,
                    help='legacy compatibility option; ignored: archive matching lasts at most 60 seconds')
    ap.add_argument('--ranking-archive-dir', default=str(ranking_archive.DEFAULT_OUTPUT),
                    help='independent watcher output; no live ranking requests from submission workers')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--public-test', nargs='+', choices=('C0', 'C1', 'C2', 'C3', 'C4'),
                    help='public-case timing label only; all jobs already have zero global cooldown')
    ap.add_argument('--diagnostic-probe', action='store_true',
                    help='explicit user-authorized NaN isolation probe with verified probe.json')
    ap.add_argument('--sid')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.submit == bool(args.sid) or (args.submit and not args.source_dir):
        ap.error('Choose --submit with --source-dir, or --sid to watch an existing submission')
    if args.initial_delay < 0 or args.interval < 1 or args.ranking_retries < 0 or args.lock_seconds < 0:
        ap.error('Delays must be nonnegative and interval must be positive')
    # Old controllers pass 120/5 explicitly; they must adopt the current rule too.
    args.initial_delay = FIRST_POLL_SECONDS
    args.interval = DETAIL_POLL_SECONDS
    if args.worker:
        try:
            work(args)
        except Exception as error:
            path = Path(args.job_dir).resolve() / 'job.json'
            job = json.loads(path.read_text())
            job.update(status='error', error=str(error), failed_utc=now())
            save(path, job)
            raise
        return
    if args.submit:
        probe = dispatch_policy.validate_package(Path(args.source_dir).resolve(), args.diagnostic_probe)
    directory = Path(args.job_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    job = {'created_utc': now(), 'status': 'queued', 'account': args.account,
           'requested_account': args.account,
           'tag': args.tag, 'initial_delay_seconds': args.initial_delay,
           'poll_interval_seconds': args.interval,
           'submission_lock_seconds': 0,
           'global_cooldown_disabled': True, 'account_cooldown_seconds': ACCOUNT_COOLDOWN_SECONDS,
           'public_test_points': args.public_test or [],
           'public_test_cooldown_bypassed': public_test_override(args),
           'ranking_archive_dir': args.ranking_archive_dir,
           'ranking_match_interval_seconds': RANKING_INTERVAL,
           'ranking_match_timeout_seconds': RANKING_WAIT_SECONDS,
           'legacy_lock_seconds_ignored': args.lock_seconds,
           'submission_policy': POLICY}
    if args.source_dir:
        source = Path(args.source_dir).resolve() / 'kernel.asc'
        job.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if args.submit and probe:
            job.update(purpose='diagnostic hidden-point score isolation', probe=probe,
                       restore_required=True)
    save(directory / 'job.json', job)
    with (directory / 'worker.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    *sys.argv[1:], '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'job': str(directory / 'job.json'),
                      'log': str(directory / 'worker.log')}))


if __name__ == '__main__':
    main()
