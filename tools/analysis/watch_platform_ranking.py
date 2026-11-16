#!/usr/bin/env python3
"""Read-only CannJudge monitor. No submission or submission-detail requests."""
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter_kit/tools'))
import cann_client
import ranking_archive
import scoring_snapshot


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def summary(row):
    if row is None:
        return None
    return dict(rank=row.get('rank'), team_id=row.get('team', {}).get('_id'),
                team=row.get('team', {}).get('team_name'),
                submitter=row.get('submitter', row.get('user', {})).get('nickname'),
                submission_id=row.get('submission_id', row.get('submission', {}).get('_id')),
                submission_time=row.get('create_time') or row.get('submission', {}).get('create_time'), status=row.get('status'),
                score=row.get('score'), public=[r for r in row.get('result', [])
                if r.get('testcase_status') != 'Hidden' or scoring_snapshot.positive_time(r.get('time'))])


def build_observed(pages, team_id, job=None):
    """Normalize an already fetched board; no network or archive mutations."""
    first = pages[0]
    rows = [row for page in pages for row in page.get('rows', [])]
    own = None
    payload = first
    for page in pages:
        own = next((row for row in page.get('rows', [])
                    if row.get('team', {}).get('_id') == team_id), None)
        if own is not None:
            payload = page
            break
    cases = payload.get('testcases', first.get('testcases', []))
    catalog, issues = scoring_snapshot.observe_testcases(cases, (own or {}).get('result', []))
    score = None
    if own is not None:
        try:
            sid = own.get('submission_id') or own.get('submission', {}).get('_id')
            server_time = own.get('create_time') or own.get('submission', {}).get('create_time')
            score = scoring_snapshot.from_pages(pages, sid, expected_create_time=server_time)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass  # Raw observations still belong in the monitor during transitions.
    visible = [case for case in catalog if case['visibility'] == 'public']
    hidden_count = sum(case['visibility'] == 'hidden' for case in catalog)
    count = len(catalog)
    try:
        count = max(count, int(payload.get('total_testcase_count', count)))
    except (TypeError, ValueError, OverflowError):
        pass
    return dict(our_team=summary(own), top_ten=[summary(row) for row in rows[:10]],
                public_tbest=[dict(testcase_id=case['testcase_id'], tbest_us=case['tbest_us'])
                              for case in visible],
                visible_testcases=visible, testcase_catalog=catalog,
                testcase_count=count, visible_count=len(visible), hidden_count=hidden_count,
                unknown_count=count - len(visible) - hidden_count,
                scoring_coverage_complete=bool(score and score.get('identity_verified') is True
                                               and score.get('scoring_coverage_complete')),
                scoring_coverage=(score or {}).get('coverage', dict(complete=False, issues=issues)),
                latest_local_accepted_job=job,
                local_latest_matches_board=bool(job and own and job['submission_id'] ==
                                               (own.get('submission_id') or own.get('submission', {}).get('_id'))),
                all_visible_scores=[dict(submission_id=row.get('submission_id'), score=row.get('score'),
                                         rank=row.get('rank'), team_id=row.get('team', {}).get('_id'))
                                    for row in rows])


def changed_observation_fields(previous, observed):
    # Include the schema and visibility changes, including a 15 -> 20 rollout.
    return [key for key in ('top_ten', 'public_tbest', 'testcase_count', 'visible_count',
                            'hidden_count', 'unknown_count', 'testcase_catalog',
                            'scoring_coverage_complete')
            if previous is None or observed.get(key) != previous.get(key)]


def latest_local_job(results):
    jobs = []
    for path in (results / 'jobs').glob('*/job.json'):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if job.get('submission_id'):
            jobs.append((job.get('accepted_utc', job.get('created_utc', '')), path, job))
    if not jobs:
        return None
    _, path, job = max(jobs, key=lambda item: item[0])
    return {key: job.get(key) for key in ['submission_id', 'source', 'source_sha256',
            'account', 'accepted_utc', 'status', 'last_result']} | {'job': str(path)}


def markdown(path, state):
    content = state.get('observed', {})
    own = content.get('our_team') or {}
    job = content.get('latest_local_accepted_job') or {}
    lines = ['# GroupMatmulGelu 平台监视', '',
             f"最近成功抓取（北京时间）：{state.get('last_success_beijing', '尚无')}。轮询间隔 {state['interval_seconds']} 秒；PID {state['pid']}。",
             f"运行状态：{state['status']}；成功 {state['successful_polls']} 次；连续失败 {state['consecutive_errors']} 次。", '',
             '## 我们的当前榜单代表提交', '',
             f"队伍：{own.get('team')}；排名：{own.get('rank')}；总分：{own.get('score')}；状态：{own.get('status')}。",
             f"榜单提交 ID：{own.get('submission_id')}；提交时间：{own.get('submission_time')}。",
             f"本工作区最新已接收提交 ID：{job.get('submission_id')}；后台状态：{job.get('status')}。",
             f"本地最新与榜单 ID 一致：{content.get('local_latest_matches_board')}。不一致可能是排队中或其他环境进行了提交。", '',
             '## 榜单前十名', '', '|排名|队伍|总分|状态|提交 ID|', '|---:|---|---:|---|---|']
    for row in content.get('top_ten', []):
        name = str(row.get('team') or row.get('submitter')).replace('|', '\\|')
        lines.append(f"|{row.get('rank')}|{name}|{row.get('score')}|{row.get('status')}|{row.get('submission_id')}|")
    lines += ['', '## 可见逐点观测', '',
              '|testcase ID|状态|耗时（微秒）|TBest（微秒）|评分覆盖|', '|---|---|---:|---:|---|']
    visible = content.get('visible_testcases')
    if visible is None:  # A saved pre-upgrade latest.json is still readable.
        results = {str(row.get('testcase_id')): row for row in own.get('public', [])}
        visible = [dict(row, status=results.get(str(row['testcase_id']), {}).get('testcase_status'),
                        time_us=results.get(str(row['testcase_id']), {}).get('time'))
                   for row in content.get('public_tbest', [])]
    for row in visible:
        coverage = ', '.join(row.get('coverage_issues', [])) or '完整'
        lines.append(f"|{row['testcase_id']}|{row.get('status')}|{row.get('time_us')}|{row.get('tbest_us')}|{coverage}|")
    lines += ['', f"当前记录 {content.get('testcase_count', '未知')} 点；可见条目 {len(visible)} 个，隐藏 {content.get('hidden_count')} 个，未知 {content.get('unknown_count', 0)} 个。"]
    if content.get('hidden_count') == 0 and content.get('unknown_count', 0) == 0:
        lines.append('当前没有隐藏条目，不推算隐藏点残差或隐藏均值。')
    else:
        lines.append('同一提交 ID 在可见耗时不变时总分变化，可记录为评分尺度变化证据，不能反推出具体隐藏点耗时或 TBest。')
    if content.get('scoring_coverage_complete') is not True:
        lines.append('评分覆盖不足：保留已见耗时、TBest 和状态；缺失或冲突条目不编造点分，也不推算隐藏均值。')
    lines += ['',
              '## 文件与边界', '',
              'latest.json 保存最近成功观测；status.json 保存进程心跳及错误；每次抓取均向 by-submission-time/SHA256(规范服务器提交时间)/observations.jsonl 记录本队成绩，record.json 保留首次有效成绩与最新观测。前十名或公开tbest变化另存 changes 与 snapshots。',
              '只做 GET 排名查询，不占提交配额，不改提交队列，不自动恢复旧包。榜单当前分数与历史最高分分开记录；本机最新提交记录只覆盖本工作区。',
              f"首次观测以来本队最高可见分数：{state.get('best_observed_team_score')}。该值是历史观察，不是当前有效成绩。", '']
    if state.get('last_error'):
        lines += [f"最近错误：{state['last_error']}", '']
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(path)


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'snapshots').mkdir(exist_ok=True)
    lock = (output / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    account = cann_client.account(args.account)
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    previous = None
    old = output / 'latest.json'
    if old.exists():
        previous = json.loads(old.read_text()).get('observed')
    state = dict(pid=os.getpid(), started_utc=stamp(), interval_seconds=args.interval,
                 status='starting', successful_polls=0, consecutive_errors=0,
                 team_id=args.team_id, observed=previous or {}, best_observed_team_score=None)
    if (output / 'status.json').exists():
        state['best_observed_team_score'] = json.loads((output / 'status.json').read_text()).get('best_observed_team_score')
    (output / 'monitor.pid').write_text(str(os.getpid()) + '\n')
    save(output / 'status.json', state)
    while not stop:
        started = time.monotonic()
        delay = args.interval
        try:
            first = cann_client.request('GET', f'/problems/{args.problem_id}/ranking?size=100&page=1', account['cookie'])
            pages = [first]
            rows = list(first.get('rows', []))
            for page in range(2, int(first.get('pages', 1)) + 1):
                if any(r.get('team', {}).get('_id') == args.team_id for r in rows):
                    break
                more = cann_client.request('GET', f'/problems/{args.problem_id}/ranking?size=100&page={page}', account['cookie'])
                pages.append(more)
                rows.extend(more.get('rows', []))
            own = next((r for r in rows if r.get('team', {}).get('_id') == args.team_id), None)
            job = latest_local_job(ROOT / 'test_harness/results')
            observed = build_observed(pages, args.team_id, job)
            now = stamp()
            if own is not None:
                archived = ranking_archive.record(output, pages, own, now)
                state['last_own_archive_record'] = str(archived)
                state['last_own_archive_utc'] = now
            changes = changed_observation_fields(previous, observed)
            state.update(status='running', last_success_utc=now,
                         last_success_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                         successful_polls=state['successful_polls'] + 1, consecutive_errors=0, last_error=None,
                         observed=observed)
            score = (observed['our_team'] or {}).get('score')
            if isinstance(score, (int, float)):
                state['best_observed_team_score'] = max(score, state['best_observed_team_score'] or score)
            if changes:
                identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                raw = output / 'snapshots' / (identity + '.json')
                save(raw, dict(queried_utc=now, pages=pages))
                event = dict(queried_utc=now, beijing=state['last_success_beijing'], changed=changes,
                             observed=observed, raw_snapshot=str(raw))
                with (output / 'changes.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                with (output / 'changes.md').open('a') as stream:
                    stream.write(f"\n- {event['beijing']}：变化 {', '.join(changes)}；本队排名 {(observed['our_team'] or {}).get('rank')}，分数 {score}，SID {(observed['our_team'] or {}).get('submission_id')}。原始快照 `{raw.name}`。\n")
                previous = observed
            save(output / 'latest.json', dict(queried_utc=now, observed=observed))
        except Exception as error:
            state['consecutive_errors'] += 1
            state.update(status='retrying', last_error=str(error).replace(account['cookie'], '[REDACTED]'),
                         last_error_utc=stamp())
            delay = min(300, args.interval * (2 ** min(state['consecutive_errors'], 6)))
            if isinstance(error, cann_client.CannHttpError):
                delay = max(delay, float(error.metadata.get('retry_after_seconds', 0) or 0))
        save(output / 'status.json', state)
        markdown(output / 'MONITOR.md', state)
        if args.once:
            break
        # A single thread prevents overlapping requests when the endpoint is slow.
        deadline = started + delay
        while not stop and time.monotonic() < deadline:
            time.sleep(min(0.5, deadline - time.monotonic()))
    state.update(status='stopped', stopped_utc=stamp())
    save(output / 'status.json', state)
    markdown(output / 'MONITOR.md', state)
    return 0 if state['successful_polls'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem-id', default='6ae3fad11dac5d235b148620')
    parser.add_argument('--team-id', default='6ae493451dac5d235ba11f8c')
    parser.add_argument('--account', default='A')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--output', default=str(ROOT / 'test_harness/results/platform-watch'))
    parser.add_argument('--once', action='store_true')
    options = parser.parse_args()
    if options.interval < 5:
        parser.error('interval must be at least 5 seconds')
    raise SystemExit(run(options))
