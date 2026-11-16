#!/usr/bin/env python3
"""Download publicly submitted final sources at immutable PR head commits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import requests
import os
import subprocess
import time

ROOT = Path(os.environ.get('CANN_ARCHIVE_DOWNLOAD_ROOT', 'public-source-download')).resolve()
PR_SPEC = Path(__file__).with_name('public-pr-heads.json')
DISCOVERY = ROOT / 'discovery'
ARCHIVE = ROOT / 'archive'
PREFIX = '03_university/CANN-ops-challenge-2026-shanghe/final/submissions/'
API = 'https://gitcode.com/api/v5/repos/cann/cann-ops-competitions'


def git_blob(pr, path):
    """Use normal anonymous Git when the web raw service says no preview."""
    folder = ROOT / 'upstream/pr-forks' / f"pr-{pr['number']}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key)
    env['GIT_TERMINAL_PROMPT'] = '0'
    git = ['git', '-c', 'http.proxy=', '-c', 'credential.helper=', '-c', 'http.version=HTTP/1.1']
    if not (folder / '.git/HEAD').exists():
        subprocess.run(git + ['clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                              '--branch', pr['head']['ref'],
                              f"https://gitcode.com/{pr['head']['repo']['full_name']}.git", str(folder)],
                       env=env, check=True, capture_output=True, timeout=120)
    commit = pr['head']['sha']
    known = subprocess.run(git + ['cat-file', '-e', commit+'^{commit}'], cwd=folder, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if known.returncode:
        subprocess.run(git + ['fetch', '--depth', '1', 'origin', commit], cwd=folder, env=env,
                       check=True, capture_output=True, timeout=120)
    return subprocess.check_output(git + ['show', commit+':'+path], cwd=folder, env=env, timeout=120)


def get(url):
    session = requests.Session()
    session.trust_env = False
    for attempt in range(3):
        try:
            result = session.get(url, timeout=35)
            result.raise_for_status()
            return result
        except requests.RequestException:
            if 'result' in locals() and result.status_code in (401, 403, 404):
                raise
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def acquire(pr):
    number = pr['number']
    head = pr['head']['sha']
    files = []
    for page in range(1, 20):
        cache = DISCOVERY / f'pr-{number}-files-{page}.json'
        if cache.exists():
            rows = json.loads(cache.read_text())
        else:
            rows = get(f'{API}/pulls/{number}/files?per_page=100&page={page}').json()
            cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
        if not isinstance(rows, list):
            raise RuntimeError(f'PR{number}: files response is not a list')
        files.extend(rows)
        if len(rows) < 100:
            break
    record = dict(number=number, title=pr['title'], state_at_download=pr['state'],
                  pr_url=f'https://gitcode.com/cann/cann-ops-competitions/pull/{number}',
                  head_commit=head, source_repository=pr['head']['repo']['full_name'],
                  branch=pr['head']['ref'], created_at=pr['created_at'],
                  downloaded_utc=datetime.now(timezone.utc).isoformat(), files=[], skipped_paths=[])
    output = ARCHIVE / 'third_party/shanghe_final' / f'pr-{number}'
    use_git = (ROOT / 'upstream/pr-forks' / f'pr-{number}' / '.git/HEAD').exists()
    for entry in files:
        path = entry['filename']
        if not path.startswith(PREFIX) or entry.get('status') == 'removed':
            record['skipped_paths'].append(path)
            continue
        relative = PurePosixPath(path[len(PREFIX):])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError(f'Invalid source path in PR{number}')
        # The upstream object's immutable head is available through public raw.
        url = f"https://raw.gitcode.com/{pr['head']['repo']['full_name']}/raw/{head}/{path}"
        target = output / str(relative)
        if target.exists():
            data = target.read_bytes()
        else:
            if use_git:
                data = git_blob(pr, path)
            else:
                try:
                    response = get(url)
                    data = response.content
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        raise RuntimeError(f'PR{number}: HTML received for {path}')
                except requests.HTTPError as error:
                    if error.response.status_code != 403 or '暂不支持预览' not in error.response.text:
                        raise
                    data = git_blob(pr, path)
                    use_git = True
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if entry.get('blob_id'):
            assert blob == entry['blob_id'], (number, path, 'Git blob identity differs')
        record['files'].append(dict(original_path=path, local_path=str(target.relative_to(ARCHIVE)),
                                    url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                    git_blob_sha1=blob, api_blob_identity_verified=True))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'PROVENANCE.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


def main():
    DISCOVERY.mkdir(parents=True, exist_ok=True)
    prs = json.loads(PR_SPEC.read_text())
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(acquire, pr): pr['number'] for pr in prs}
        for job in as_completed(jobs):
            number = jobs[job]
            try:
                record = job.result()
                records.append(record)
                print(json.dumps(dict(PR=number, files=len(record['files']), head=record['head_commit']), ensure_ascii=False), flush=True)
            except Exception as error:
                errors.append(dict(PR=number, type=type(error).__name__, message=str(error)))
                print(json.dumps(errors[-1], ensure_ascii=False), flush=True)
    target = ARCHIVE / 'catalog'
    target.mkdir(parents=True, exist_ok=True)
    document = dict(downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    upstream_repository='https://gitcode.com/cann/cann-ops-competitions',
                    upstream_main_commit='776f2e552a125d3c210b497385f8b57292ebdc92',
                    scope='Public PRs matching GroupedMatmulGelu and 上合; PRs not yet merged are retained as submissions, not certified winners.',
                    matching_PR_count=len(prs), downloaded_PR_count=len(records),
                    file_count=sum(len(r['files']) for r in records), records=sorted(records, key=lambda r:r['number']),
                    errors=errors)
    (target / 'public-final-sources.json').write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in document.items() if k not in ['records']}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
