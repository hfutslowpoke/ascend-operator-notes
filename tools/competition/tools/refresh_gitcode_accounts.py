#!/usr/bin/env python3
"""Exchange private GitCode cookies for verified CannJudge account cookies.

The input and account files contain credentials. Only nonsecret status metadata
is emitted, and all three account identities are checked before installation.
"""
import argparse
import fcntl
import http.cookies
import json
import os
from pathlib import Path
import stat
import tempfile
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request

import cann_client
from submit_background import count_today


class RefreshError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
UA = 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/120 Safari/537.36'
LAST_REQUEST = {}


def request(url, *, headers=None, data=None):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in (
        'cannjudge.cn', 'web-api.gitcode.com',
    ):
        raise RefreshError('Unexpected authentication endpoint')
    req = urllib.request.Request(url, headers={'User-Agent': UA, **(headers or {})}, data=data)
    try:
        response = OPENER.open(req, timeout=45)
    except urllib.error.HTTPError as error:
        response = error
    LAST_REQUEST.clear()
    LAST_REQUEST.update(host=parsed.hostname, path=parsed.path, status=response.status,
                        content_type=response.headers.get('Content-Type', '').split(';')[0])
    return response


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def exchange(header, allowed_uids):
    if '\r' in header or '\n' in header:
        raise RefreshError('Invalid cookie header')
    essential = []
    for part in header.split(';'):
        name, separator, value = part.strip().partition('=')
        if separator and name in ('GITCODE_ACCESS_TOKEN', 'GITCODE_REFRESH_TOKEN'):
            essential.append(name + '=' + value)
    if len(essential) != 2:
        raise RefreshError('GitCode access and refresh cookies are required')
    header = '; '.join(essential)
    with request('https://cannjudge.cn/api/users/oauth/gitcode/authorize?scene=login') as response:
        if response.status not in (301, 302, 303, 307, 308):
            raise RefreshError('CannJudge authorization redirect unavailable; HTTP ' + str(response.status))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(response.headers.get('Location', '')).query)
    if any(key not in query for key in ('client_id', 'state', 'redirect_uri')):
        raise RefreshError('Incomplete authorization redirect')
    callback = urllib.parse.urlsplit(query['redirect_uri'][0])
    if callback.scheme != 'https' or callback.hostname != 'cannjudge.cn':
        raise RefreshError('Unexpected authorization callback')
    form = {key: query[key][0] for key in ('client_id', 'state', 'redirect_uri')}
    form.update(response_type='code', scopes=query.get('scopes', query.get('scope', ['all_user,all_key']))[0])
    session_cookies = dict(part.strip().split('=', 1) for part in header.split(';'))
    document = None
    for session_attempt in range(3):
        with request('https://web-api.gitcode.com/uc/api/v1/oauth/checkOrAuthorize', headers={
            'Cookie': '; '.join(name + '=' + value for name, value in session_cookies.items()),
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://gitcode.com', 'Referer': 'https://gitcode.com/',
            'Accept': 'application/json, text/plain, */*',
            'Authorization': 'Bearer ' + session_cookies['GITCODE_ACCESS_TOKEN'],
        }, data=urllib.parse.urlencode(form).encode()) as response:
            if response.status != 200:
                raise RefreshError('GitCode authorization HTTP ' + str(response.status))
            body = response.read()
            changed = False
            for line in response.headers.get_all('Set-Cookie') or []:
                returned = http.cookies.SimpleCookie()
                returned.load(line)
                for name, morsel in returned.items():
                    if session_cookies.get(name) != morsel.value:
                        session_cookies[name] = morsel.value
                        changed = True
            if not body:
                if changed and session_attempt == 0:
                    continue  # Accept a refreshed edge session once.
                break  # The browser displays the existing app's consent form.
            try:
                document = json.loads(body)
            except (ValueError, UnicodeError):
                raise RefreshError('GitCode authorization returned a non-JSON response') from None
            break
    if document is None:
        consent = dict(form, access='true')
        with request('https://web-api.gitcode.com/uc/api/v1/oauth/authorize', headers={
            'Cookie': '; '.join(name + '=' + value for name, value in session_cookies.items()),
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://gitcode.com', 'Referer': 'https://gitcode.com/',
            'Authorization': 'Bearer ' + session_cookies['GITCODE_ACCESS_TOKEN'],
        }, data=urllib.parse.urlencode(consent).encode()) as response:
            if response.status not in (301, 302, 303, 307, 308):
                raise RefreshError('Existing CannJudge app authorization HTTP ' + str(response.status))
            document = {'redirect_uri': response.headers.get('Location', '')}
    if not isinstance(document, dict):
        raise RefreshError('GitCode authorization did not return a JSON object')
    callback_url = document.get('redirect_uri')
    if not isinstance(callback_url, str):
        raise RefreshError('GitCode did not return an authorization callback')
    with request(callback_url) as response:
        if response.status not in (301, 302, 303, 307, 308):
            raise RefreshError('CannJudge callback did not return a login ticket')
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(response.headers.get('Location', '')).query)
    ticket = query.get('ticket', [None])[0]
    if not ticket:
        raise RefreshError('CannJudge login ticket missing')
    with request('https://cannjudge.cn/api/users/oauth/gitcode/login/exchange',
                 headers={'Content-Type': 'application/json'},
                 data=json.dumps({'ticket': ticket}).encode()) as response:
        if response.status != 200:
            raise RefreshError('CannJudge ticket exchange HTTP ' + str(response.status))
        token = None
        for cookie in response.headers.get_all('Set-Cookie') or []:
            first = cookie.split(';', 1)[0]
            name, separator, value = first.partition('=')
            if separator and name.strip() == 'cannjudge_auth' and value:
                token = value
        if not token:
            raise RefreshError('CannJudge authentication cookie missing')
        try:
            exchange_user = json.load(response)
        except (ValueError, UnicodeError):
            exchange_user = {}
    metadata = cann_client.token_metadata(token)
    if metadata['uid'] not in allowed_uids:
        staging = cann_client.ACCOUNT_FILE.parent / 'new-identity-review-20261106.json'
        atomic_json(staging, dict(metadata, cookie=token, usable_for_submit=False, exchange_user=exchange_user))
        safe_user = {k:exchange_user[k] for k in ('nickname', 'username') if isinstance(exchange_user, dict) and isinstance(exchange_user.get(k), str)}
        raise RefreshError('Authenticated UID is outside the existing three accounts; no credentials installed; received UID ' + metadata['uid'] + '; user metadata ' + json.dumps(safe_user, ensure_ascii=False))
    expected_uid = metadata['uid']
    history = cann_client.unwrap(cann_client.request(
        'GET', '/submissions/user/' + expected_uid + '/problem/' + PROBLEM,
        token,
    ))
    used = count_today(history)
    return dict(metadata, cookie=token, usable_for_submit=True), used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credential-input', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--problem-id', default='6ae3fad11dac5d235b148620')
    parser.add_argument('--provided-positions', type=int, nargs='+', choices=(1, 2, 3), default=[1, 2, 3],
                        help='Input positions to refresh; identity matching always uses saved platform UIDs')
    args = parser.parse_args()
    global PROBLEM
    PROBLEM = args.problem_id
    if stat.S_IMODE(args.credential_input.stat().st_mode) & 0o077:
        raise RefreshError('Credential input permissions must exclude group and other users')
    headers = json.loads(args.credential_input.read_text())
    if set(headers) != set('ABC') or not all(isinstance(value, str) for value in headers.values()):
        raise RefreshError('Credential input must map exactly A, B and C to cookie headers')
    before = json.loads(cann_client.ACCOUNT_FILE.read_text())
    originals = {row['label']: row for row in before}
    updates, summary = {}, []
    by_uid = {originals[label]['uid']: label for label in 'ABC'}
    if len(by_uid) != 3:
        raise RefreshError('Saved account identities are not unique')
    if len(set(args.provided_positions)) != len(args.provided_positions):
        raise RefreshError('Input positions must be unique')
    for position in args.provided_positions:
        input_label = 'ABC'[position - 1]
        values, used = exchange(headers[input_label], set(by_uid))
        label = by_uid[values['uid']]
        if label in updates:
            raise RefreshError('Two provided cookies authenticated as the same account')
        updates[label] = values
        summary.append(dict(label=label, provided_position=position, identity_matched=True, authenticated_history_http=200,
                            used_today=used, remaining_today=max(0, 50-used),
                            expires_utc=datetime.fromtimestamp(values['expires_epoch'], timezone.utc).isoformat()))
        print(json.dumps(dict(label=label, provided_position=position, identity_matched=True, authenticated_history_http=200), ensure_ascii=False), flush=True)
    if len(updates) != len(args.provided_positions):
        raise RefreshError('Input positions do not cover distinct saved identities')
    lock = cann_client.ACCOUNT_FILE.with_name(cann_client.ACCOUNT_FILE.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        rows = json.loads(cann_client.ACCOUNT_FILE.read_text())
        for row in rows:
            label = row.get('label')
            if label in updates:
                if row['uid'] != originals[label]['uid']:
                    raise RefreshError('Concurrent account identity change; no credentials installed')
                row.update(updates[label])
        backup = cann_client.ACCOUNT_FILE.with_name('accounts-before-refresh-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f') + '.json')
        atomic_json(backup, json.loads(cann_client.ACCOUNT_FILE.read_text()))
        atomic_json(cann_client.ACCOUNT_FILE, rows)
    report = dict(status='all_three_accounts_updated_and_authenticated' if set(updates) == set('ABC') else 'known_accounts_updated_and_authenticated',
                  updated_utc=datetime.now(timezone.utc).isoformat(), accounts=summary,
                  account_file=str(cann_client.ACCOUNT_FILE), backup_file=str(backup),
                  permissions=oct(stat.S_IMODE(cann_client.ACCOUNT_FILE.stat().st_mode)),
                  no_submission_POSTs_by_this_tool=True)
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Exceptions may carry redirect queries or headers. Never print them.
        print(json.dumps(dict(status='authentication_refresh_failed', error_type=type(error).__name__,
                              credentials_installed=False,
                              reason=str(error) if isinstance(error, RefreshError) else 'Nonpublic error details suppressed',
                              last_request=LAST_REQUEST), ensure_ascii=False), flush=True)
        raise SystemExit(1)
