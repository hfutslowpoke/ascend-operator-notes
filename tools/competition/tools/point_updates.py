#!/usr/bin/env python3
"""Append an agent's own journal and atomically promote its owned point rows.

This edits the shared development document only; it never runs an NPU or POSTs.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT = ROOT / 'POINT-UPDATES.md'
SHAPES = {
    'C0': (1024, 2048, 1024, 16), 'C1': (1024, 1024, 1024, 8),
    'C2': (4, 3, 5, 3), 'C3': (8192, 256, 256, 128),
    'C4': (2048, 2048, 768, 16), 'R1': (64, 128, 128, 1),
    'R2': (2048, 1024, 768, 32), 'R3': (4096, 512, 768, 64),
    'I1': (1, 1, 1, 5), 'I2': (513, 8191, 513, 3),
    'I3': (2047, 1025, 769, 12), 'I4': (4093, 1019, 765, 8),
    'I4b': (2559, 1027, 773, 32), 'I5': (9217, 513, 385, 24),
    'I6': (2559, 1031, 1025, 16),
}
OWNERS = {point: ('regular6' if point in {'C0', 'C1', 'C3', 'C4', 'R2', 'R3'}
                  else 'oddstride' if point in {'I2', 'I3', 'I4b', 'I6'} else 'main')
          for point in SHAPES}
LEVELS = {'平台公开观测', '平台隐藏受控', '本地优选', '继承基线'}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def cell(value):
    return str(value).replace('|', '&#124;').replace('\n', '<br>')


def link(path):
    path = str(path)
    if any(ch in path for ch in '\n\r<>[]|'):
        raise ValueError('Path contains unsupported Markdown characters')
    return f'[{path}](<{path}>)'


def point_rows(document):
    rows = {}
    for line in document.splitlines():
        match = re.match(r'^\| (C[0-4]|R[1-3]|I[1-6]|I4b) \|', line)
        if not match:
            continue
        point = match[1]
        if point in rows:
            raise ValueError(f'Duplicate point row: {point}')
        rows[point] = line
    if set(rows) != set(SHAPES):
        raise ValueError('Document must contain exactly the fixed15 point rows')
    return rows


def source_identity(point, row):
    """Separate the current carrier from a documented unchanged algorithm."""
    source = Path(re.search(r'\]\(<([^>]+)>\)', row)[1]).resolve(strict=True)
    sha256 = re.search(r'SHA256 `([a-f0-9]{64})`', row)[1]
    identity = dict(source=str(source), source_sha256=sha256,
                    algorithm_source=str(source), algorithm_sha256=sha256)
    mapping = source.parent / 'POINT-COMPONENTS.json'
    if mapping.exists():
        data = json.loads(mapping.read_text())
        if data['carrier']['source_sha256'] != sha256:
            raise ValueError(f'{point}: carrier identity differs from POINT-COMPONENTS.json')
        component = data['points'].get(point, {})
        # New component handoffs use source/source_sha256 for the donor;
        # older complete mappings use algorithm_source/algorithm_sha256.
        identity.update(algorithm_source=component.get('algorithm_source', component.get('source', str(source))),
                        algorithm_sha256=component.get('algorithm_sha256', component.get('source_sha256', sha256)),
                        integration_guide=component.get('integration_guide', str(mapping)),
                        component_mapping=str(mapping))
    return identity


def render_row(point, source, sha256, level, summary, record_id, date):
    return (f'| {point} | {OWNERS[point]} | {"public" if point.startswith("C") else "hidden"} | '
            f'{"/".join(map(str, SHAPES[point]))} | {link(source)}<br>SHA256 `{sha256}` | '
            f'{cell(level)}：{cell(summary)} | [{cell(date)}](#{record_id}) |')


def log_sections(document):
    pattern = r'<!-- agent:([a-z0-9_-]+):begin -->(.*?)<!-- agent:\1:end -->'
    sections = {}
    for match in re.finditer(pattern, document, re.S):
        if match[1] in sections:
            raise ValueError('Duplicate agent journal section')
        sections[match[1]] = match[2]
    return sections


def checked_source(path, expected=None):
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError('Source must be an existing immutable file')
    raw = source.read_bytes()
    sha256 = digest(raw)
    if expected and sha256 != expected:
        raise ValueError('Source SHA256 differs from the event')
    return source, sha256, raw


def append_event(document, agent, event):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', agent):
        raise ValueError('Use a stable lowercase agent ID')
    if event.get('agent') != agent:
        raise ValueError('Event author must equal --agent')
    points = event.get('points', [])
    if not points or len(set(points)) != len(points) or any(p not in SHAPES for p in points):
        raise ValueError('Provide unique known point IDs')
    for key in ['title', 'conclusion', 'local', 'platform', 'integration']:
        if not isinstance(event.get(key), str) or not event[key].strip():
            raise ValueError(f'Missing nonempty {key}')
    source, sha256, raw = checked_source(event['source'], event.get('source_sha256'))
    evidence = [Path(p).resolve(strict=True) for p in event.get('evidence', [])]
    if not evidence or any(not p.is_file() for p in evidence):
        raise ValueError('Attach existing result/evidence files')
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    date = now.strftime('%Y-%m-%d %H:%M:%S +08:00')
    record_id = event.get('id') or f'{agent}-{now.strftime("%Y%m%dT%H%M%S")}-{uuid.uuid4().hex[:8]}'
    if not re.fullmatch(r'[a-z0-9][a-zA-Z0-9_-]{0,95}', record_id):
        raise ValueError('Invalid record ID')
    if f'id="{record_id}"' in document:
        raise ValueError('Record ID already exists; entries are append-only')
    rows = point_rows(document)
    promotions = event.get('promote', [])
    seen = set()
    for change in promotions:
        point = change['point']
        if point in seen or point not in points or OWNERS.get(point) != agent:
            raise ValueError('Only the point owner can promote its row, once per event')
        seen.add(point)
        if event['conclusion'] not in {'improved', 'rollback'}:
            raise ValueError('Recommendation changes require improved or rollback with recorded evidence')
        if change['expected_row_sha256'] != digest(rows[point].encode()):
            raise ValueError(f'{point} changed since the snapshot; re-read and reassess')
        if change['level'] not in LEVELS or change['level'] == '继承基线':
            raise ValueError('Promotion needs public/controlled-hidden/local evidence')
        if not change.get('summary'):
            raise ValueError('Promotion requires a point-specific result summary')
    if promotions:
        import dispatch_policy
        dispatch_policy.validate(raw.decode())
    before = log_sections(document)
    details = [f'<a id="{record_id}"></a>', f'### {date} · {agent} · {cell(event["title"])}', '',
               f'- 记录ID：`{record_id}`；目标：{", ".join(points)}；结论：`{event["conclusion"]}`。',
               f'- 源码：{link(source)}；SHA256 `{sha256}`。',
               f'- 实验时间：{cell(event.get("observed_at", "见平台job或本地原始报告"))}。',
               f'- 母本/变量：{cell(event.get("baseline", "初始化或只读观察；未主张受控比较"))}。',
               f'- 本地：{cell(event["local"])}。', f'- 平台：{cell(event["platform"])}。',
               f'- 集成说明：{cell(event["integration"])}。',
               f'- 备注：{cell(event.get("notes", "无"))}。',
               '- 证据：' + '；'.join(link(p) for p in evidence) + '。',
               '- 索引变更：' + (', '.join(c['point'] for c in promotions) if promotions else '无') + '。', '']
    entry = '\n'.join(details) + '\n'
    if '<!-- agent:' in entry or '<!-- /agent' in entry:
        raise ValueError('Journal marker injection is not allowed')
    marker = f'<!-- agent:{agent}:end -->'
    if agent not in before:
        document += f'\n### agent `{agent}`\n\n<!-- agent:{agent}:begin -->\n\n{marker}\n'
    document = document.replace(marker, entry + marker, 1)
    for change in promotions:
        point = change['point']
        new_row = render_row(point, source, sha256, change['level'], change['summary'], record_id, date)
        document = document.replace(rows[point], new_row, 1)
    after = log_sections(document)
    for other, content in before.items():
        if other != agent and after.get(other) != content:
            raise ValueError('Refusing to change another agent journal')
    for point, line in rows.items():
        if point not in seen and point_rows(document)[point] != line:
            raise ValueError('Refusing to change an unrelated point row')
    return document, record_id, sorted(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path, default=DEFAULT_DOCUMENT)
    sub = parser.add_subparsers(dest='action', required=True)
    show = sub.add_parser('snapshot', help='Read point row hashes before proposing an update')
    show.add_argument('--point', choices=list(SHAPES), action='append')
    append = sub.add_parser('append', help='Append own log and optional owned-point promotions')
    append.add_argument('--agent', required=True)
    append.add_argument('--event', type=Path, required=True)
    append.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    document = args.document.resolve(strict=True)
    lock_path = document.with_name(document.name + '.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = document.read_text()
        if args.action == 'snapshot':
            rows = point_rows(original)
            print(json.dumps({p: {'owner': OWNERS[p], 'row_sha256': digest(line.encode()), 'row': line,
                                  **source_identity(p, line)}
                              for p, line in rows.items() if not args.point or p in args.point},
                             ensure_ascii=False, indent=2))
            return
        event = json.loads(args.event.read_text())
        updated, record_id, promoted = append_event(original, args.agent, event)
        if not args.dry_run:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=document.parent,
                                             prefix=document.name + '.', delete=False) as tmp:
                tmp.write(updated)
                temporary = Path(tmp.name)
            try:
                temporary.chmod(document.stat().st_mode & 0o777)
                temporary.replace(document)
            finally:
                temporary.unlink(missing_ok=True)
        print(json.dumps({'record_id': record_id, 'promoted': promoted,
                          'document': str(document), 'written': not args.dry_run}, ensure_ascii=False))


if __name__ == '__main__':
    main()
