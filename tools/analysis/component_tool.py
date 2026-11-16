#!/usr/bin/env python3
"""Reversible v5 component staging. No compiler, NPU, network, or index writes.

All generated images are unselected staging artifacts. C0 and C4 deliberately
share one integration kind: their present SDKs do not justify a point-only gate.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

PARENT = Path('/mnt/workspace/groupmatmulgelu-legal-integration-worker-20261113/opt/v5_restore_C3_reference/kernel.asc')
PARENT_SHA = '5cec55dbaefbeaec67aa63cdcd058cb74210688b8f8fd0a1bc1b5dd590dae8a5'
LIMIT = 524288
OPS = ['%:%:', '<<=', '>>=', '->*', '...', '::', '->', '.*', '++', '--', '<<', '>>',
       '<=', '>=', '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', '%=', '^=', '&=', '|=',
       '##', '<:', ':>', '<%', '%>', '%:']
LEX = re.compile(r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
                 r'|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u8|u|U|L)?\'(?:\\.|[^\'\\])*\''
                 r'|(?:\d|\.\d)(?:[eEpP][+-]|[A-Za-z_0-9.])*|[A-Za-z_$][A-Za-z_0-9$]*|'
                 + '|'.join(re.escape(v) for v in OPS) + r'|[^\s]')
MASK = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u8|u|U|L)?\'(?:\\.|[^\'\\])*\'')


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def tokens(text):
    assert not re.search(r'\b(?:u8|u|U|L)?R"', text), 'Raw strings are intentionally unsupported.'
    assert '\\\n' not in text, 'Spliced lines are intentionally unsupported.'
    return [m[0] for m in LEX.finditer(text) if not m.group('comment')]


@lru_cache(maxsize=65536)
def separator(left, right):
    return (tokens(left + right) != [left, right] or
            (left == '/' and right in ('*', '/')) or
            (left == '.' and right == '.') or
            (left.endswith(('"', "'")) and bool(re.match(r'[A-Za-z_]', right))))


def render(seq):
    seq = list(seq)
    out = []
    for i, token in enumerate(seq):
        if i and separator(seq[i-1], token):
            out.append(' ')
        out.append(token)
    result = ''.join(out)
    assert tokens(result) == seq
    return result


def compact(source):
    """Compact code and #define replacement lists; keep directive boundaries.

    Function-like vs object-like macro declaration syntax is retained. No name,
    literal, macro expansion, entry, template argument, or Host predicate changes.
    The actual CANN preprocessor check is a separate, explicit verification step.
    """
    assert not any(x in tokens(source) for x in ('__LINE__', '__COUNTER__'))
    chunks, pos = [], 0
    for match in re.finditer(r'^[ \t]*#[^\n]*(?:\n|$)', source, flags=re.M):
        before = source[pos:match.start()]
        if tokens(before):
            chunks.append(render(tokens(before)) + '\n')
        line = match[0].rstrip('\n')
        define = re.match(r'[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(.*)', line)
        if define:
            name, rest = define.groups()
            if rest.startswith('('):
                close = rest.index(')')
                head, body = name + rest[:close+1], rest[close+1:]
            else:
                head, body = name, rest
            packed = render(tokens(body))
            chunks.append('#define ' + head + (' ' + packed if packed else '') + '\n')
        else:
            chunks.append(line.strip() + '\n')
        pos = match.end()
    tail = source[pos:]
    if tokens(tail):
        chunks.append(render(tokens(tail)) + '\n')
    result = ''.join(chunks)
    # A raw token check also covers every replacement list and every directive.
    assert tokens(result) == tokens(source), 'Lexical token stream changed.'
    return result


def clean(source):
    # Preserve offsets and braces outside strings/comments.
    return MASK.sub(lambda m: ' ' * len(m[0]), source)


def brace_end(source, position):
    stripped = clean(source)
    brace = stripped.index('{', position)
    depth = 1
    end = brace + 1
    while depth:
        depth += (stripped[end] == '{') - (stripped[end] == '}')
        end += 1
    return end


def namespace_span(source, name):
    hits = list(re.finditer(r'\bnamespace\s+' + re.escape(name) + r'\s*\{', clean(source)))
    assert len(hits) == 1, (name, len(hits))
    return hits[0].start(), brace_end(source, hits[0].start())


def function_span(source, marker):
    assert source.count(marker) == 1, (marker, source.count(marker))
    start = source.index(marker)
    return start, brace_end(source, start)


def rename(source, old, new):
    return render(new if t == old else t for t in tokens(source))


def aliases(source):
    # Only existing exact one-identifier aliases. Grammar macros stay untouched.
    pairs = re.findall(r'^#define\s+((?:RGI|Qj)\d+)\s+([A-Za-z_]\w*)\s*$', source, re.M)
    result = {}
    for short, long in pairs:
        result.setdefault(long, short)
    return result


def encode(component, parent):
    mapping = aliases(parent)
    return render(mapping.get(t, t) for t in tokens(component))


def expand_aliases(source, fragment):
    definitions = dict(re.findall(r'^#define\s+((?:RGI|Qj|OSM)\d+)\s+([^\n]*)$', source, re.M))
    for _ in range(32):
        changed = re.sub(r'\b(?:RGI|Qj|OSM)\d+\b', lambda m: definitions.get(m[0], m[0]), fragment)
        if changed == fragment:
            return fragment
        fragment = changed
    raise AssertionError('Macro expansion depth exceeded.')


class Editor:
    def __init__(self, source, directory):
        self.original = self.source = source
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.steps = []

    def replace(self, start, end, replacement, label):
        old = self.source[start:end]
        number = len(self.steps)
        old_path = self.directory / f'{number:02d}-{label}-old.asc'
        new_path = self.directory / f'{number:02d}-{label}-new.asc'
        old_path.write_text(old)
        new_path.write_text(replacement)
        self.steps.append(dict(label=label, offset=start, old=str(old_path), new=str(new_path),
                               old_sha256=sha(old), new_sha256=sha(replacement)))
        self.source = self.source[:start] + replacement + self.source[end:]

    def once(self, old, new, label):
        assert self.source.count(old) == 1, (label, self.source.count(old))
        start = self.source.index(old)
        self.replace(start, start+len(old), new, label)

    def verify(self):
        restored = undo(self.source, self.steps)
        assert restored == self.original
        return dict(inverse_byte_exact=True, restored_sha256=sha(restored), edits=len(self.steps))


def undo(source, steps):
    for step in reversed(steps):
        old = Path(step['old']).read_text()
        new = Path(step['new']).read_text()
        assert sha(old) == step['old_sha256'] and sha(new) == step['new_sha256']
        at = step['offset']
        assert source[at:at+len(new)] == new, step['label']
        source = source[:at] + old + source[at+len(new):]
    return source


def host_bodies(source):
    result = {}
    for match in re.finditer(r'extern\s+"C"\s+void\s+(\w+)\s*\(', source):
        end = brace_end(source, match.start())
        result[match[1]] = source[match.start():end]
    return result


def c3(editor, component, namespace):
    a, b = namespace_span(component, namespace)
    region = rename(component[a:b], namespace, 'rg_c3_lifetime_pipeline')
    a, b = namespace_span(editor.source, 'rg_c3_lifetime_pipeline')
    editor.replace(a, b, encode(region, editor.original), 'C3-namespace')
    return ['C3'], []


def streamed(editor, component, namespace):
    a, b = namespace_span(component, namespace)
    region = rename(component[a:b], namespace, 'gmm_streamedB_overlap')
    # These shared FP32 helpers must be the mother's exact token implementation.
    for name in ('RgScaleRows', 'RgGeluInplace'):
        marker = '__aicore__ inline void ' + name
        if marker in component:
            ca, cb = function_span(component, marker)
            parent = expand_aliases(editor.original, editor.original)
            pa, pb = function_span(parent, marker)
            assert tokens(component[ca:cb]) == tokens(parent[pa:pb]), name + ' dependency changed'
    a, b = namespace_span(editor.source, 'gmm_streamedB_overlap')
    editor.replace(a, b, encode(region, editor.original), 'C0-C4-shared-namespace')
    return ['C0', 'C4'], []


def c1(editor, component, namespace, prefetch, unitflag):
    assert namespace not in editor.source, 'Candidate namespace collides with parent.'
    a, b = namespace_span(component, namespace)
    region = encode(component[a:b], editor.original)
    # Keep the original B8 supply implementation and all fallback implementations.
    marker = 'template<int P> RGI3 RGI14(1,2) LBI0 void RgC1B8K128('
    a, b = function_span(editor.source, marker)
    args = str(prefetch).lower()
    has_unitflag = bool(re.search(r'template\s*<\s*bool\s+PREFETCH\s*,\s*bool\s+UNITFLAG', component))
    assert has_unitflag == (unitflag is not None), 'Specify --unitflag only for the two-parameter component.'
    if has_unitflag:
        args += ',' + str(unitflag).lower()
    entry = (f'template<int P> RGI3 RGI14(1,2) LBI0 void RgC1B8K128('
             f'RGI0 x,RGI0 w,RGI0 s,RGI0 g,RGI0 y,RGI0 c,int m,int e){{'
             f'{namespace}::RGI11<{args}>(x,w,s,g,y,c,m,P*256,e);}}')
    editor.replace(a, b, region+'\n'+entry, 'C1-component-and-entry')
    a, b = function_span(editor.source, 'extern "C" void rg499_frozen_parent(')
    old = editor.source[a:b]
    switch_start = old.index('   switch(n/256){')
    switch_end = brace_end(old, switch_start)
    switch = old[switch_start:switch_end]
    changed = switch.replace('(x,w,s,g,y,m,e)', '(x,w,s,g,y,(RGI0)c,m,e)')
    assert changed != switch and changed.count('(RGI0)c') == 2
    allocation = ('   void*c=nullptr;\n'
                  '   if(aclrtMalloc(&c,cores*2*64*256*sizeof(RGI1),RGI49)!=ACL_SUCCESS)'
                  'throw std::runtime_error("C1 dual64 workspace allocation");\n')
    new = old[:switch_start] + allocation + changed + old[switch_end:]
    assert new.count('const auto status=RGI38(RGI15);') == 1
    new = new.replace('const auto status=RGI38(RGI15);', 'const auto status=RGI38(RGI15);aclrtFree(c);')
    # The target predicate / core cap / prelaunch estimation are byte-identical.
    assert new[:switch_start] == old[:switch_start]
    editor.replace(a, b, new, 'C1-existing-target-workspace-and-launch')
    return ['C1'], ['rg499_frozen_parent']


def stage(args):
    if args.command == 'apply' and args.kind in ('C0', 'C4'):
        raise AssertionError('C0/C4 cannot be applied independently with the current shared route. Use --kind C0_C4.')
    parent = Path(args.parent).resolve()
    source = parent.read_text()
    if parent == PARENT:
        assert sha(source) == PARENT_SHA
    destination = Path(args.dest).resolve()
    assert not destination.exists(), 'Destination must be new; frozen files are never overwritten.'
    destination.mkdir(parents=True)
    editor = Editor(source, destination/'deltas')
    before_hosts = host_bodies(source)
    component_path = None
    changed, permitted_hosts = [], []
    if args.command == 'apply':
        component_path = Path(args.source).resolve()
        component = component_path.read_text()
        assert args.namespace, '--namespace is required to make extraction explicit.'
        if args.kind == 'C3':
            changed, permitted_hosts = c3(editor, component, args.namespace)
        elif args.kind == 'C0_C4':
            changed, permitted_hosts = streamed(editor, component, args.namespace)
        elif args.kind == 'C1':
            changed, permitted_hosts = c1(editor, component, args.namespace, args.prefetch, args.unitflag)
        else:
            raise AssertionError('C0/C4 cannot be applied independently with the current shared route.')
    after_hosts = host_bodies(editor.source)
    assert before_hosts.keys() == after_hosts.keys()
    for name in before_hosts:
        if name not in permitted_hosts:
            assert before_hosts[name] == after_hosts[name], name + ' Host bytes changed'
    if args.pack or args.command == 'pack':
        packed = compact(editor.source)
        editor.replace(0, len(editor.source), packed, 'token-preserving-lexical-pack')
    inverse = editor.verify()
    (destination/'kernel.asc').write_text(editor.source)
    for name in ('CMakeLists.txt', 'main.asc', 'bench.asc', 'verify_batch.asc'):
        if (parent.parent/name).exists():
            shutil.copy2(parent.parent/name, destination/name)
    result = dict(status='unselected_staging_only_not_built', selected=False, POSTs=0,
                  parent_source=str(parent), parent_sha256=sha(source), source=str(destination/'kernel.asc'),
                  source_sha256=sha(editor.source), source_bytes=len(editor.source.encode()),
                  headroom_bytes=LIMIT-len(editor.source.encode()), within_limit=len(editor.source.encode())<=LIMIT,
                  changed_points=changed, permitted_Host_changes=permitted_hosts,
                  Host_functions_before_lexical_pack_byte_exact_except_permitted=True,
                  all_other_Host_functions_byte_exact=not bool(args.pack or args.command == 'pack'),
                  Host_tokens_preserved_by_lexical_pack=bool(args.pack or args.command == 'pack'),
                  component_source=str(component_path) if component_path else None,
                  component_sha256=sha(component_path.read_bytes()) if component_path else None,
                  component_namespace=getattr(args, 'namespace', None),
                  parent_and_unmentioned_regions_preserved_by_inverse=True,
                  lexical_pack=bool(args.pack or args.command == 'pack'),
                  entry_specializations_removed=0, generic_fallback_removed=False,
                  build_started=False, NPU_started=False, new_platform_gain_proven=False,
                  inverse=inverse, steps=editor.steps)
    save(destination/'manifest.json', result)
    (destination/'DO-NOT-SUBMIT.md').write_text('Unselected source staging only. No build, numeric, OFF, or platform evidence. Root must select and validate a frozen candidate before submission.\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'steps'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('pack', 'apply'):
        part = sub.add_parser(command)
        part.add_argument('--parent', default=str(PARENT))
        part.add_argument('--dest', required=True)
        part.add_argument('--pack', action='store_true')
        if command == 'apply':
            part.add_argument('--kind', choices=['C3', 'C1', 'C0_C4', 'C0', 'C4'], required=True)
            part.add_argument('--source', required=True)
            part.add_argument('--namespace', required=True)
            part.add_argument('--prefetch', action=argparse.BooleanOptionalAction, default=False)
            part.add_argument('--unitflag', action=argparse.BooleanOptionalAction, default=None)
    part = sub.add_parser('restore')
    part.add_argument('--manifest', required=True)
    part.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'restore':
        manifest = json.loads(Path(args.manifest).read_text())
        source = Path(manifest['source']).read_text()
        assert sha(source) == manifest['source_sha256']
        restored = undo(source, manifest['steps'])
        assert sha(restored) == manifest['parent_sha256']
        path = Path(args.output)
        assert not path.exists()
        path.write_text(restored)
        print(json.dumps(dict(restored=str(path), sha256=sha(restored), byte_exact=True)))
    else:
        stage(args)


if __name__ == '__main__':
    main()
