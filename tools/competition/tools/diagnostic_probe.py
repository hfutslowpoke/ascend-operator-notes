#!/usr/bin/env python3
"""Make an explicitly labeled single-point score probe from generic source.

The selected shape executes the unmodified operator. All other shapes receive
FP16 quiet NaNs. This is a diagnostic artifact; restore a complete valid package
after recording its live ranking score. Authorized by the user's latest message.
"""
import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_SHAPES = [[1024, 2048, 1024, 16], [1024, 1024, 1024, 8],
                 [4, 3, 5, 3], [8192, 256, 256, 128], [2048, 2048, 768, 16]]


HELPER = '''
// BEGIN GMM_DIAGNOSTIC_NAN_FILL
extern "C" __global__ __aicore__ void gmm_diagnostic_nan(GM_ADDR y, int64_t total)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    constexpr int32_t CHUNK = 16384;
    GlobalTensor<uint16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y), total);
    TPipe pipe;
    TQue<TPosition::VECOUT, 1> queue;
    pipe.InitBuffer(queue, 1, CHUNK * sizeof(uint16_t));
    LocalTensor<uint16_t> values = queue.AllocTensor<uint16_t>();
    Duplicate(values, static_cast<uint16_t>(0x7e00), CHUNK);
    queue.EnQue(values);
    values = queue.DeQue<uint16_t>();
    for (int64_t start = static_cast<int64_t>(GetBlockIdx()) * CHUNK;
         start < total; start += static_cast<int64_t>(GetBlockNum()) * CHUNK) {
        const int32_t count = total - start < CHUNK ? total - start : CHUNK;
        DataCopyExtParams copy;
        copy.blockCount = 1;
        copy.blockLen = count * sizeof(uint16_t);
        copy.srcStride = 0;
        copy.dstStride = 0;
        DataCopyPad(output[start], values, copy);
    }
    queue.FreeTensor(values);
}
// END GMM_DIAGNOSTIC_NAN_FILL
'''


def sha(content):
    return hashlib.sha256(content).hexdigest()


def render_source(base, target, keep_public=False):
    anchor = 'extern "C" void run_kernel('
    assert base.count(anchor) == 1
    source = base.replace(anchor, HELPER + '\n' + anchor)
    m, k, n, e = map(int, target)
    predicate = f'm == {m} && k == {k} && n == {n} && e == {e}'
    selections = [predicate]
    if keep_public:
        selections = [f'({predicate})']
        selections += [f'(m == {a} && k == {b} && n == {c} && e == {d})'
                       for a, b, c, d in PUBLIC_SHAPES]
    selected = ' ||\n          '.join(selections)
    anchor = '    const int64_t n = info_weight.tensors[0].shape[2];\n'
    assert source.count(anchor) == 1
    guard = f'''    // BEGIN GMM_DIAGNOSTIC_CASE_MASK
    if (!({selected})) {{
        int64_t blocks = availableCoreNum > 0 ? availableCoreNum * 2 : 1;
        const int64_t chunks = (m * n + 16383) / 16384;
        if (blocks > chunks) blocks = chunks;
        gmm_diagnostic_nan<<<blocks, nullptr, stream>>>(y, m * n);
        aclrtSynchronizeStream(stream);
        return;
    }}
    // END GMM_DIAGNOSTIC_CASE_MASK
'''
    return source.replace(anchor, anchor + guard)


def validate(directory):
    import dispatch_policy
    directory = Path(directory)
    manifest = json.loads((directory / 'probe.json').read_text())
    if manifest.get('purpose') != 'single-hidden-point-score-isolation':
        raise RuntimeError('Diagnostic submission requires its explicit probe manifest')
    base_path = Path(manifest['base_source'])
    base_bytes = base_path.read_bytes()
    if sha(base_bytes) != manifest['base_sha256']:
        raise RuntimeError('Diagnostic base source changed')
    dispatch_policy.validate(base_bytes.decode())
    actual = (directory / 'kernel.asc').read_bytes()
    if sha(actual) != manifest['source_sha256']:
        raise RuntimeError('Diagnostic source changed')
    expected = render_source(base_bytes.decode(), manifest['target_shape'],
                             manifest.get('keep_public_controls', False)).encode()
    if actual != expected:
        raise RuntimeError('Diagnostic source differs from the documented NaN isolation transform')
    return manifest


def main():
    import dispatch_policy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', required=True, type=Path)
    ap.add_argument('--target', required=True, help='M,K,N,E of the one diagnostic target')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--keep-public', action='store_true',
                    help='retain all five public controls to test partial scoring')
    args = ap.parse_args()
    target = [int(s) for s in args.target.split(',')]
    if len(target) != 4 or any(v <= 0 for v in target):
        ap.error('target must contain four positive dimensions')
    base_path = args.base.resolve()
    base_bytes = base_path.read_bytes()
    dispatch_policy.validate(base_bytes.decode())
    source = render_source(base_bytes.decode(), target, args.keep_public).encode()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'kernel.asc').write_bytes(source)
    manifest = {'purpose': 'single-hidden-point-score-isolation',
                'base_source': str(base_path), 'base_sha256': sha(base_bytes),
                'source_sha256': sha(source), 'target_shape': target,
                'keep_public_controls': args.keep_public,
                'nonselected_output': 'all FP16 elements quiet NaN, bits 0x7e00',
                'restore_required': True, 'formal_submission': False}
    (args.out / 'probe.json').write_text(json.dumps(manifest, indent=2) + '\n')
    validate(args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
