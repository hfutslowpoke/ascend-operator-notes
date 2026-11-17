import argparse
import csv
import inspect
import json
from pathlib import Path
import numpy as np


def _validate(x, weight, scale, group_list):
    arrays = (x, weight, scale, group_list)
    dtypes = (np.int8, np.int8, np.float32, np.int64)
    if any((not isinstance(a, np.ndarray) for a in arrays)):
        raise TypeError('all inputs must be numpy arrays')
    if any((a.dtype != dtype for (a, dtype) in zip(arrays, dtypes))):
        raise TypeError('expected int8, int8, float32, int64 inputs')
    if any((not a.flags.c_contiguous for a in arrays)):
        raise ValueError('only contiguous ND tensors are supported')
    if x.ndim != 2 or weight.ndim != 3 or scale.ndim != 2 or (group_list.ndim != 1):
        raise ValueError('expected input ranks 2, 3, 2, 1')
    (m, k) = x.shape
    (e, wk, n) = weight.shape
    if not (1 <= e <= 128 and 1 <= m <= 65536 and (1 <= k <= 8192) and (1 <= n <= 4096)):
        raise ValueError('E/M/K/N are outside the declared bounds')
    if wk != k or scale.shape != (e, n) or group_list.shape != (e,):
        raise ValueError('incompatible input shapes')
    if np.any(group_list < 0) or np.any(group_list > m) or np.any(group_list[1:] < group_list[:-1]) or (group_list[-1] != m):
        raise ValueError('group_list must be nondecreasing, nonnegative and end at M')
    if not np.all(np.isfinite(scale)) or np.any(scale < 0) or np.any(scale > 1):
        raise ValueError('scale must be finite and in [0, 1]')
    if 16384 * k * float(scale.max()) > 60000:
        raise ValueError('scale can exceed the specified FP16-safe output bound')
    if weight.size > 268435456:
        raise ValueError('INT8 weights must not exceed 256 MiB')
    if sum((a.nbytes for a in arrays)) + 2 * m * n > 536870912:
        raise ValueError('input and output tensors must not exceed 512 MiB')


def _exact_int32_matmul(x, weight):
    return (x.astype(np.float64) @ weight.astype(np.float64)).astype(np.int32)


def _gelu_tanh(z):
    z64 = z.astype(np.float64)
    inner = np.sqrt(2.0 / np.pi) * (z64 + 0.044715 * z64 * z64 * z64)
    tail = np.exp(-2.0 * np.abs(inner))
    return z64 * np.where(inner >= 0, 1.0 / (1.0 + tail), tail / (1.0 + tail))


def impl(x, weight, scale, group_list):
    _validate(x, weight, scale, group_list)
    (m, _) = x.shape
    (e, _, n) = weight.shape
    y = np.empty((m, n), dtype=np.float16)
    start = 0
    for expert in range(e):
        end = int(group_list[expert])
        for row in range(start, end, 256):
            row_end = min(row + 256, end)
            for col in range(0, n, 512):
                col_end = min(col + 512, n)
                c = _exact_int32_matmul(x[row:row_end], weight[
                    expert,
                    :,
                    col:col_end
                ])
                z = c.astype(np.float32) * scale[expert, col:col_end]
                y[row:row_end, col:col_end] = _gelu_tanh(z).astype(np.float16)
        start = end
    return y


def _generate(desc, rng):
    dtype = np.dtype(desc['datatype'])
    shape = tuple(desc['shape'])
    if 'value' in desc:
        result = np.asarray(desc['value'], dtype=dtype)
        if result.shape != shape:
            raise ValueError('explicit value shape differs from JSON shape')
        return result
    (low, high) = desc['range']
    if not low < high:
        raise ValueError('range must be nonempty and left-closed/right-open')
    if np.issubdtype(dtype, np.integer):
        return rng.integers(
            low,
            high,
            size=shape,
            dtype=np.int16
        ).astype(dtype)
    return rng.uniform(low, high, size=shape).astype(dtype)


def _assert_accuracy(actual, golden):
    if actual.shape != golden.shape or actual.dtype != golden.dtype:
        raise AssertionError('output shape/dtype differs from golden')
    actual64 = actual.astype(np.float64)
    golden64 = golden.astype(np.float64)
    error = np.abs(actual64 - golden64)
    absolute_ok = error < 0.001
    relative_ok = (golden64 == 0) | (error < 0.001 * np.abs(golden64))
    if not np.all(np.isfinite(actual64)):
        raise AssertionError('output contains NaN or Inf')
    if not np.all(absolute_ok & relative_ok):
        raise AssertionError('output must meet both absolute and relative errors < 1e-3')


def _check_samples(inputs, y):
    (x, weight, scale, group_list) = inputs
    n = weight.shape[2]
    cols = np.unique(np.linspace(0, n - 1, min(n, 9), dtype=np.int64))
    start = 0
    for (expert, end_value) in enumerate(group_list):
        end = int(end_value)
        if end > start:
            rows = np.unique([start, (start + end - 1) // 2, end - 1])
            c64 = x[rows].astype(np.int64) @ weight[expert][
                :,
                cols
            ].astype(np.int64)
            z = c64.astype(np.float32) * scale[expert, cols]
            z64 = z.astype(np.float64)
            reference = (0.5 * z64 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z64 + 0.044715 * z64 ** 3)))).astype(np.float16)
            _assert_accuracy(y[np.ix_(rows, cols)], reference)
        start = end


def _check_metadata(config, rows):
    op = config['op']
    names = [d['name'] for d in op['input_desc']] + [d['name'] for d in op['attr_desc']]
    if list(inspect.signature(impl).parameters) != names:
        raise AssertionError('impl signature and JSON descriptors differ')
    if op['name'] != 'GroupedMatmulGelu' or op['impl'] != 'GroupedMatmulGelu:impl':
        raise AssertionError('operator name or impl differs')
    if [d['datatype'] for d in op['input_desc']] != [
        ['int8'],
        ['int8'],
        ['float32'],
        ['int64']
    ] or [d['datatype'] for d in op['output_desc']] != [['float16']]:
        raise AssertionError('descriptor dtypes differ')
    cases = config['npu_cases']
    if len(cases) != 15 or len(rows) != 15:
        raise AssertionError('exactly 15 JSON/CSV cases are required')
    regular_count = 0
    for (cid, (case, row)) in enumerate(zip(cases, rows), 1):
        desc = case['input']
        if len(desc) != 4 or [d['datatype'] for d in desc] != [
            'int8',
            'int8',
            'float32',
            'int64'
        ]:
            raise AssertionError('case input count/dtypes differ from descriptors')
        if any((('value' in d) == ('range' in d) for d in desc)):
            raise AssertionError('each input must specify exactly one of value/range')
        (m, k) = desc[0]['shape']
        (e, wk, n) = desc[1]['shape']
        ends = desc[3]['value']
        sizes = np.diff([0] + ends).tolist()
        axes = [m, k, n, *sizes]
        alignment = 128 if all((v % 128 == 0 for v in axes)) else 64 if all((v % 64 == 0 for v in axes)) else 0
        regular_count += int(alignment > 0)
        if not alignment and any((v % 64 == 0 for v in (m, k, n))):
            raise AssertionError('irregular cases require M, K and N all non-64-aligned')
        if not (wk == k and desc[2]['shape'] == [
            e,
            n
        ] and (desc[3]['shape'] == [e])):
            raise AssertionError('JSON input shapes differ')
        if len(ends) != e or min(sizes) < 0 or ends[-1] != m or (case['attr'] != []):
            raise AssertionError('invalid group_list or unexpected attrs')
        if case['output'] != [{'datatype': 'float16', 'shape': [m, n]}]:
            raise AssertionError('JSON output shape/dtype differs')
        expected = {
            'case_id': str(cid),
            'E': str(e),
            'M': str(m),
            'N': str(n),
            'K': str(k),
            'dtype': 'int8->fp16',
            'aligned': 'YES' if alignment else 'NO',
            'alignment': str(alignment) if alignment else 'NONE',
            'est_time': '待实测'
        }
        for (field, value) in expected.items():
            if row[field] != value:
                raise AssertionError(f'CSV case {cid} field {field} differs')
        for (name, d) in zip(names, desc):
            if json.loads(row[f'{name}_shape']) != d['shape']:
                raise AssertionError('CSV input shape differs')
        for (name, d) in zip(names[:3], desc[:3]):
            actual_range = row[f'{name}_range']
            matches = json.loads(actual_range) == d['range'] if 'range' in d else actual_range == 'value:see JSON'
            if not matches:
                raise AssertionError('CSV input range differs')
        if json.loads(row['group_list_value']) != ends or json.loads(row['group_sizes']) != sizes:
            raise AssertionError('CSV group values differ')
    if regular_count != 7:
        raise AssertionError('expected 7 regular and 8 irregular cases')


def _check_boundaries():
    x = np.full((1, 8192), 127, dtype=np.int8)
    weight = np.full((8192, 2), 127, dtype=np.int8)
    weight[-1, 0] = 126
    exact = x.astype(np.int64) @ weight.astype(np.int64)
    np.testing.assert_array_equal(_exact_int32_matmul(x, weight), exact)
    x = np.array([[1, -2], [-1, 2], [3, 1]], dtype=np.int8)
    weight = np.tile(np.eye(2, dtype=np.int8), (5, 1, 1))
    scale = np.array([[
        1,
        1
    ], [0.1, 0], [0.2, 0.2], [0.5, 0.25], [0.3, 0.3]], dtype=np.float32)
    groups = np.array([0, 2, 2, 3, 3], dtype=np.int64)
    y = impl(x, weight, scale, groups)
    _check_samples((x, weight, scale, groups), y)
    np.testing.assert_array_equal(y[:2, 1], np.zeros(2, dtype=np.float16))
    if not y[1, 0] < 0:
        raise AssertionError('GELU must retain small negative outputs')
    changed_groups = np.array([1, 1, 2, 2, 3], dtype=np.int64)
    changed_y = impl(x, weight, scale, changed_groups)
    _check_samples((x, weight, scale, changed_groups), changed_y)
    if np.array_equal(y, changed_y):
        raise AssertionError('output must depend on actual group_list values')
    for invalid in (np.array([
        0,
        2,
        1,
        3,
        3
    ], dtype=np.int64), np.array([
        0,
        2,
        2,
        3,
        4
    ], dtype=np.int64), np.array([
        0,
        2 ** 63 - 1,
        -2 ** 63 + 1,
        0,
        3
    ], dtype=np.int64)):
        try:
            impl(x, weight, scale, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('golden must reject invalid group_list')
    reference = np.ones((2, 2), dtype=np.float16)
    for invalid in (np.ones((
        1,
        2
    ), dtype=np.float16), reference.astype(np.float32), np.full((
        2,
        2
    ), np.nan, dtype=np.float16), np.array([[
        1,
        1
    ], [1, 1.01]], dtype=np.float16)):
        try:
            _assert_accuracy(invalid, reference)
        except AssertionError:
            pass
        else:
            raise AssertionError('accuracy checker accepted an invalid output')
    _assert_accuracy(np.zeros((
        2,
        2
    ), dtype=np.float16), np.zeros((2, 2), dtype=np.float16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--case',
        type=int,
        nargs='+',
        help='run selected case IDs (1..15)'
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'GroupedMatmulGelu.json').read_text(encoding='utf-8'))
    with (root / 'GroupedMatmulGelu_testcases.csv').open(
        encoding='utf-8',
        newline=''
    ) as f:
        rows = list(csv.DictReader(f))
    _check_metadata(config, rows)
    _check_boundaries()
    if args.case and any((cid < 1 or cid > 15 for cid in args.case)):
        parser.error('--case IDs must be in 1..15')
    for (cid, case) in enumerate(config['npu_cases'], 1):
        if args.case and cid not in args.case:
            continue
        rng = np.random.default_rng(20261010 + cid)
        inputs = tuple((_generate(d, rng) for d in case['input']))
        y = impl(*inputs)
        if y.shape != tuple(case['output'][0]['shape']) or y.dtype != np.float16:
            raise AssertionError('golden output shape/dtype differs')
        if not np.all(np.isfinite(y)):
            raise AssertionError('golden output contains NaN or Inf')
        _check_samples(inputs, y)
        print(
            f'Case {cid:02d}: y.shape={y.shape}, y.dtype={y.dtype}, y_range=[{float(y.min()):.6f}, {float(y.max()):.6f}]',
            flush=True
        )
        del inputs, y
    print('All tests passed!')
