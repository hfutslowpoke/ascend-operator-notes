"""Independent CPU references for the 2026 manufacturing operator problems.

These functions follow the public problem statements, not any hidden golden.
They are intentionally small and explicit; use them to test adapters and to
settle tie/empty/first-K questions. They are not Ascend implementations and
must not be reported as official results.
"""

from __future__ import annotations

import numpy as np


def segment_min_csr(src, indptr):
    """Official SegmentMinCsr: out + arg_out, empty 0/E, first-match tie."""
    src = np.asarray(src)
    indptr = np.asarray(indptr, dtype=np.int64)
    if indptr.ndim != 1 or indptr.size < 1:
        raise ValueError("indptr must be 1-D with at least one element")
    e, k = src.shape[0], int(np.prod(src.shape[1:], dtype=np.int64))
    n = indptr.size - 1
    out = np.zeros((n, k), dtype=src.dtype)
    arg = np.full((n, k), e, dtype=np.int64)
    flat = src.reshape(e, k)
    for r in range(n):
        start, end = int(indptr[r]), int(indptr[r + 1])
        if end < start or start < 0 or end > e:
            raise ValueError("indptr must be monotone and within [0, E]")
        if start == end:
            continue
        chunk = flat[start:end]
        out[r] = chunk.min(axis=0)
        arg[r] = start + chunk.argmin(axis=0)  # argmin returns first match
    return out.reshape((n,) + src.shape[1:]), arg.reshape((n,) + src.shape[1:])


def furthest_point_sample(points, npoint, fp16_mode="per_dim"):
    """Official FPS: first point 0, npoint rounds, smallest-index tie.

    fp16_mode:
      - "per_dim": torch_cluster/950-community style per-feature half rounding
      - "fp32": accumulate squared distances in float32, cast back at the end
    """
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must be [B, N, 3]")
    b, n, _ = points.shape
    if not (1 <= npoint <= n):
        raise ValueError("1 <= npoint <= N")
    out = np.zeros((b, npoint), dtype=np.int32)
    for bi in range(b):
        p = points[bi]
        selected = 0
        if p.dtype == np.float16:
            dist = np.full(n, 5e4, dtype=np.float16)
        else:
            dist = np.full(n, np.inf, dtype=np.float32)
        for t in range(npoint):
            out[bi, t] = selected
            center = p[selected]
            if p.dtype == np.float16:
                if fp16_mode == "per_dim":
                    current = np.zeros(n, dtype=np.float16)
                    for f in range(p.shape[1]):
                        delta = (p[:, f] - center[f]).astype(np.float16)
                        square = (delta.astype(np.float32) ** 2).astype(np.float16)
                        current = (current.astype(np.float32) + square.astype(np.float32)).astype(np.float16)
                elif fp16_mode == "fp32":
                    current = ((p.astype(np.float32) - center.astype(np.float32)) ** 2).sum(axis=1).astype(np.float16)
                else:
                    raise ValueError("fp16_mode must be 'per_dim' or 'fp32'")
            else:
                current = ((p.astype(np.float32) - center.astype(np.float32)) ** 2).sum(axis=1)
            dist = np.minimum(dist, current)
            selected = int(np.argmax(dist))  # first maximum -> smallest index
    return out


def radius(x, y, ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index=False):
    """Official Radius: fixed slots, scan-order first-K, d2<=r2, miss (-1,-1)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ptr_x = np.asarray(ptr_x, dtype=np.int64)
    ptr_y = np.asarray(ptr_y, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x/y must be 2-D with the same D")
    if x.shape[1] not in (2, 3):
        raise ValueError("D must be 2 or 3")
    if ptr_x.size != ptr_y.size:
        raise ValueError("ptr_x/ptr_y must have the same batch count")
    k = int(max_num_neighbors)
    if k <= 0:
        raise ValueError("max_num_neighbors must be positive")
    m = y.shape[0]
    out = np.full((2, m * k), -1, dtype=np.int64)
    r2 = np.float32(r) * np.float32(r)
    for i in range(m):
        bi = int(np.searchsorted(ptr_y, i, side="right") - 1)
        if bi < 0 or bi + 1 >= ptr_y.size:
            raise ValueError("ptr_y does not cover query index")
        xs, xe = int(ptr_x[bi]), int(ptr_x[bi + 1])
        cnt = 0
        if r > 0:
            for j in range(xs, xe):
                if ignore_same_index and j == i:
                    continue
                d = x[j] - y[i]
                d2 = float(np.dot(d, d))
                if d2 <= float(r2):
                    base = i * k + cnt
                    out[0, base] = i
                    out[1, base] = j
                    cnt += 1
                    if cnt == k:
                        break
    return out


def ball_query(xyz, new_xyz, radius, nsample):
    """Official BallQuery: strict d2<r2, scan-order first-K, first-hit padding."""
    xyz = np.asarray(xyz, dtype=np.float32)
    new_xyz = np.asarray(new_xyz, dtype=np.float32)
    if xyz.ndim != 3 or new_xyz.ndim != 3 or xyz.shape[2] != 3 or new_xyz.shape[2] != 3:
        raise ValueError("xyz/new_xyz must be [B, N, 3]")
    if xyz.shape[0] != new_xyz.shape[0]:
        raise ValueError("batch mismatch")
    b, n, _ = xyz.shape
    m = new_xyz.shape[1]
    ns = int(nsample)
    if radius <= 0 or ns <= 0:
        raise ValueError("radius/nsample must be positive")
    out = np.zeros((b, m, ns), dtype=np.int32)
    r2 = np.float32(radius) * np.float32(radius)
    for bi in range(b):
        for j in range(m):
            cnt = 0
            for k in range(n):
                d = xyz[bi, k] - new_xyz[bi, j]
                d2 = float(np.dot(d, d))
                if d2 < float(r2):
                    if cnt == 0:
                        out[bi, j, :] = k
                    out[bi, j, cnt] = k
                    cnt += 1
                    if cnt == ns:
                        break
    return out


def cylinder_query(xyz, new_xyz, rot, radius, hmin, hmax, nsample):
    """Official CylinderQuery: rot^T*delta, strict bounds, first-K, first-hit padding."""
    xyz = np.asarray(xyz, dtype=np.float32)
    new_xyz = np.asarray(new_xyz, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    if xyz.ndim != 3 or new_xyz.ndim != 3 or xyz.shape[2] != 3 or new_xyz.shape[2] != 3:
        raise ValueError("xyz/new_xyz must be [B, N, 3]")
    if rot.shape != (xyz.shape[0], new_xyz.shape[1], 3, 3):
        raise ValueError("rot must be [B, M, 3, 3]")
    if radius <= 0 or nsample <= 0 or not (hmin < hmax):
        raise ValueError("invalid radius/hmin/hmax/nsample")
    b, n, _ = xyz.shape
    m = new_xyz.shape[1]
    ns = int(nsample)
    out = np.zeros((b, m, ns), dtype=np.int32)
    r2 = np.float32(radius) * np.float32(radius)
    for bi in range(b):
        for j in range(m):
            r = rot[bi, j]
            cnt = 0
            for k in range(n):
                d = xyz[bi, k] - new_xyz[bi, j]
                x_l = r[0, 0] * d[0] + r[1, 0] * d[1] + r[2, 0] * d[2]
                y_l = r[0, 1] * d[0] + r[1, 1] * d[1] + r[2, 1] * d[2]
                z_l = r[0, 2] * d[0] + r[1, 2] * d[1] + r[2, 2] * d[2]
                if y_l * y_l + z_l * z_l < float(r2) and hmin < x_l < hmax:
                    if cnt == 0:
                        out[bi, j, :] = k
                    out[bi, j, cnt] = k
                    cnt += 1
                    if cnt == ns:
                        break
    return out


def spmm_sum(rowptr, col, value, mat):
    """Official SpmmSum: CSR x dense, empty rows zero, fp32 accumulation."""
    rowptr = np.asarray(rowptr, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    value = np.asarray(value)
    mat = np.asarray(mat)
    if rowptr.ndim != 1 or rowptr.size < 2:
        raise ValueError("rowptr must have M+1 entries")
    m, n = rowptr.size - 1, mat.shape[0]
    if mat.ndim != 2 or value.shape != (col.size,) or col.size != rowptr[-1]:
        raise ValueError("inconsistent CSR/value/mat shapes")
    if np.any(col < 0) or np.any(col >= n):
        raise ValueError("col out of range")
    out = np.zeros((m, mat.shape[1]), dtype=value.dtype)
    for r in range(m):
        acc = np.zeros(mat.shape[1], dtype=np.float32)
        for e in range(int(rowptr[r]), int(rowptr[r + 1])):
            acc += np.float32(value[e]) * mat[int(col[e])].astype(np.float32)
        out[r] = acc.astype(value.dtype)
    return out


def spspmm(index_a, value_a, index_b, value_b, m, n, k):
    """Official Spspmm: COO x COO, fixed U=m*n slots, zero sums retained."""
    index_a = np.asarray(index_a, dtype=np.int64)
    index_b = np.asarray(index_b, dtype=np.int64)
    value_a = np.asarray(value_a)
    value_b = np.asarray(value_b)
    if index_a.shape[0] != 2 or index_b.shape[0] != 2:
        raise ValueError("indexA/indexB must be [2, E]")
    if value_a.shape != (index_a.shape[1],) or value_b.shape != (index_b.shape[1],):
        raise ValueError("value length mismatch")
    u = int(m) * int(n)
    if u <= 0 or u > 250000:
        raise ValueError("U=m*n must be in [1, 250000]")
    b_rows: dict[int, list[tuple[int, np.float32]]] = {}
    for t in range(index_b.shape[1]):
        i, j = int(index_b[0, t]), int(index_b[1, t])
        if not (0 <= i < k and 0 <= j < n):
            raise ValueError("indexB out of range")
        b_rows.setdefault(i, []).append((j, np.float32(value_b[t])))
    acc: dict[tuple[int, int], np.float32] = {}
    for t in range(index_a.shape[1]):
        i, kk = int(index_a[0, t]), int(index_a[1, t])
        if not (0 <= i < m and 0 <= kk < k):
            raise ValueError("indexA out of range")
        av = np.float32(value_a[t])
        for j, bv in b_rows.get(kk, ()):
            key = (i, j)
            acc[key] = acc.get(key, np.float32(0)) + av * bv
    order = sorted(acc)
    if len(order) > u:
        raise ValueError("nnz(C) exceeds U=m*n")
    index_c = np.full((2, u), -1, dtype=np.int64)
    value_c = np.zeros(u, dtype=value_a.dtype)
    for t, key in enumerate(order):
        index_c[0, t], index_c[1, t] = key
        value_c[t] = acc[key].astype(value_a.dtype)
    return index_c, value_c
