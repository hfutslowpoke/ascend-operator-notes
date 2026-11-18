#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
生成 SegmentMinCsr AclNN 轻量化验收所需的单用例数据
"""

import argparse
from pathlib import Path

import numpy as np

SRC_SHAPE = (8, 2048)
DEFAULT_SEGMENTS = 4
DEFAULT_CASE = "single_case"
DTYPE = np.float32


def write_binary(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    print(f"[INFO] 写入 {path} | shape={array.shape}, dtype={array.dtype}")


def write_shape_file(
    path: Path,
    src_shape,
    indptr_shape,
    out_shape,
    nsegments: int,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype_name = np.dtype(dtype).name
    lines = [
        f"src_shape: {tuple(src_shape)}",
        f"indptr_shape: {tuple(indptr_shape)}",
        f"out_shape: {tuple(out_shape)}",
        f"dtype: {dtype_name}",
        f"nSegments: {nsegments}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] 写入 {path} | 描述 {dtype_name} 形状信息")


def gen_indptr(M: int, n_segments: int, rng: np.random.Generator) -> np.ndarray:
    cuts = sorted(rng.integers(0, M + 1, size=n_segments - 1).tolist())
    return np.array([0] + cuts + [M], dtype=np.int32)


def compute_segment_min_csr(src: np.ndarray, indptr: np.ndarray):
    M = src.shape[0]
    n_segments = len(indptr) - 1
    suffix_shape = src.shape[1:]
    suffix_total = int(np.prod(suffix_shape)) if suffix_shape else 1

    src_view = src.reshape(M, suffix_total)
    fill = np.finfo(src.dtype).max if np.issubdtype(src.dtype, np.floating) else np.iinfo(src.dtype).max
    out = np.full((n_segments, suffix_total), fill, dtype=src.dtype)

    for seg in range(n_segments):
        start = int(indptr[seg])
        end = int(indptr[seg + 1])
        if start < end:
            out[seg] = np.min(src_view[start:end], axis=0)

    return out.reshape((n_segments,) + suffix_shape)


def main():
    parser = argparse.ArgumentParser(description="生成 SegmentMinCsr 单用例数据")
    parser.add_argument("--case", type=str, default=DEFAULT_CASE, help="用例名称")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--nsegments", type=int, default=DEFAULT_SEGMENTS, help="段数量")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    M = SRC_SHAPE[0]
    n_segments = args.nsegments

    src = rng.uniform(-1.0, 1.0, size=SRC_SHAPE).astype(DTYPE)
    indptr = gen_indptr(M, n_segments, rng)
    out = compute_segment_min_csr(src, indptr)

    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    case = args.case

    write_binary(input_dir / f"src_{case}.bin", src)
    write_binary(input_dir / f"indptr_{case}.bin", indptr)
    write_binary(output_dir / f"golden_{case}.bin", out)
    write_shape_file(
        input_dir / f"shape_{case}.txt",
        src.shape,
        indptr.shape,
        out.shape,
        n_segments,
        DTYPE,
    )
    print(f"src shape: {src.shape}")
    print(f"indptr: {indptr}")
    print(f"out shape: {out.shape}")
    print(f"[INFO] 用例 {case} 数据生成完毕")


if __name__ == "__main__":
    main()
