"""CPU checks for the independent manufacturing references."""

from __future__ import annotations

import numpy as np

from mfg_reference import (
    ball_query,
    cylinder_query,
    furthest_point_sample,
    radius,
    segment_min_csr,
    spmm_sum,
    spspmm,
)


def test_segment_min_csr_example_and_empty():
    src = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    out, arg = segment_min_csr(src, np.array([0, 2, 2, 3], dtype=np.int64))
    assert np.array_equal(out, np.array([[1, 2], [0, 0], [5, 6]], dtype=np.float32))
    assert np.array_equal(arg, np.array([[0, 0], [3, 3], [2, 2]], dtype=np.int64))

    src0 = np.zeros((0, 2), dtype=np.float32)
    out0, arg0 = segment_min_csr(src0, np.array([0, 0], dtype=np.int64))
    assert out0.shape == (1, 2) and np.all(out0 == 0)
    assert np.array_equal(arg0, np.array([[0, 0]], dtype=np.int64))


def test_segment_min_csr_first_match_tie():
    src = np.array([[5, 5], [5, 5], [5, 5]], dtype=np.float32)
    out, arg = segment_min_csr(src, np.array([0, 3], dtype=np.int64))
    assert np.array_equal(out, np.array([[5, 5]], dtype=np.float32))
    assert np.array_equal(arg, np.array([[0, 0]], dtype=np.int64))


def test_fps_tie_and_repeat():
    points = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]], dtype=np.float32)
    idx = furthest_point_sample(points, 3)
    assert np.array_equal(idx, np.array([[0, 1, 2]], dtype=np.int32))

    same = np.zeros((1, 4, 3), dtype=np.float32)
    idx2 = furthest_point_sample(same, 4)
    assert np.array_equal(idx2, np.zeros((1, 4), dtype=np.int32))


def test_fps_fp16_modes_are_explicit():
    points = np.array([[[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]]], dtype=np.float16)
    a = furthest_point_sample(points, 3, fp16_mode="per_dim")
    b = furthest_point_sample(points, 3, fp16_mode="fp32")
    assert a.shape == b.shape == (1, 3)


def test_radius_example_and_boundary():
    x = np.array([[0, 0], [1, 0], [5, 5]], dtype=np.float32)
    y = np.array([[0, 0]], dtype=np.float32)
    out = radius(x, y, [0, 3], [0, 1], 1.5, 2)
    assert np.array_equal(out, np.array([[0, 0], [0, 1]], dtype=np.int64))

    # d2 == r2 must hit for Radius.
    out2 = radius(np.array([[1, 0]], dtype=np.float32), np.array([[0, 0]], dtype=np.float32),
                  [0, 1], [0, 1], 1.0, 2)
    assert np.array_equal(out2, np.array([[0, -1], [0, -1]], dtype=np.int64))

    # r == 0 is defined as no hits.
    out3 = radius(np.array([[0, 0]], dtype=np.float32), np.array([[0, 0]], dtype=np.float32),
                  [0, 1], [0, 1], 0.0, 2)
    assert np.array_equal(out3, np.array([[-1, -1], [-1, -1]], dtype=np.int64))


def test_ball_query_strict_and_padding():
    xyz = np.array([[[0, 0, 0], [1, 0, 0], [3, 0, 0]]], dtype=np.float32)
    new = np.array([[[0, 0, 0]]], dtype=np.float32)
    idx = ball_query(xyz, new, 1.5, 3)
    assert np.array_equal(idx, np.array([[[0, 1, 0]]], dtype=np.int32))

    # d2 == r2 must not hit BallQuery.
    idx2 = ball_query(np.array([[[1, 0, 0]]], dtype=np.float32), new, 1.0, 2)
    assert np.array_equal(idx2, np.zeros((1, 1, 2), dtype=np.int32))


def test_cylinder_query_identity_and_padding():
    xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 0.2, 0]]], dtype=np.float32)
    new = np.array([[[0, 0, 0]]], dtype=np.float32)
    rot = np.eye(3, dtype=np.float32)[None, None, :, :]
    idx = cylinder_query(xyz, new, rot, 1.0, -0.5, 0.5, 3)
    # x in (-0.5, 0.5) and y^2+z^2 < 1: point 0 and point 2; point 1 has x=1.
    assert np.array_equal(idx, np.array([[[0, 2, 0]]], dtype=np.int32))

    # hmin/hmax are strict: x == hmax does not hit.
    idx2 = cylinder_query(np.array([[[0.5, 0, 0]]], dtype=np.float32), new, rot, 1.0, -0.5, 0.5, 2)
    assert np.array_equal(idx2, np.zeros((1, 1, 2), dtype=np.int32))


def test_spmm_sum_example_and_empty_row():
    rowptr = np.array([0, 2, 3], dtype=np.int64)
    col = np.array([0, 1, 2], dtype=np.int32)
    value = np.array([1, 2, 3], dtype=np.float32)
    mat = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    out = spmm_sum(rowptr, col, value, mat)
    assert np.array_equal(out, np.array([[7, 10], [15, 18]], dtype=np.float32))

    out2 = spmm_sum(np.array([0, 0, 3], dtype=np.int64), col, value, mat)
    assert np.array_equal(out2[0], np.zeros(2, dtype=np.float32))
    assert np.array_equal(out2[1], np.array([22, 28], dtype=np.float32))


def test_spspmm_fixed_slots_and_zero_sum():
    index_a = np.array([[0, 0], [0, 1]], dtype=np.int64)
    value_a = np.array([1, -1], dtype=np.float32)
    index_b = np.array([[0, 1], [0, 0]], dtype=np.int64)
    value_b = np.array([1, 1], dtype=np.float32)
    index_c, value_c = spspmm(index_a, value_a, index_b, value_b, 1, 1, 2)
    assert np.array_equal(index_c, np.array([[0], [0]], dtype=np.int64))
    assert value_c[0] == 0  # explicit zero must be retained

    index_c2, value_c2 = spspmm(
        np.array([[0], [0]], dtype=np.int64), np.array([1], dtype=np.float32),
        np.array([[0], [1]], dtype=np.int64), np.array([2], dtype=np.float32),
        2, 2, 1,
    )
    assert np.array_equal(index_c2, np.array([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=np.int64))
    assert np.array_equal(value_c2, np.array([2, 0, 0, 0], dtype=np.float32))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"REFERENCE TESTS: PASS ({len(tests)})")


if __name__ == "__main__":
    main()
