# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except
# in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text.

"""Official FPS task cases for Ascend 950 (arch35)."""

import os
from pathlib import Path
import runpy

import pytest
import torch


_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
NPU_READY = _GOLDEN["NPU_READY"]
ops_gnn = _GOLDEN["ops_gnn"]
fps_cpu = _GOLDEN["fps_cpu"]

pytestmark = pytest.mark.skipif(
    not NPU_READY, reason="requires an available Ascend NPU"
)

SUPPORTED_DTYPES = (torch.float16, torch.float32)
TEST_SHAPES = (
    (128, 3),
    (1024, 64),
    (4096, 128),
    (8192, 256),
    (512, 768),
    (256, 3),
    (2048, 32),
    (512, 3),
)
GENERAL_SHAPES = (
    (1, 3),
    (8, 3),
    (2, 64),
    (4, 128),
    (10000, 3),
    (16000, 128),
    (32000, 3),
)


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(42)


def _assert_exact(actual, expected):
    assert actual.device.type == "npu"
    assert actual.dtype == torch.int64
    assert actual.shape == expected.shape
    assert torch.equal(actual.cpu(), expected)


class TestFps:
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, dtype):
        src = torch.randn(shape).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        expected = fps_cpu(src, ratio=0.5, random_start=False)
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, dtype):
        src = torch.randn(shape).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        expected = fps_cpu(src, ratio=0.5, random_start=False)
        _assert_exact(actual, expected)

    def test_ratio_one(self):
        src = torch.randn(128, 3)
        actual = ops_gnn.fps(src.npu(), ratio=1.0, random_start=False)
        expected = fps_cpu(src, ratio=1.0, random_start=False)
        assert actual.shape[0] == 128
        _assert_exact(actual, expected)

    def test_ratio_small_ceil(self):
        src = torch.randn(128, 3)
        actual = ops_gnn.fps(src.npu(), ratio=0.01, random_start=False)
        expected = fps_cpu(src, ratio=0.01, random_start=False)
        assert actual.shape[0] >= 1
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_random_start(self, dtype):
        src = torch.randn(128, 3).to(dtype).npu()
        torch.manual_seed(42)
        first = ops_gnn.fps(src, ratio=0.5, random_start=True)
        torch.manual_seed(42)
        second = ops_gnn.fps(src, ratio=0.5, random_start=True)
        _assert_exact(first, second.cpu())

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_random_start_false(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        expected = fps_cpu(src, ratio=0.5, random_start=False)
        _assert_exact(actual, expected)

    def test_empty_input(self):
        src = torch.empty(0, 3, dtype=torch.float32, device="npu")
        actual = ops_gnn.fps(src, ratio=0.5, random_start=False)
        assert actual.device.type == "npu"
        assert actual.dtype == torch.int64
        assert actual.numel() == 0

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_batch_none(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        expected = fps_cpu(src, ratio=0.5, random_start=False)
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_multi_batch(self, dtype):
        src = torch.randn(256, 3).to(dtype)
        batch = torch.arange(4).repeat_interleave(64)
        actual = ops_gnn.fps(
            src.npu(), batch=batch.npu(), ratio=0.5, random_start=False
        )
        expected = fps_cpu(
            src, batch=batch, ratio=0.5, random_start=False
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_single_batch_per_node(self, dtype):
        src = torch.randn(5, 3).to(dtype)
        batch = torch.arange(5)
        actual = ops_gnn.fps(
            src.npu(), batch=batch.npu(), ratio=0.5, random_start=False
        )
        expected = fps_cpu(
            src, batch=batch, ratio=0.5, random_start=False
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_large(self, dtype):
        src = torch.randn(10000, 3).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.3, random_start=False)
        expected = fps_cpu(src, ratio=0.3, random_start=False)
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "ptr",
        ([0, 128], [0, 64, 128], [0, 32, 96, 128]),
    )
    def test_ptr_path(self, dtype, ptr):
        src = torch.randn(128, 3).to(dtype)
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=False, ptr=ptr
        )
        expected = fps_cpu(
            src, ratio=0.5, random_start=False, ptr=ptr
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_ptr_path_tensor(self, dtype):
        src = torch.randn(256, 64).to(dtype)
        ptr = torch.tensor([0, 100, 200, 256])
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=False, ptr=ptr.npu()
        )
        expected = fps_cpu(
            src, ratio=0.5, random_start=False, ptr=ptr
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_ptr_path_nonuniform(self, dtype):
        src = torch.randn(120, 3).to(dtype)
        ptr = [0, 40, 100, 120]
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=False, ptr=ptr
        )
        expected = fps_cpu(
            src, ratio=0.5, random_start=False, ptr=ptr
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_per_batch_ratio_tensor(self, dtype):
        src = torch.randn(256, 3).to(dtype)
        batch = torch.arange(4).repeat_interleave(64)
        ratio = torch.tensor([0.1, 0.5, 0.9, 1.0])
        actual = ops_gnn.fps(
            src.npu(), ratio=ratio.npu(), batch=batch.npu(),
            random_start=False
        )
        expected = fps_cpu(
            src, batch=batch, ratio=ratio, random_start=False
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_per_batch_ratio_ptr(self, dtype):
        src = torch.randn(200, 3).to(dtype)
        ratio = torch.tensor([0.2, 0.8])
        ptr = [0, 150, 200]
        actual = ops_gnn.fps(
            src.npu(), ratio=ratio.npu(), random_start=False, ptr=ptr
        )
        expected = fps_cpu(
            src, ratio=ratio, random_start=False, ptr=ptr
        )
        _assert_exact(actual, expected)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_per_batch_ratio_single_batch(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        ratio = torch.tensor([0.25])
        actual = ops_gnn.fps(
            src.npu(), ratio=ratio.npu(), random_start=False
        )
        expected = fps_cpu(src, ratio=ratio, random_start=False)
        _assert_exact(actual, expected)
