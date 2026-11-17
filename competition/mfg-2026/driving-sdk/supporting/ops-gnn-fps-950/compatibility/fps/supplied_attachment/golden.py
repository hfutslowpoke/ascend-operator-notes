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

"""Independent CPU golden implementation for farthest point sampling."""

import os

import torch


def _npu_ready():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    try:
        torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
        return bool(torch.npu.is_available())
    except RuntimeError:
        return False


NPU_READY = _npu_ready()

if NPU_READY:
    import ops_gnn
else:
    ops_gnn = None


def fps_single(src, ratio, random_start):
    """Return one segment's FPS indices using the task's exact arithmetic."""
    num_points = src.size(0)
    if num_points == 0:
        return torch.empty(0, dtype=torch.long)

    sample_count = max(1, int(torch.ceil(torch.tensor(ratio * num_points)).item()))
    output = torch.empty(sample_count, dtype=torch.long)
    src_float = src.float()
    distance = torch.full((num_points,), 5e4, dtype=src.dtype)
    if random_start:
        generator = torch.Generator()
        generator.manual_seed(42)
        farthest = int(
            torch.randint(0, num_points, (1,), generator=generator).item()
        )
    else:
        farthest = 0

    for sample in range(sample_count):
        output[sample] = farthest
        center = src_float[farthest]
        if src.dtype == torch.float16:
            current = torch.zeros(num_points, dtype=torch.float16)
            for feature in range(src.size(1)):
                delta = (src_float[:, feature] - center[feature]).half()
                square = (delta.float() * delta.float()).half()
                current = (current.float() + square.float()).half()
        else:
            current = torch.zeros(num_points, dtype=torch.float32)
            for feature in range(src.size(1)):
                delta = src_float[:, feature] - center[feature]
                current = torch.addcmul(current, delta, delta)
        distance = torch.minimum(distance, current)

        # The supplied task golden models torch_cluster's 256-lane CUDA
        # reduction: choose the lowest lane (index % 256), then the first
        # point encountered by that lane.
        candidates = (distance == distance.max()).nonzero().flatten()
        farthest = int(candidates[torch.argmin(candidates % 256)].item())
    return output


def fps_cpu(src, batch=None, ratio=0.5, random_start=True, ptr=None):
    """CPU reference used for bit-wise index comparison in NPU tests."""
    if src.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    src_cpu = src.cpu().reshape(src.size(0), -1).contiguous()
    if isinstance(ratio, torch.Tensor):
        ratio = ratio.cpu()

    if ptr is not None:
        ptr_cpu = ptr.cpu() if isinstance(ptr, torch.Tensor) else torch.tensor(
            ptr, dtype=torch.long
        )
        outputs = []
        for batch_id in range(ptr_cpu.numel() - 1):
            start = int(ptr_cpu[batch_id])
            end = int(ptr_cpu[batch_id + 1])
            segment_ratio = (
                ratio[batch_id].item()
                if isinstance(ratio, torch.Tensor)
                else ratio
            )
            outputs.append(
                fps_single(src_cpu[start:end], segment_ratio, random_start)
                + start
            )
        return torch.cat(outputs, dim=0)

    if batch is None:
        batch = torch.zeros(src.size(0), dtype=torch.long)
    else:
        batch = batch.cpu()
    batch_count = int(batch.max().item()) + 1
    outputs = []
    for batch_id in range(batch_count):
        mask = batch == batch_id
        segment_ratio = (
            ratio[batch_id].item()
            if isinstance(ratio, torch.Tensor)
            else ratio
        )
        local = fps_single(src_cpu[mask], segment_ratio, random_start)
        global_indices = mask.nonzero(as_tuple=False).flatten().long()
        outputs.append(global_indices[local])
    return torch.cat(outputs, dim=0)
