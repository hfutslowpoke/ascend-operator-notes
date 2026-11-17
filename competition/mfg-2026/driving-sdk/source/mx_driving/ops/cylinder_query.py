# Copyright 2026 Huawei Technologies Co., Ltd.
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Modification by: Huawei Developers
Modification date: 2025-09-10
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
from torch.autograd import Function
import mx_driving._C


class CylinderQuery(Function):
    # 'pylint: disable=too-many-arguments
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, new_xyz, xyz, rot):
        rot = rot.reshape(rot.shape[0], rot.shape[1], 9)
        group_idx = mx_driving._C.cylinder_query(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
        out = CylinderQuery.sortRes(group_idx, nsample)
        return out

    @classmethod
    def sortRes(cls, group_idx, nsample):
        b = group_idx.shape[0]  # noqa: F841
        m = group_idx.shape[1]  # noqa: F841
        n = group_idx.shape[2]
        sorted_idx = group_idx.sort(dim=-1)[0]
        head, _ = torch.split(sorted_idx, [nsample, sorted_idx.size(-1) - nsample], dim=-1)
        group_idx = head
        mask = group_idx >= n
        group_first = (~mask).float() * group_idx
        group_idx = torch.lerp(group_idx, group_first[..., 0:1], mask.float())

        return group_idx.to(dtype=torch.int32)


cylinder_query = CylinderQuery.apply
