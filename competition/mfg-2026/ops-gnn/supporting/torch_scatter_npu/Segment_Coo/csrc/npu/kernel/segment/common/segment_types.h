/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#include <cstdint>

namespace segment {

enum SegmentReduce : uint32_t {
    SegmentSum = 0,
    SegmentMean = 1,
    SegmentMin = 2,
    SegmentMax = 3,
};

enum SegmentDType : uint32_t {
    SegmentBFloat16 = 0,
    SegmentFloat16 = 1,
    SegmentFloat32 = 2,
    SegmentInt8 = 3,
    SegmentUInt8 = 4,
    SegmentInt16 = 5,
    SegmentInt32 = 6,
};

}  // namespace segment
