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

// Host/device 共享的 segment tiling 字段。
// 结构按 32 字节块对齐声明，字段全部为定宽整数，host 侧按值打包传入 kernel。
struct SegmentTilingData {
    // 输入规模
    uint64_t srcNumel;       // src 总元素数
    uint64_t outNumel;       // out 总元素数
    uint64_t E_1;            // 归约维之前 batch 前缀积（index/indptr 前缀维乘积上限）
    uint64_t M;              // 每个 batch 归约维长度（src 行数）
    uint64_t K;              // 归约维之后的特征元素数（src.size(-1) 及后续维乘积）
    uint64_t nSegments;      // 段数
    uint64_t indexNumel;     // index/indptr 元素数
    uint32_t reduceCode;     // segment::SegmentReduce
    uint32_t dtypeCode;      // segment::SegmentDType
    uint32_t usedBlockNum;   // 实际使用的 AIV 数
    uint32_t hasInitial;     // 是否提供 out 初值（原地语义）
    uint32_t alignElems;     // dtype 对齐块元素数（32B 对齐）
    uint32_t tileRows;       // 每次搬运的行数
    uint32_t kBlocks;        // K / alignElems（K 为对齐块整数倍时 >0）
    uint32_t tailElems;      // K 尾部非对齐元素数
    uint32_t hasArg;         // min/max 是否输出 argOut
    uint32_t indexMode;      // 0: CSR indptr；1: 有序 COO index
    uint32_t extremaMode;    // 非对齐 fp32 极值专用路径
    uint64_t totalTasks;     // E_1 * nSegments 总段任务数
};
