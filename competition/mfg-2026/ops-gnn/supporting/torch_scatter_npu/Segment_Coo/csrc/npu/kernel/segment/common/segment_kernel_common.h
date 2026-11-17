/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#include "kernel_operator.h"
#include "segment/common/segment_tiling_fields.h"
#include "segment/common/segment_types.h"

using namespace AscendC;

namespace segment {

// ---------------------------------------------------------------------------
// dtype 策略（dav-2201 实测 API 支持矩阵）：
//   Add:      float/half/int32/int16 ✓
//   Max/Min:  float/int32 ✓（half/bf16 ✗）
//   Compare:  float/int32/half ✓（int16/bf16 ✗）
//   Select:   仅 float 源（arg 用 float 域）
//   Duplicate: float/int32 ✓（int64 ✗）
//   Cast:     f32->i64 / half->f32 / i32->i16 / f32->half / f32->bf16 ✓
// 由此：
//   sum/mean 累积域  half 用 half（逐行舍入，对齐 golden）、bf16 用 float、int16/int32 用原 dtype
//                    int8/uint8 用 int16 累积；16 位回绕保留目标 8 位低位，写回后再做整数 Mean
//   min/max 归约域    ExtAcc = fp 一律 float、int32 用 int32、int16 升 int32
//   arg_out 行号在 float 域累积（Select 限制），最终 Cast f32->i64 写出
// ---------------------------------------------------------------------------

template <typename T>
struct SegmentTraits;

template <>
struct SegmentTraits<float> {
    using SumAcc = float;
    using ExtAcc = float;
    __aicore__ static inline float SumInit() { return 0.0f; }
    __aicore__ static inline float MinInit() { return -3.402823466e+38F; }
    __aicore__ static inline float MaxInit() { return 3.402823466e+38F; }
};

template <>
struct SegmentTraits<half> {
    using SumAcc = half;
    using ExtAcc = float;
    __aicore__ static inline half SumInit() { return static_cast<half>(0.0f); }
    __aicore__ static inline float MinInit() { return -65504.0F; }
    __aicore__ static inline float MaxInit() { return 65504.0F; }
};

template <>
struct SegmentTraits<bfloat16_t> {
    using SumAcc = float;
    using ExtAcc = float;
    __aicore__ static inline float SumInit() { return 0.0f; }
    __aicore__ static inline float MinInit() { return -3.38953139e+38F; }
    __aicore__ static inline float MaxInit() { return 3.38953139e+38F; }
};

template <>
struct SegmentTraits<int8_t> {
    using SumAcc = int16_t;
    using ExtAcc = half;
    __aicore__ static inline int16_t SumInit() { return 0; }
    __aicore__ static inline half MinInit() { return static_cast<half>(-128.0f); }
    __aicore__ static inline half MaxInit() { return static_cast<half>(127.0f); }
};

template <>
struct SegmentTraits<uint8_t> {
    using SumAcc = int16_t;
    using ExtAcc = half;
    __aicore__ static inline int16_t SumInit() { return 0; }
    __aicore__ static inline half MinInit() { return static_cast<half>(0.0f); }
    __aicore__ static inline half MaxInit() { return static_cast<half>(255.0f); }
};

template <>
struct SegmentTraits<int32_t> {
    using SumAcc = int32_t;
    using ExtAcc = int32_t;
    __aicore__ static inline int32_t SumInit() { return 0; }
    __aicore__ static inline int32_t MinInit() { return INT32_MIN; }
    __aicore__ static inline int32_t MaxInit() { return INT32_MAX; }
};

template <>
struct SegmentTraits<int16_t> {
    // extrema 在 float 域（int16 值域 float 精确；int 间 Cast 与其他实例共存会触发重载冲突）
    using SumAcc = int16_t;
    using ExtAcc = float;
    __aicore__ static inline int16_t SumInit() { return 0; }
    __aicore__ static inline float MinInit() { return -32768.0F; }
    __aicore__ static inline float MaxInit() { return 32768.0F; }
};

template <typename T, SegmentReduce Reduce, bool HalfExtrema128 = false>
class SegmentReduceDevice {
public:
    using SumAcc = typename SegmentTraits<T>::SumAcc;
    using ExtAcc = typename SegmentTraits<T>::ExtAcc;
    using DefaultAcc = typename std::conditional<Reduce == SegmentReduce::SegmentMin ||
                                                     Reduce == SegmentReduce::SegmentMax,
                                                 ExtAcc, SumAcc>::type;
    using WorkAcc = typename std::conditional<HalfExtrema128, half, DefaultAcc>::type;

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indptr, GM_ADDR out, GM_ADDR argOut,
                                const SegmentTilingData& tiling)
    {
        tiling_ = &tiling;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), tiling.srcNumel);
        indptrGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indptr), tiling.indexNumel);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), tiling.outNumel);
        if (tiling.hasArg != 0) {
            argGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(argOut), tiling.outNumel);
        }

        const uint32_t kElems = tiling.kBlocks > 0
            ? static_cast<uint32_t>(tiling.K)
            : static_cast<uint32_t>(
                ((tiling.K + tiling.alignElems - 1) / tiling.alignElems) * tiling.alignElems);
        const uint64_t srcTileElems = static_cast<uint64_t>(tiling.tileRows) * kElems;

        eventIdMte2ToV_ = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        eventIdVToMte2_ = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3_ = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToV_ = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());

        pipe.InitBuffer(srcBuf, srcTileElems * sizeof(T));
        pipe.InitBuffer(accBuf, kElems * 8 * sizeof(WorkAcc));
        pipe.InitBuffer(rowAccBuf, kElems * sizeof(WorkAcc));
        pipe.InitBuffer(outCastBuf, kElems * sizeof(T));
        pipe.InitBuffer(argFBuf, kElems * 8 * sizeof(float));
        pipe.InitBuffer(rowFBuf, kElems * sizeof(float));
        pipe.InitBuffer(arg64Buf, kElems * 8 * sizeof(int64_t));
        pipe.InitBuffer(zeroFBuf, kElems * sizeof(float));
        pipe.InitBuffer(cmpBuf, kElems * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        if (tiling_->hasInitial != 0) {
            const uint64_t blockId = GetBlockIdx();
            const uint64_t blocks = tiling_->usedBlockNum;
            const uint64_t total = tiling_->totalTasks;
            const uint64_t taskBeg = blockId * total / blocks;
            const uint64_t taskEnd = (blockId + 1) * total / blocks;
            for (uint64_t task = taskBeg; task < taskEnd; ++task) {
                ProcessTask(task);
            }
            return;
        }
        if constexpr (!std::is_same<T, bfloat16_t>::value &&
                      !std::is_same<T, int8_t>::value &&
                      !std::is_same<T, uint8_t>::value) {
            if (tiling_->kBlocks == 0) {
                if constexpr (std::is_same<T, float>::value &&
                              (Reduce == SegmentReduce::SegmentSum ||
                               Reduce == SegmentReduce::SegmentMean)) {
                    ProcessPaddedFloatSumMean();
                    return;
                }
                if constexpr (std::is_same<T, float>::value &&
                    (Reduce == SegmentReduce::SegmentMin ||
                               Reduce == SegmentReduce::SegmentMax)) {
                    if (tiling_->extremaMode == 1) {
                        ProcessPaddedFloatExtremaGroupedSmallK();
                        return;
                    }
                    ProcessPaddedFloatExtrema();
                    return;
                }
                ProcessScalar();
                return;
            }
            if constexpr (std::is_same<T, float>::value &&
                          (Reduce == SegmentReduce::SegmentSum ||
                           Reduce == SegmentReduce::SegmentMean)) {
                if (tiling_->K == 128 && tiling_->tileRows >= 128) {
                    ProcessFloatSumMean128Grouped();
                    return;
                }
            }
        }
        const uint64_t blockId = GetBlockIdx();
        const uint64_t blocks = tiling_->usedBlockNum;
        const uint64_t total = tiling_->totalTasks;
        const uint64_t taskBeg = blockId * total / blocks;
        const uint64_t taskEnd = (blockId + 1) * total / blocks;
        for (uint64_t task = taskBeg; task < taskEnd; ++task) {
            ProcessTask(task);
        }
    }

    __aicore__ inline void ProcessPaddedFloatExtremaGroupedSmallK()
    {
        const uint64_t blockId = GetBlockIdx();
        const uint64_t blocks = tiling_->usedBlockNum;
        const uint32_t kElems = static_cast<uint32_t>(tiling_->K);
        constexpr uint32_t paddedK = 16;
        constexpr uint64_t groupTasks = 8;
        const uint64_t totalGroups = (tiling_->totalTasks + groupTasks - 1) / groupTasks;
        const uint64_t groupBeg = blockId * totalGroups / blocks;
        const uint64_t groupEnd = (blockId + 1) * totalGroups / blocks;
        LocalTensor<float> srcUb = srcBuf.Get<float>();
        LocalTensor<float> accUb = accBuf.Get<float>();
        LocalTensor<float> argFUb = argFBuf.Get<float>();
        LocalTensor<int64_t> arg64Ub = arg64Buf.Get<int64_t>();
        LocalTensor<uint8_t> cmpUb = cmpBuf.Get<uint8_t>();
        const DataCopyPadParams padParams(
            false, 0, static_cast<uint8_t>(paddedK - kElems), 0);
        const float neutral = Reduce == SegmentReduce::SegmentMin
            ? SegmentTraits<float>::MaxInit()
            : SegmentTraits<float>::MinInit();

        for (uint64_t group = groupBeg; group < groupEnd; ++group) {
            const uint64_t taskBeg = group * groupTasks;
            const uint64_t taskEnd = taskBeg + groupTasks <= tiling_->totalTasks
                ? taskBeg + groupTasks
                : tiling_->totalTasks;
            const uint64_t firstE = taskBeg / tiling_->nSegments;
            uint64_t firstR0 = 0;
            uint64_t firstR1 = 0;
            SegmentBounds(firstE, taskBeg % tiling_->nSegments, firstR0, firstR1);
            const uint64_t lastTask = taskEnd - 1;
            const uint64_t lastE = lastTask / tiling_->nSegments;
            uint64_t lastR0 = 0;
            uint64_t lastR1 = 0;
            SegmentBounds(lastE, lastTask % tiling_->nSegments, lastR0, lastR1);
            if (firstE != lastE || lastR1 < firstR0 || lastR1 - firstR0 > tiling_->tileRows) {
                ProcessPaddedFloatExtrema();
                return;
            }
            const uint64_t totalRows = lastR1 - firstR0;
            if (totalRows != 0) {
                Duplicate(srcUb[0], neutral, static_cast<uint32_t>(totalRows * paddedK));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                DataCopyPad(srcUb[0], srcGm[(tiling_->M * firstE + firstR0) * kElems],
                            {static_cast<uint16_t>(totalRows),
                             static_cast<uint16_t>(kElems * sizeof(float)), 0, 0},
                            padParams);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
            }
            for (uint64_t task = taskBeg; task < taskEnd; ++task) {
                const uint64_t seg = task % tiling_->nSegments;
                uint64_t r0 = 0;
                uint64_t r1 = 0;
                SegmentBounds(firstE, seg, r0, r1);
                const uint64_t rows = r1 > r0 ? r1 - r0 : 0;
                const uint64_t outOffset = task * kElems;
                if (rows == 0) {
                    Duplicate(accUb[0], 0.0f, paddedK);
                    if (tiling_->hasArg != 0) {
                        Duplicate(argFUb[0], static_cast<float>(static_cast<int32_t>(tiling_->M)), paddedK);
                    }
                } else {
                    Duplicate(accUb[0], neutral, paddedK);
                    if (tiling_->hasArg != 0) {
                        Duplicate(argFUb[0], -1.0f, paddedK);
                    }
                    for (uint64_t row = 0; row < rows; ++row) {
                        const int32_t rowIdx = static_cast<int32_t>(r0 + row);
                        ReduceRow<false>(accUb, srcUb[(r0 - firstR0 + row) * paddedK], cmpUb,
                                         argFUb, rowIdx, paddedK);
                    }
                }
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                DataCopyPad(outGm[outOffset], accUb,
                            {1, static_cast<uint32_t>(kElems * sizeof(float)), 0, 0, 0});
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                if (tiling_->hasArg != 0) {
                    CastArgToInt64(arg64Ub, argFUb, paddedK);
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                    DataCopyPad(argGm[outOffset], arg64Ub,
                                {1, static_cast<uint32_t>(kElems * sizeof(int64_t)), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                }
            }
            if (totalRows != 0) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
            }
        }
    }

private:
    __aicore__ inline uint64_t GreatestCommonDivisor(uint64_t lhs, uint64_t rhs) const
    {
        while (rhs != 0) {
            const uint64_t remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        return lhs;
    }

    __aicore__ inline uint64_t IndptrBase(uint64_t e) const
    {
        return e * (tiling_->nSegments + 1);
    }

    __aicore__ inline uint64_t LowerBound(uint64_t e, int64_t target) const
    {
        const uint64_t base = e * tiling_->M;
        uint64_t left = 0;
        uint64_t right = tiling_->M;
        while (left < right) {
            const uint64_t middle = left + (right - left) / 2;
            if (indptrGm.GetValue(base + middle) < target) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        return left;
    }

    __aicore__ inline void SegmentBounds(uint64_t e, uint64_t seg,
                                         uint64_t& r0, uint64_t& r1) const
    {
        if (tiling_->indexMode == 0) {
            const uint64_t base = IndptrBase(e);
            r0 = static_cast<uint64_t>(indptrGm.GetValue(base + seg));
            r1 = static_cast<uint64_t>(indptrGm.GetValue(base + seg + 1));
            return;
        }

        if (tiling_->nSegments != 0 && tiling_->M % tiling_->nSegments == 0) {
            const uint64_t rowsPerSegment = tiling_->M / tiling_->nSegments;
            const uint64_t candidate0 = seg * rowsPerSegment;
            const uint64_t candidate1 = candidate0 + rowsPerSegment;
            const uint64_t base = e * tiling_->M;
            const bool leftBoundary = candidate0 == 0 ||
                indptrGm.GetValue(base + candidate0 - 1) < static_cast<int64_t>(seg);
            const bool rightBoundary = candidate1 == tiling_->M ||
                indptrGm.GetValue(base + candidate1) > static_cast<int64_t>(seg);
            if (rowsPerSegment != 0 && leftBoundary && rightBoundary &&
                indptrGm.GetValue(base + candidate0) == static_cast<int64_t>(seg) &&
                indptrGm.GetValue(base + candidate1 - 1) == static_cast<int64_t>(seg)) {
                r0 = candidate0;
                r1 = candidate1;
                return;
            }
        }

        r0 = LowerBound(e, static_cast<int64_t>(seg));
        r1 = LowerBound(e, static_cast<int64_t>(seg + 1));
    }

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        const uint64_t nSeg = tiling_->nSegments;
        const uint64_t e = task / nSeg;
        const uint64_t seg = task % nSeg;
        uint64_t r0 = 0;
        uint64_t r1 = 0;
        SegmentBounds(e, seg, r0, r1);
        const uint64_t rows = r1 > r0 ? r1 - r0 : 0;

        const uint32_t kElems = static_cast<uint32_t>(tiling_->K);
        if (rows == 0) {
            WriteEmptySegment(e, seg, kElems);
            return;
        }

        const uint64_t srcOffset = (tiling_->M * e + r0) * kElems;
        const uint64_t outOffset = (nSeg * e + seg) * kElems;

        LocalTensor<WorkAcc> accUb = accBuf.Get<WorkAcc>();
        LocalTensor<T> srcUb = srcBuf.Get<T>();
        LocalTensor<WorkAcc> rowAccUb = rowAccBuf.Get<WorkAcc>();
        LocalTensor<uint8_t> cmpUb = cmpBuf.Get<uint8_t>();
        LocalTensor<float> argFUb = argFBuf.Get<float>();
        LocalTensor<float> rowFUb = rowFBuf.Get<float>();

        if constexpr ((Reduce == SegmentReduce::SegmentMin ||
                       Reduce == SegmentReduce::SegmentMax) &&
                      (HalfExtrema128 || std::is_same<T, float>::value)) {
            if (tiling_->hasInitial == 0 && kElems == 128 && rows <= tiling_->tileRows) {
                ProcessExtrema128Ra(srcOffset, outOffset, r0, rows, kElems,
                                    accUb, srcUb, cmpUb, argFUb);
                return;
            }
        }

        if constexpr ((Reduce == SegmentReduce::SegmentSum ||
                       Reduce == SegmentReduce::SegmentMean) &&
                      std::is_same<T, float>::value) {
            if (tiling_->hasInitial == 0 && rows <= tiling_->tileRows) {
                DataCopy(srcUb, srcGm[srcOffset], rows * kElems);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                uint32_t srcShape[2] = {static_cast<uint32_t>(rows), kElems};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    accUb, srcUb, srcShape, false);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                FinalizeSegment(accUb, kElems, rows);
                CopyOut(accUb, argFUb, outOffset, kElems, rows);
                return;
            }
        }

        if constexpr ((Reduce == SegmentReduce::SegmentSum ||
                       Reduce == SegmentReduce::SegmentMean) &&
                      std::is_same<T, half>::value) {
            if (tiling_->hasInitial == 0 && kElems == 128 && rows <= tiling_->tileRows &&
                (rows & (rows - 1)) == 0) {
                ProcessHalfSumMeanTree(srcOffset, outOffset, rows, kElems,
                                       accUb, srcUb, argFUb);
                return;
            }
        }

        if (tiling_->hasInitial != 0) {
            LocalTensor<T> initialUb = outCastBuf.Get<T>();
            DataCopy(initialUb, outGm[outOffset], kElems);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
            if constexpr (std::is_same<WorkAcc, T>::value) {
                Adds(accUb[0], initialUb[0], static_cast<T>(0), kElems);
            } else if constexpr ((std::is_same<T, int8_t>::value ||
                                  std::is_same<T, uint8_t>::value) &&
                                 std::is_same<WorkAcc, int16_t>::value) {
                LocalTensor<half> initialHalfUb = rowFBuf.Get<half>();
                Cast(initialHalfUb[0], initialUb[0], RoundMode::CAST_NONE, kElems);
                PipeBarrier<PIPE_V>();
                Cast(accUb[0], initialHalfUb[0], RoundMode::CAST_RINT, kElems);
            } else {
                Cast(accUb[0], initialUb[0], RoundMode::CAST_NONE, kElems);
            }
            if (tiling_->hasArg != 0) {
                Duplicate(argFUb[0], static_cast<float>(static_cast<int32_t>(tiling_->M)), kElems);
            }
            PipeBarrier<PIPE_V>();
        } else if constexpr (!HalfExtrema128) {
            if constexpr (Reduce == SegmentReduce::SegmentMin) {
                Duplicate(accUb[0], static_cast<WorkAcc>(SegmentTraits<T>::MaxInit()), kElems);
            } else if constexpr (Reduce == SegmentReduce::SegmentMax) {
                Duplicate(accUb[0], static_cast<WorkAcc>(SegmentTraits<T>::MinInit()), kElems);
            } else {
                Duplicate(accUb[0], static_cast<WorkAcc>(SegmentTraits<T>::SumInit()), kElems);
            }
            if (tiling_->hasArg != 0) {
                Duplicate(argFUb[0], -1.0f, kElems);
            }
            PipeBarrier<PIPE_V>();
        }

        const uint64_t rowSpan = tiling_->tileRows;
        for (uint64_t rowBase = 0; rowBase < rows; rowBase += rowSpan) {
            const uint64_t n = rowBase + rowSpan <= rows ? rowSpan : rows - rowBase;
            DataCopy(srcUb, srcGm[srcOffset + rowBase * kElems], n * kElems);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);

            uint64_t firstReduceRow = 0;
            if constexpr (HalfExtrema128) {
                if (rowBase == 0) {
                    const LocalTensor<T> firstRow = srcUb[0];
                    Adds(accUb[0], firstRow[0], static_cast<half>(0.0f), kElems);
                    if (tiling_->hasArg != 0) {
                        Duplicate(argFUb[0], static_cast<float>(static_cast<int32_t>(r0)), kElems);
                    }
                    PipeBarrier<PIPE_V>();
                    firstReduceRow = 1;
                }
            }

            for (uint64_t i = firstReduceRow; i < n; ++i) {
                const int32_t rowIdx = static_cast<int32_t>(r0 + rowBase + i);
                const LocalTensor<T> rowView = srcUb[static_cast<uint64_t>(i) * kElems];
                if constexpr (!std::is_same<WorkAcc, T>::value) {
                    if constexpr ((std::is_same<T, int8_t>::value ||
                                   std::is_same<T, uint8_t>::value) &&
                                  std::is_same<WorkAcc, int16_t>::value) {
                        LocalTensor<half> rowHalfUb = rowFBuf.Get<half>();
                        Cast(rowHalfUb[0], rowView[0], RoundMode::CAST_NONE, kElems);
                        PipeBarrier<PIPE_V>();
                        Cast(rowAccUb[0], rowHalfUb[0], RoundMode::CAST_RINT, kElems);
                    } else {
                        Cast(rowAccUb[0], rowView[0], RoundMode::CAST_NONE, kElems);
                    }
                    PipeBarrier<PIPE_V>();
                    ReduceRow<true>(accUb, rowAccUb, cmpUb, argFUb, rowIdx, kElems);
                } else {
                    ReduceRow<false>(accUb, rowView, cmpUb, argFUb, rowIdx, kElems);
                }
            }
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        }

        FinalizeSegment(accUb, kElems, rows);
        CopyOut(accUb, argFUb, outOffset, kElems, rows);
    }

    __aicore__ inline void ProcessExtrema128Ra(
        uint64_t srcOffset, uint64_t outOffset, uint64_t r0, uint64_t rows,
        uint32_t kElems, LocalTensor<WorkAcc> accUb, LocalTensor<T> srcUb,
        LocalTensor<uint8_t> cmpUb, LocalTensor<float> argFUb)
    {
        DataCopy(srcUb, srcGm[srcOffset], rows * kElems);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);

        ProcessExtrema128RaFromUb(srcUb, outOffset, r0, rows, kElems,
                                  accUb, cmpUb, argFUb);
    }

    __aicore__ inline void ProcessExtrema128RaComputeFromUb(
        LocalTensor<T> srcUb, uint64_t r0, uint64_t rows, uint32_t kElems,
        LocalTensor<WorkAcc> accUb, LocalTensor<uint8_t> cmpUb,
        LocalTensor<float> argFUb)
    {
        uint32_t srcShape[2] = {static_cast<uint32_t>(rows), kElems};
        if constexpr (Reduce == SegmentReduce::SegmentMin) {
            ReduceMin<T, AscendC::Pattern::Reduce::RA, false>(
                accUb, srcUb, srcShape, false);
        } else {
            ReduceMax<T, AscendC::Pattern::Reduce::RA, false>(
                accUb, srcUb, srcShape, false);
        }
        PipeBarrier<PIPE_V>();

        if (tiling_->hasArg == 0) {
            return;
        }

        Duplicate(argFUb[0], -1.0f, kElems);
        PipeBarrier<PIPE_V>();
        constexpr uint32_t argChunk = 64;
        constexpr uint8_t dataRepeatStride =
            static_cast<uint8_t>(argChunk * sizeof(T) / 32);
        for (uint64_t row = 0; row < rows; ++row) {
            const LocalTensor<T> rowView = srcUb[row * kElems];
            Compare(cmpUb[0], accUb[0], rowView[0], CMPMODE::NE,
                    argChunk, 2, {1, 1, 1, 1, dataRepeatStride, dataRepeatStride});
            Select(argFUb[0], cmpUb[0], argFUb[0],
                   static_cast<float>(static_cast<int32_t>(r0 + row)),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   argChunk, 2, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessExtrema128RaFromUb(
        LocalTensor<T> srcUb, uint64_t outOffset, uint64_t r0, uint64_t rows,
        uint32_t kElems, LocalTensor<WorkAcc> accUb,
        LocalTensor<uint8_t> cmpUb, LocalTensor<float> argFUb)
    {
        ProcessExtrema128RaComputeFromUb(srcUb, r0, rows, kElems,
                                         accUb, cmpUb, argFUb);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        CopyOut(accUb, argFUb, outOffset, kElems, rows);
    }

    __aicore__ inline void ProcessFloatSumMean128Grouped()
    {
        const uint64_t blockId = GetBlockIdx();
        const uint64_t blocks = tiling_->usedBlockNum;
        constexpr uint64_t groupTasks = 8;
        constexpr uint32_t kElems = 128;
        const uint64_t totalGroups = (tiling_->totalTasks + groupTasks - 1) / groupTasks;
        const uint64_t groupBeg = blockId * totalGroups / blocks;
        const uint64_t groupEnd = (blockId + 1) * totalGroups / blocks;
        LocalTensor<float> srcUb = srcBuf.Get<float>();
        LocalTensor<float> accUb = accBuf.Get<float>();

        for (uint64_t group = groupBeg; group < groupEnd; ++group) {
            const uint64_t taskBeg = group * groupTasks;
            const uint64_t taskEnd = taskBeg + groupTasks <= tiling_->totalTasks
                ? taskBeg + groupTasks
                : tiling_->totalTasks;
            const uint64_t firstE = taskBeg / tiling_->nSegments;
            uint64_t firstR0 = 0;
            uint64_t firstR1 = 0;
            SegmentBounds(firstE, taskBeg % tiling_->nSegments, firstR0, firstR1);
            const uint64_t lastTask = taskEnd - 1;
            const uint64_t lastE = lastTask / tiling_->nSegments;
            uint64_t lastR0 = 0;
            uint64_t lastR1 = 0;
            SegmentBounds(lastE, lastTask % tiling_->nSegments, lastR0, lastR1);
            if (firstE != lastE || lastR1 < firstR0 || lastR1 - firstR0 > tiling_->tileRows) {
                for (uint64_t task = taskBeg; task < taskEnd; ++task) {
                    ProcessTask(task);
                }
                continue;
            }

            const uint64_t totalRows = lastR1 - firstR0;
            if (totalRows != 0) {
                DataCopy(srcUb, srcGm[(tiling_->M * firstE + firstR0) * kElems],
                         totalRows * kElems);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
            }

            for (uint64_t task = taskBeg; task < taskEnd; ++task) {
                const uint64_t seg = task % tiling_->nSegments;
                uint64_t r0 = 0;
                uint64_t r1 = 0;
                SegmentBounds(firstE, seg, r0, r1);
                const uint64_t rows = r1 > r0 ? r1 - r0 : 0;
                const uint64_t slot = task - taskBeg;
                LocalTensor<float> slotAcc = accUb[slot * kElems];
                if (rows == 0) {
                    Duplicate(slotAcc[0], 0.0f, kElems);
                    PipeBarrier<PIPE_V>();
                    continue;
                }

                uint32_t srcShape[2] = {static_cast<uint32_t>(rows), kElems};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    slotAcc, srcUb[(r0 - firstR0) * kElems], srcShape, false);
                PipeBarrier<PIPE_V>();
                FinalizeSegment(slotAcc, kElems, rows);
            }

            const uint64_t taskCount = taskEnd - taskBeg;
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(outGm[taskBeg * kElems], accUb, taskCount * kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        }
    }

    __aicore__ inline void ProcessHalfSumMeanTree(
        uint64_t srcOffset, uint64_t outOffset, uint64_t rows, uint32_t kElems,
        LocalTensor<WorkAcc> accUb, LocalTensor<T> srcUb, LocalTensor<float> argFUb)
    {
        DataCopy(srcUb, srcGm[srcOffset], rows * kElems);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);

        uint64_t activeRows = rows;
        while (activeRows > 1) {
            const uint64_t halfRows = activeRows / 2;
            const uint64_t halfElems = halfRows * kElems;
            Add(srcUb[0], srcUb[0], srcUb[halfElems], halfElems);
            PipeBarrier<PIPE_V>();
            activeRows = halfRows;
        }
        Adds(accUb[0], srcUb[0], static_cast<half>(0.0f), kElems);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        FinalizeSegment(accUb, kElems, rows);
        CopyOut(accUb, argFUb, outOffset, kElems, rows);
    }

    // 行归约（模板参数 Casted 标记 rowSrc 已处于 WorkAcc 域）
    template <bool Casted, typename RowTensor>
    __aicore__ inline void ReduceRow(LocalTensor<WorkAcc> accUb, RowTensor rowSrc,
                                     LocalTensor<uint8_t> cmpUb, LocalTensor<float> argFUb,
                                     int32_t rowIdx, uint32_t kElems)
    {
        if constexpr (Reduce == SegmentReduce::SegmentSum ||
                      Reduce == SegmentReduce::SegmentMean) {
            if constexpr (Casted) {
                Add(accUb[0], accUb[0], rowSrc[0], kElems);
            } else {
                Add(accUb[0], accUb[0], rowSrc[0], kElems);
            }
            PipeBarrier<PIPE_V>();
        } else if constexpr (Reduce == SegmentReduce::SegmentMax) {
            constexpr uint32_t argChunk = 64;
            if constexpr (std::is_same<WorkAcc, int32_t>::value) {
                Max(accUb[0], accUb[0], rowSrc[0], kElems);
                PipeBarrier<PIPE_V>();
                if (tiling_->hasArg != 0) {
                    LocalTensor<float> rowFUb = rowFBuf.Get<float>();
                    Duplicate(rowFUb[0], static_cast<float>(rowIdx), kElems);
                    PipeBarrier<PIPE_V>();
                    if (kElems == 128) {
                        Compare(cmpUb[0], accUb[0], rowSrc[0], CMPMODE::EQ,
                                argChunk, 2, {1, 1, 1, 1, 8, 8});
                        PipeBarrier<PIPE_V>();
                        Select(argFUb[0], cmpUb[0], rowFUb[0], argFUb[0],
                               SELMODE::VSEL_TENSOR_TENSOR_MODE,
                               argChunk, 2, {1, 1, 1, 8, 8, 8});
                    } else {
                        for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                            const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                            Compare(cmpUb[offset], accUb[offset], rowSrc[offset], CMPMODE::EQ,
                                    count, 1, {1, 1, 1, 8, 8, 8});
                            PipeBarrier<PIPE_V>();
                            Select(argFUb[offset], cmpUb[offset], rowFUb[offset], argFUb[offset],
                                   SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                   count, 1, {1, 1, 1, 8, 8, 8});
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                return;
            }
            if (kElems == 128) {
                constexpr uint8_t dataRepeatStride =
                    static_cast<uint8_t>(argChunk * sizeof(WorkAcc) / 32);
                Compare(cmpUb[0], accUb[0], rowSrc[0], CMPMODE::GT,
                        argChunk, 2, {1, 1, 1, 1, dataRepeatStride, dataRepeatStride});
            } else {
                for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                    const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                    Compare(cmpUb[offset], accUb[offset], rowSrc[offset], CMPMODE::GT,
                            count, 1, {1, 1, 1, 8, 8, 8});
                }
            }
            if constexpr (std::is_same<WorkAcc, half>::value) {
                if (kElems == 128) {
                    Select(accUb[0], cmpUb[0], accUb[0], rowSrc[0],
                           SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           argChunk, 2, {1, 1, 1, 4, 4, 4});
                } else {
                    for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                        const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                        Select(accUb[offset], cmpUb[offset], accUb[offset], rowSrc[offset],
                               SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                    }
                }
            } else {
                Max(accUb[0], accUb[0], rowSrc[0], kElems);
            }
            PipeBarrier<PIPE_V>();
            if (tiling_->hasArg != 0) {
                if (kElems == 128) {
                    Select(argFUb[0], cmpUb[0], argFUb[0], static_cast<float>(rowIdx),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           argChunk, 2, {1, 1, 1, 8, 8, 1});
                } else {
                    for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                        const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                        Select(argFUb[offset], cmpUb[offset], argFUb[offset], static_cast<float>(rowIdx),
                               SELMODE::VSEL_TENSOR_SCALAR_MODE,
                               count, 1, {1, 1, 1, 8, 8, 8});
                    }
                }
            }
            PipeBarrier<PIPE_V>();
        } else {
            constexpr uint32_t argChunk = 64;
            if constexpr (std::is_same<WorkAcc, int32_t>::value) {
                Min(accUb[0], accUb[0], rowSrc[0], kElems);
                PipeBarrier<PIPE_V>();
                if (tiling_->hasArg != 0) {
                    LocalTensor<float> rowFUb = rowFBuf.Get<float>();
                    Duplicate(rowFUb[0], static_cast<float>(rowIdx), kElems);
                    PipeBarrier<PIPE_V>();
                    if (kElems == 128) {
                        Compare(cmpUb[0], accUb[0], rowSrc[0], CMPMODE::EQ,
                                argChunk, 2, {1, 1, 1, 1, 8, 8});
                        PipeBarrier<PIPE_V>();
                        Select(argFUb[0], cmpUb[0], rowFUb[0], argFUb[0],
                               SELMODE::VSEL_TENSOR_TENSOR_MODE,
                               argChunk, 2, {1, 1, 1, 8, 8, 8});
                    } else {
                        for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                            const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                            Compare(cmpUb[offset], accUb[offset], rowSrc[offset], CMPMODE::EQ,
                                    count, 1, {1, 1, 1, 8, 8, 8});
                            PipeBarrier<PIPE_V>();
                            Select(argFUb[offset], cmpUb[offset], rowFUb[offset], argFUb[offset],
                                   SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                   count, 1, {1, 1, 1, 8, 8, 8});
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                return;
            }
            if (kElems == 128) {
                constexpr uint8_t dataRepeatStride =
                    static_cast<uint8_t>(argChunk * sizeof(WorkAcc) / 32);
                Compare(cmpUb[0], accUb[0], rowSrc[0], CMPMODE::LT,
                        argChunk, 2, {1, 1, 1, 1, dataRepeatStride, dataRepeatStride});
            } else {
                for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                    const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                    Compare(cmpUb[offset], accUb[offset], rowSrc[offset], CMPMODE::LT,
                            count, 1, {1, 1, 1, 8, 8, 8});
                }
            }
            if constexpr (std::is_same<WorkAcc, half>::value) {
                if (kElems == 128) {
                    Select(accUb[0], cmpUb[0], accUb[0], rowSrc[0],
                           SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           argChunk, 2, {1, 1, 1, 4, 4, 4});
                } else {
                    for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                        const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                        Select(accUb[offset], cmpUb[offset], accUb[offset], rowSrc[offset],
                               SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                    }
                }
            } else {
                Min(accUb[0], accUb[0], rowSrc[0], kElems);
            }
            PipeBarrier<PIPE_V>();
            if (tiling_->hasArg != 0) {
                if (kElems == 128) {
                    Select(argFUb[0], cmpUb[0], argFUb[0], static_cast<float>(rowIdx),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           argChunk, 2, {1, 1, 1, 8, 8, 1});
                } else {
                    for (uint32_t offset = 0; offset < kElems; offset += argChunk) {
                        const uint32_t count = offset + argChunk <= kElems ? argChunk : kElems - offset;
                        Select(argFUb[offset], cmpUb[offset], argFUb[offset], static_cast<float>(rowIdx),
                               SELMODE::VSEL_TENSOR_SCALAR_MODE,
                               count, 1, {1, 1, 1, 8, 8, 8});
                    }
                }
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void FinalizeSegment(LocalTensor<WorkAcc> accUb, uint32_t kElems, uint64_t rows)
    {
        if constexpr (Reduce == SegmentReduce::SegmentMean) {
            if constexpr (std::is_same<WorkAcc, float>::value) {
                LocalTensor<float> cntUb = rowFBuf.Get<float>();
                Duplicate(cntUb[0], static_cast<float>(static_cast<int32_t>(rows)), kElems);
                PipeBarrier<PIPE_V>();
                Div(accUb[0], accUb[0], cntUb[0], kElems);
                PipeBarrier<PIPE_V>();
            } else if constexpr (std::is_same<WorkAcc, half>::value) {
                LocalTensor<half> cntUb = outCastBuf.Get<half>();
                Duplicate(cntUb[0], static_cast<half>(static_cast<int32_t>(rows)), kElems);
                PipeBarrier<PIPE_V>();
                Div(accUb[0], accUb[0], cntUb[0], kElems);
                PipeBarrier<PIPE_V>();
            } else if constexpr (!((std::is_same<T, int8_t>::value ||
                                    std::is_same<T, uint8_t>::value) &&
                                   std::is_same<WorkAcc, int16_t>::value)) {
                // torch_scatter segment_mean_coo 的整数 mean 使用向零截断。
                for (uint32_t k = 0; k < kElems; ++k) {
                    const int32_t v = static_cast<int32_t>(accUb.GetValue(k));
                    accUb.SetValue(k, static_cast<WorkAcc>(v / static_cast<int32_t>(rows)));
                }
            }
        }
    }

    __aicore__ inline void WriteEmptySegment(uint64_t e, uint64_t seg, uint32_t kElems)
    {
        const uint64_t outOffset = (tiling_->nSegments * e + seg) * kElems;
        if (tiling_->hasInitial != 0) {
            if (tiling_->hasArg != 0) {
                LocalTensor<float> argFUb = argFBuf.Get<float>();
                LocalTensor<int64_t> arg64Ub = arg64Buf.Get<int64_t>();
                Duplicate(argFUb[0], static_cast<float>(static_cast<int32_t>(tiling_->M)), kElems);
                CastArgToInt64(arg64Ub, argFUb, kElems);
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                DataCopy(argGm[outOffset], arg64Ub, kElems);
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            }
            return;
        }
        LocalTensor<float> zeroFUb = zeroFBuf.Get<float>();
        LocalTensor<T> outUb = outCastBuf.Get<T>();
        Duplicate(zeroFUb[0], 0.0f, kElems);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(outGm[outOffset], zeroFUb, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            Duplicate(outUb[0], static_cast<int32_t>(0), kElems);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(outGm[outOffset], outUb, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        } else {
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outUb[0], zeroFUb[0], RoundMode::CAST_RINT, kElems);
            } else if constexpr (std::is_same<T, int16_t>::value) {
                LocalTensor<int32_t> zeroIUb = zeroFBuf.Get<int32_t>();
                Duplicate(zeroIUb[0], static_cast<int32_t>(0), kElems);
                PipeBarrier<PIPE_V>();
                Cast(outUb[0], zeroIUb[0], RoundMode::CAST_NONE, kElems);
            } else if constexpr (std::is_same<T, int8_t>::value ||
                                 std::is_same<T, uint8_t>::value) {
                LocalTensor<half> zeroHUb = accBuf.Get<half>();
                Duplicate(zeroHUb[0], static_cast<half>(0.0f), kElems);
                PipeBarrier<PIPE_V>();
                Cast(outUb[0], zeroHUb[0], RoundMode::CAST_RINT, kElems);
            } else {
                Cast(outUb[0], zeroFUb[0], RoundMode::CAST_NONE, kElems);
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(outGm[outOffset], outUb, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        }
        if (tiling_->hasArg != 0) {
            LocalTensor<float> argFUb = argFBuf.Get<float>();
            LocalTensor<int64_t> arg64Ub = arg64Buf.Get<int64_t>();
            Duplicate(argFUb[0], static_cast<float>(static_cast<int32_t>(tiling_->M)), kElems);
            CastArgToInt64(arg64Ub, argFUb, kElems);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(argGm[outOffset], arg64Ub, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        }
    }

    __aicore__ inline void PackLowBytes(LocalTensor<T> dstUb, LocalTensor<int16_t> srcUb,
                                        uint32_t kElems)
    {
        for (uint32_t k = 0; k < kElems; ++k) {
            dstUb.SetValue(k, static_cast<T>(srcUb.GetValue(k)));
        }
    }

    __aicore__ inline void CopyOut(LocalTensor<WorkAcc> accUb, LocalTensor<float> argFUb,
                                   uint64_t outOffset, uint32_t kElems, uint64_t rows)
    {
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        if constexpr (std::is_same<WorkAcc, T>::value) {
            DataCopy(outGm[outOffset], accUb, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        } else {
            LocalTensor<T> outUb = outCastBuf.Get<T>();
            if constexpr ((std::is_same<T, int8_t>::value ||
                           std::is_same<T, uint8_t>::value) &&
                          std::is_same<WorkAcc, int16_t>::value) {
                PackLowBytes(outUb, accUb, kElems);
                PipeBarrier<PIPE_V>();
                if constexpr (Reduce == SegmentReduce::SegmentMean) {
                    LocalTensor<half> valueHUb = rowFBuf.Get<half>();
                    LocalTensor<half> countHUb = zeroFBuf.Get<half>();
                    Cast(valueHUb[0], outUb[0], RoundMode::CAST_NONE, kElems);
                    PipeBarrier<PIPE_V>();
                    Duplicate(countHUb[0], static_cast<half>(static_cast<int32_t>(rows)), kElems);
                    PipeBarrier<PIPE_V>();
                    Div(valueHUb[0], valueHUb[0], countHUb[0], kElems);
                    PipeBarrier<PIPE_V>();
                    Cast(outUb[0], valueHUb[0], RoundMode::CAST_TRUNC, kElems);
                }
            } else {
                Cast(outUb[0], accUb[0], RoundMode::CAST_RINT, kElems);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(outGm[outOffset], outUb, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        }
        if (tiling_->hasArg != 0) {
            LocalTensor<int64_t> arg64Ub = arg64Buf.Get<int64_t>();
            CastArgToInt64(arg64Ub, argFUb, kElems);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopy(argGm[outOffset], arg64Ub, kElems);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        }
    }

    __aicore__ inline void CastArgToInt64(LocalTensor<int64_t> dst,
                                          LocalTensor<float> src,
                                          uint32_t kElems)
    {
        constexpr uint32_t castChunk = 64;
        for (uint32_t offset = 0; offset < kElems; offset += castChunk) {
            const uint32_t count = offset + castChunk <= kElems ? castChunk : kElems - offset;
            Cast(dst[offset], src[offset], RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessPaddedFloatSumMean()
    {
        const uint64_t blockId = GetBlockIdx();
        const uint64_t blocks = tiling_->usedBlockNum;
        const uint64_t kElems = tiling_->K;
        const uint64_t paddedK =
            ((kElems + tiling_->alignElems - 1) / tiling_->alignElems) * tiling_->alignElems;
        const uint64_t groupTasks =
            tiling_->alignElems / GreatestCommonDivisor(kElems, tiling_->alignElems);
        const uint64_t totalGroups = (tiling_->totalTasks + groupTasks - 1) / groupTasks;
        const uint64_t groupBeg = blockId * totalGroups / blocks;
        const uint64_t groupEnd = (blockId + 1) * totalGroups / blocks;
        LocalTensor<float> srcUb = srcBuf.Get<float>();
        LocalTensor<float> accUb = accBuf.Get<float>();
        LocalTensor<float> countUb = rowFBuf.Get<float>();
        const DataCopyPadParams padParams(
            true, 0, static_cast<uint8_t>(paddedK - kElems), 0);
        for (uint64_t group = groupBeg; group < groupEnd; ++group) {
            const uint64_t taskBeg = group * groupTasks;
            const uint64_t taskEnd = taskBeg + groupTasks <= tiling_->totalTasks
                ? taskBeg + groupTasks
                : tiling_->totalTasks;
            for (uint64_t task = taskBeg; task < taskEnd; ++task) {
                const uint64_t e = task / tiling_->nSegments;
                const uint64_t seg = task % tiling_->nSegments;
                uint64_t r0 = 0;
                uint64_t r1 = 0;
                SegmentBounds(e, seg, r0, r1);
                const uint64_t rows = r1 > r0 ? r1 - r0 : 0;
                const uint64_t srcOffset = (tiling_->M * e + r0) * kElems;
                const uint64_t outOffset = task * kElems;
                Duplicate(accUb[0], 0.0f, paddedK);
                PipeBarrier<PIPE_V>();
                for (uint64_t rowBase = 0; rowBase < rows; rowBase += tiling_->tileRows) {
                    const uint64_t n = rowBase + tiling_->tileRows <= rows
                        ? tiling_->tileRows
                        : rows - rowBase;
                    DataCopyPad(srcUb[0], srcGm[srcOffset + rowBase * kElems],
                                {static_cast<uint16_t>(n),
                                 static_cast<uint16_t>(kElems * sizeof(float)), 0, 0},
                                padParams);
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                    for (uint64_t row = 0; row < n; ++row) {
                        Add(accUb[0], accUb[0], srcUb[row * paddedK], paddedK);
                        PipeBarrier<PIPE_V>();
                    }
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                }
                if constexpr (Reduce == SegmentReduce::SegmentMean) {
                    if (rows != 0) {
                        Duplicate(countUb[0], static_cast<float>(static_cast<int32_t>(rows)), paddedK);
                        PipeBarrier<PIPE_V>();
                        Div(accUb[0], accUb[0], countUb[0], paddedK);
                        PipeBarrier<PIPE_V>();
                    }
                }
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                DataCopyPad(outGm[outOffset], accUb,
                            {1, static_cast<uint32_t>(kElems * sizeof(float)), 0, 0, 0});
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            }
        }
    }

    __aicore__ inline void ProcessPaddedFloatExtrema()
    {
        const uint64_t blockId = GetBlockIdx();
        const uint64_t blocks = tiling_->usedBlockNum;
        const uint32_t kElems = static_cast<uint32_t>(tiling_->K);
        const uint32_t paddedK = static_cast<uint32_t>(
            ((tiling_->K + tiling_->alignElems - 1) / tiling_->alignElems) *
            tiling_->alignElems);
        const uint64_t groupTasks =
            tiling_->alignElems / GreatestCommonDivisor(kElems, tiling_->alignElems);
        const uint64_t totalGroups = (tiling_->totalTasks + groupTasks - 1) / groupTasks;
        const uint64_t groupBeg = blockId * totalGroups / blocks;
        const uint64_t groupEnd = (blockId + 1) * totalGroups / blocks;
        LocalTensor<float> srcUb = srcBuf.Get<float>();
        LocalTensor<float> accUb = accBuf.Get<float>();
        LocalTensor<float> argFUb = argFBuf.Get<float>();
        LocalTensor<int64_t> arg64Ub = arg64Buf.Get<int64_t>();
        LocalTensor<uint8_t> cmpUb = cmpBuf.Get<uint8_t>();
        const DataCopyPadParams padParams(
            false, 0, static_cast<uint8_t>(paddedK - kElems), 0);
        const float neutral = Reduce == SegmentReduce::SegmentMin
            ? SegmentTraits<float>::MaxInit()
            : SegmentTraits<float>::MinInit();
        for (uint64_t group = groupBeg; group < groupEnd; ++group) {
            const uint64_t taskBeg = group * groupTasks;
            const uint64_t taskEnd = taskBeg + groupTasks <= tiling_->totalTasks
                ? taskBeg + groupTasks
                : tiling_->totalTasks;
            for (uint64_t task = taskBeg; task < taskEnd; ++task) {
                const uint64_t e = task / tiling_->nSegments;
                const uint64_t seg = task % tiling_->nSegments;
                uint64_t r0 = 0;
                uint64_t r1 = 0;
                SegmentBounds(e, seg, r0, r1);
                const uint64_t rows = r1 > r0 ? r1 - r0 : 0;
                const uint64_t srcOffset = (tiling_->M * e + r0) * kElems;
                const uint64_t outOffset = task * kElems;
                if (rows == 0) {
                    Duplicate(accUb[0], 0.0f, paddedK);
                    if (tiling_->hasArg != 0) {
                        Duplicate(argFUb[0],
                                  static_cast<float>(static_cast<int32_t>(tiling_->M)), paddedK);
                    }
                } else {
                    Duplicate(accUb[0], neutral, paddedK);
                    if (tiling_->hasArg != 0) {
                        Duplicate(argFUb[0], -1.0f, paddedK);
                    }
                }
                PipeBarrier<PIPE_V>();
                for (uint64_t rowBase = 0; rowBase < rows; rowBase += tiling_->tileRows) {
                    const uint64_t n = rowBase + tiling_->tileRows <= rows
                        ? tiling_->tileRows
                        : rows - rowBase;
                    Duplicate(srcUb[0], neutral, static_cast<uint32_t>(n * paddedK));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                    DataCopyPad(srcUb[0], srcGm[srcOffset + rowBase * kElems],
                                {static_cast<uint16_t>(n),
                                 static_cast<uint16_t>(kElems * sizeof(float)), 0, 0},
                                padParams);
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
                    for (uint64_t row = 0; row < n; ++row) {
                        const int32_t rowIdx = static_cast<int32_t>(r0 + rowBase + row);
                        ReduceRow<false>(accUb, srcUb[row * paddedK], cmpUb, argFUb,
                                         rowIdx, paddedK);
                    }
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
                }
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                DataCopyPad(outGm[outOffset], accUb,
                            {1, static_cast<uint32_t>(kElems * sizeof(float)), 0, 0, 0});
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                if (tiling_->hasArg != 0) {
                    CastArgToInt64(arg64Ub, argFUb, paddedK);
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
                    DataCopyPad(argGm[outOffset], arg64Ub,
                                {1, static_cast<uint32_t>(kElems * sizeof(int64_t)), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // K 非 32B 对齐时的标量保底路径（正确性优先）
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessScalar()
    {
        const uint64_t blockId = GetBlockIdx();
        const uint64_t blocks = tiling_->usedBlockNum;
        const uint64_t kElems = tiling_->K;
        const uint64_t chunkElems = tiling_->alignElems;
        const uint64_t totalElems = tiling_->totalTasks * kElems;
        const uint64_t totalChunks = (totalElems + chunkElems - 1) / chunkElems;
        const uint64_t chunkBeg = blockId * totalChunks / blocks;
        const uint64_t chunkEnd = (blockId + 1) * totalChunks / blocks;
        LocalTensor<T> outUb = outCastBuf.Get<T>();
        LocalTensor<int64_t> argUb = arg64Buf.Get<int64_t>();
        for (uint64_t chunk = chunkBeg; chunk < chunkEnd; ++chunk) {
            const uint64_t outBase = chunk * chunkElems;
            const uint32_t validElems = static_cast<uint32_t>(
                outBase + chunkElems <= totalElems ? chunkElems : totalElems - outBase);
            for (uint64_t lane = 0; lane < chunkElems; ++lane) {
                const uint64_t flat = outBase + lane;
                if (flat >= totalElems) {
                    outUb.SetValue(lane, static_cast<T>(0));
                    if (tiling_->hasArg != 0) {
                        argUb.SetValue(lane, static_cast<int64_t>(tiling_->M));
                    }
                    continue;
                }
                const uint64_t task = flat / kElems;
                const uint64_t k = flat % kElems;
                const uint64_t e = task / tiling_->nSegments;
                const uint64_t seg = task % tiling_->nSegments;
                uint64_t r0 = 0;
                uint64_t r1 = 0;
                SegmentBounds(e, seg, r0, r1);
                const uint64_t rows = r1 > r0 ? r1 - r0 : 0;
                const uint64_t srcBase = (tiling_->M * e + r0) * kElems;
                if (rows == 0) {
                    outUb.SetValue(lane, static_cast<T>(0));
                    if (tiling_->hasArg != 0) {
                        argUb.SetValue(lane, static_cast<int64_t>(tiling_->M));
                    }
                    continue;
                }
                // 标量路径统一在 float 域运算（aicore 限制 half 标量算术）
                float value = static_cast<float>(srcGm.GetValue(srcBase + k));
                int64_t argValue = static_cast<int64_t>(r0);
                for (uint64_t row = 1; row < rows; ++row) {
                    const float candidate =
                        static_cast<float>(srcGm.GetValue(srcBase + row * kElems + k));
                    if constexpr (Reduce == SegmentReduce::SegmentSum ||
                                  Reduce == SegmentReduce::SegmentMean) {
                        value = value + candidate;
                    } else if constexpr (Reduce == SegmentReduce::SegmentMax) {
                        if (candidate >= value) {
                            value = candidate;
                            argValue = static_cast<int64_t>(r0 + row);
                        }
                    } else {
                        if (candidate <= value) {
                            value = candidate;
                            argValue = static_cast<int64_t>(r0 + row);
                        }
                    }
                }
                if constexpr (Reduce == SegmentReduce::SegmentMean) {
                    value = value / static_cast<float>(static_cast<int32_t>(rows));
                }
                outUb.SetValue(lane, static_cast<T>(value));
                if (tiling_->hasArg != 0) {
                    argUb.SetValue(lane, argValue);
                }
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            DataCopyPad(outGm[outBase], outUb,
                        {1, static_cast<uint32_t>(validElems * sizeof(T)), 0, 0, 0});
            if (tiling_->hasArg != 0) {
                DataCopyPad(argGm[outBase], argUb,
                            {1, static_cast<uint32_t>(validElems * sizeof(int64_t)), 0, 0, 0});
            }
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        }
    }

    const SegmentTilingData* tiling_ = nullptr;
    GlobalTensor<T> srcGm;
    GlobalTensor<int64_t> indptrGm;
    GlobalTensor<T> outGm;
    GlobalTensor<int64_t> argGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> rowAccBuf;
    TBuf<TPosition::VECCALC> outCastBuf;
    TBuf<TPosition::VECCALC> argFBuf;
    TBuf<TPosition::VECCALC> rowFBuf;
    TBuf<TPosition::VECCALC> arg64Buf;
    TBuf<TPosition::VECCALC> zeroFBuf;
    TBuf<TPosition::VECCALC> cmpBuf;
    event_t eventIdMte2ToV_;
    event_t eventIdVToMte2_;
    event_t eventIdVToMte3_;
    event_t eventIdMte3ToV_;
};

// ---------------------------------------------------------------------------
// dtype 分发
// ---------------------------------------------------------------------------

template <SegmentReduce Reduce>
__aicore__ inline void DispatchSegmentDType(GM_ADDR src, GM_ADDR indptr, GM_ADDR out,
                                            GM_ADDR argOut, const SegmentTilingData& tiling)
{
    switch (tiling.dtypeCode) {
        case SegmentFloat32: {
            SegmentReduceDevice<float, Reduce> kernel;
            kernel.Init(src, indptr, out, argOut, tiling);
            kernel.Process();
            break;
        }
        case SegmentFloat16: {
            if constexpr (Reduce == SegmentReduce::SegmentMin ||
                          Reduce == SegmentReduce::SegmentMax) {
                if (tiling.K == 128 && tiling.hasInitial == 0) {
                    SegmentReduceDevice<half, Reduce, true> kernel;
                    kernel.Init(src, indptr, out, argOut, tiling);
                    kernel.Process();
                } else {
                    SegmentReduceDevice<half, Reduce> kernel;
                    kernel.Init(src, indptr, out, argOut, tiling);
                    kernel.Process();
                }
            } else {
                SegmentReduceDevice<half, Reduce> kernel;
                kernel.Init(src, indptr, out, argOut, tiling);
                kernel.Process();
            }
            break;
        }
        case SegmentBFloat16: {
            SegmentReduceDevice<bfloat16_t, Reduce> kernel;
            kernel.Init(src, indptr, out, argOut, tiling);
            kernel.Process();
            break;
        }
        case SegmentInt8: {
            SegmentReduceDevice<int8_t, Reduce> kernel;
            kernel.Init(src, indptr, out, argOut, tiling);
            kernel.Process();
            break;
        }
        case SegmentUInt8: {
            SegmentReduceDevice<uint8_t, Reduce> kernel;
            kernel.Init(src, indptr, out, argOut, tiling);
            kernel.Process();
            break;
        }
        case SegmentInt32: {
            SegmentReduceDevice<int32_t, Reduce> kernel;
            kernel.Init(src, indptr, out, argOut, tiling);
            kernel.Process();
            break;
        }
        case SegmentInt16: {
            SegmentReduceDevice<int16_t, Reduce> kernel;
            kernel.Init(src, indptr, out, argOut, tiling);
            kernel.Process();
            break;
        }
        default:
            break;
    }
}

}  // namespace segment
