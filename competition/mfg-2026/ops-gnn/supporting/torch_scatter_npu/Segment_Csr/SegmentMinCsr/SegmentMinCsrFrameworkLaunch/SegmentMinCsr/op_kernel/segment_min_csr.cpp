#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t MAX_DEAL_NUM = 2048;
constexpr uint32_t MAX_MASK = 64;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MIN_COPY_SIZE = 8;

template <typename T>
struct SegmentMinTraits;

template <>
struct SegmentMinTraits<float> {
    static __aicore__ inline float FillValue() { return 3.4028235e+38f; }
};

template <>
struct SegmentMinTraits<int16_t> {
    static __aicore__ inline int16_t FillValue() { return 32767; }
};

template <>
struct SegmentMinTraits<int32_t> {
    static __aicore__ inline int32_t FillValue() { return 2147483647; }
};

template <>
struct SegmentMinTraits<half> {
    static __aicore__ inline half FillValue() { return static_cast<half>(65504.0f); }
};

template <typename T>
class SegmentMinCsrKernel {
public:
    __aicore__ inline SegmentMinCsrKernel() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indptr, GM_ADDR optional_out, GM_ADDR out,
                                SegmentMinCsrTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        this->srcLength = tiling_data->srcLength;
        this->E_1 = tiling_data->E_1;
        this->K = tiling_data->K;
        this->M = tiling_data->M;
        this->nSegments = tiling_data->nSegments;
        this->coreDataNum = tiling_data->coreDataNum;
        this->coreTailDataNum = tiling_data->coreTailDataNum;
        this->ALIGN_NUM = tiling_data->ALIGN_NUM;
        this->aivNum = tiling_data->aivNum;
        this->rowsPerAiv = tiling_data->rowsPerAiv;
        this->rowsLastAiv = tiling_data->rowsLastAiv;
        this->KloopTime = tiling_data->KloopTime;
        this->indptrPhysicalNum = tiling_data->indptrPhysicalNum;
        this->strideIndptr = tiling_data->strideIndptr;
        this->indptrDimNum = tiling_data->indptrDimNum;
        this->hasOptionalOut = tiling_data->hasOptionalOut;

        this->coreRowBeg = GetBlockIdx() * this->rowsPerAiv;
        this->coreRowEnd = this->coreRowBeg + ((GetBlockIdx() == this->aivNum - 1) ? this->rowsLastAiv : this->rowsPerAiv);
        if (this->coreRowEnd > this->E_1) this->coreRowEnd = this->E_1;

        auto outputLength = this->E_1 * this->nSegments * this->K;

        srcGm.SetGlobalBuffer((__gm__ T*)src, srcLength);
        indptrGm.SetGlobalBuffer((__gm__ int32_t*)indptr, indptrPhysicalNum);
        outGm.SetGlobalBuffer((__gm__ T*)out, outputLength);
        if (hasOptionalOut) {
            optionalOutGm.SetGlobalBuffer((__gm__ T*)optional_out, outputLength);
        }

        eventIdMte2ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdVToMte3_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdMte3ToMte2_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdMte2ToMte3_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());

        pipe->InitBuffer(inQueueSrc, this->coreDataNum * sizeof(T));
        pipe->InitBuffer(inQueueOptionalOut, this->coreDataNum * sizeof(T));
        pipe->InitBuffer(outQueueOut, this->coreDataNum * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t e = this->coreRowBeg; e < this->coreRowEnd; e++) {
            Compute(e);
        }
    }

private:
    __aicore__ inline uint32_t GetIndptrOffset(uint32_t e)
    {
        if (this->strideIndptr == 0) {
            return 0;
        }
        return e * this->strideIndptr;
    }

    __aicore__ inline uint32_t GetAlignedCopyLen(uint32_t len)
    {
        return ((len + this->ALIGN_NUM - 1) / this->ALIGN_NUM) * this->ALIGN_NUM;
    }

    template <typename DType>
    __aicore__ inline void ComputeDataCopy(const GlobalTensor<DType>& dst, const LocalTensor<DType>& src, const uint32_t calCount)
    {
        int32_t numPerBlock = BLOCK_SIZE / sizeof(DType);
        if (calCount % numPerBlock == 0) {
            DataCopy(dst, src, calCount);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(DType)), 0, 0, 0};
            DataCopyPad(dst, src, copyParams);
        }
    }

    __aicore__ inline void ProcessFirstWithOptional(uint32_t srcBase, uint32_t outBase)
    {
        for (int32_t loop = 0; loop < this->KloopTime - 1; loop++) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

            DataCopy(srcLocal, srcGm[srcBase + loop * MAX_DEAL_NUM], MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

            DataCopy(optionalOutLocal, optionalOutGm[outBase + loop * MAX_DEAL_NUM], MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

            Min(outLocal, optionalOutLocal, srcLocal, MAX_DEAL_NUM);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

            ComputeDataCopy<T>(outGm[outBase + loop * MAX_DEAL_NUM], outLocal, MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

        uint32_t tailOffset = (this->KloopTime - 1) * MAX_DEAL_NUM;
        uint32_t Ktaillength = this->K - tailOffset;
        uint32_t KtaillengthCopy = GetAlignedCopyLen(Ktaillength);

        DataCopy(srcLocal, srcGm[srcBase + tailOffset], KtaillengthCopy);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

        DataCopy(optionalOutLocal, optionalOutGm[outBase + tailOffset], KtaillengthCopy);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

        Min(outLocal, optionalOutLocal, srcLocal, KtaillengthCopy);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

        ComputeDataCopy<T>(outGm[outBase + tailOffset], outLocal, Ktaillength);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
    }

    __aicore__ inline void ProcessFirstNoOptional(uint32_t srcBase, uint32_t outBase)
    {
        for (int32_t loop = 0; loop < this->KloopTime - 1; loop++) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

            DataCopy(srcLocal, srcGm[srcBase + loop * MAX_DEAL_NUM], MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

            DataCopy(outLocal, srcLocal, MAX_DEAL_NUM);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

            ComputeDataCopy<T>(outGm[outBase + loop * MAX_DEAL_NUM], outLocal, MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

        uint32_t tailOffset = (this->KloopTime - 1) * MAX_DEAL_NUM;
        uint32_t Ktaillength = this->K - tailOffset;
        uint32_t KtaillengthCopy = GetAlignedCopyLen(Ktaillength);

        DataCopy(srcLocal, srcGm[srcBase + tailOffset], KtaillengthCopy);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

        DataCopy(outLocal, srcLocal, KtaillengthCopy);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

        ComputeDataCopy<T>(outGm[outBase + tailOffset], outLocal, Ktaillength);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
    }

    __aicore__ inline void ProcessAccumulate(uint32_t srcBase, uint32_t outBase)
    {
        for (int32_t loop = 0; loop < this->KloopTime - 1; loop++) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

            DataCopy(srcLocal, srcGm[srcBase + loop * MAX_DEAL_NUM], MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

            DataCopy(optionalOutLocal, outGm[outBase + loop * MAX_DEAL_NUM], MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

            Min(outLocal, optionalOutLocal, srcLocal, MAX_DEAL_NUM);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

            ComputeDataCopy<T>(outGm[outBase + loop * MAX_DEAL_NUM], outLocal, MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

        uint32_t tailOffset = (this->KloopTime - 1) * MAX_DEAL_NUM;
        uint32_t Ktaillength = this->K - tailOffset;
        uint32_t KtaillengthCopy = GetAlignedCopyLen(Ktaillength);

        DataCopy(srcLocal, srcGm[srcBase + tailOffset], KtaillengthCopy);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

        DataCopy(optionalOutLocal, outGm[outBase + tailOffset], KtaillengthCopy);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

        Min(outLocal, optionalOutLocal, srcLocal, KtaillengthCopy);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

        ComputeDataCopy<T>(outGm[outBase + tailOffset], outLocal, Ktaillength);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
    }

    __aicore__ inline void ProcessEmptySegment(uint32_t e_1, uint32_t seg)
    {
        uint32_t outBase = e_1 * this->nSegments * this->K + seg * this->K;
        for (int32_t loop = 0; loop < this->KloopTime - 1; loop++) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

            Duplicate(outLocal, SegmentMinTraits<T>::FillValue(), MAX_DEAL_NUM);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

            ComputeDataCopy<T>(outGm[outBase + loop * MAX_DEAL_NUM], outLocal, MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

        uint32_t tailOffset = (this->KloopTime - 1) * MAX_DEAL_NUM;
        uint32_t Ktaillength = this->K - tailOffset;
        uint32_t KtaillengthAligned = GetAlignedCopyLen(Ktaillength);
        Duplicate(outLocal, SegmentMinTraits<T>::FillValue(), KtaillengthAligned);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

        ComputeDataCopy<T>(outGm[outBase + tailOffset], outLocal, Ktaillength);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
    }

    __aicore__ inline void ProcessEmptySegmentWithOptional(uint32_t e_1, uint32_t seg)
    {
        uint32_t outBase = e_1 * this->nSegments * this->K + seg * this->K;
        for (int32_t loop = 0; loop < this->KloopTime - 1; loop++) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);

            DataCopy(outLocal, optionalOutGm[outBase + loop * MAX_DEAL_NUM], MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);

            ComputeDataCopy<T>(outGm[outBase + loop * MAX_DEAL_NUM], outLocal, MAX_DEAL_NUM);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);

        uint32_t tailOffset = (this->KloopTime - 1) * MAX_DEAL_NUM;
        uint32_t Ktaillength = this->K - tailOffset;
        uint32_t KtaillengthCopy = GetAlignedCopyLen(Ktaillength);

        DataCopy(outLocal, optionalOutGm[outBase + tailOffset], KtaillengthCopy);
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);

        ComputeDataCopy<T>(outGm[outBase + tailOffset], outLocal, Ktaillength);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
    }

    __aicore__ inline void Compute(uint32_t e)
    {
        srcLocal = inQueueSrc.Get<T>();
        optionalOutLocal = inQueueOptionalOut.Get<T>();
        outLocal = outQueueOut.Get<T>();

        uint32_t indptrBase = GetIndptrOffset(e);

        for (uint32_t seg = 0; seg < this->nSegments; seg++) {
            int32_t start = indptrGm.GetValue(indptrBase + seg);
            int32_t end = indptrGm.GetValue(indptrBase + seg + 1);

            uint32_t outBase = e * this->nSegments * this->K + seg * this->K;

            if (start >= end) {
                if (this->hasOptionalOut) {
                    ProcessEmptySegmentWithOptional(e, seg);
                } else {
                    ProcessEmptySegment(e, seg);
                }
                continue;
            }

            for (int32_t j = start; j < end; j++) {
                uint32_t srcBase = e * this->M * this->K + j * this->K;

                if (j == start) {
                    if (this->hasOptionalOut) {
                        ProcessFirstWithOptional(srcBase, outBase);
                    } else {
                        ProcessFirstNoOptional(srcBase, outBase);
                    }
                } else {
                    ProcessAccumulate(srcBase, outBase);
                }
            }
        }
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueSrc, inQueueOptionalOut;
    TBuf<TPosition::VECCALC> outQueueOut;

    GlobalTensor<T> srcGm;
    GlobalTensor<int32_t> indptrGm;
    GlobalTensor<T> optionalOutGm;
    GlobalTensor<T> outGm;

    LocalTensor<T> srcLocal;
    LocalTensor<T> optionalOutLocal;
    LocalTensor<T> outLocal;

    uint32_t srcLength, E_1, K, M, nSegments, coreDataNum, coreTailDataNum;
    uint32_t ALIGN_NUM, aivNum, rowsPerAiv, rowsLastAiv, KloopTime;
    uint32_t indptrPhysicalNum, strideIndptr, indptrDimNum, hasOptionalOut;
    uint32_t coreRowBeg, coreRowEnd;

    event_t eventIdMte2ToV_0, eventIdMte2ToV_1, eventIdVToMte3_0, eventIdMte3ToV_0, eventIdMte3ToMte2_0;
    event_t eventIdMte2ToMte3_0;
};

extern "C" __global__ __aicore__ void segment_min_csr(GM_ADDR src, GM_ADDR indptr, GM_ADDR optional_out, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    SegmentMinCsrKernel<DTYPE_SRC> op;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    op.Init(src, indptr, optional_out, out, &tiling_data, &pipe);
    op.Process();
}
