#include "segment/common/segment_kernel_common.h"

using namespace segment;

extern "C" __global__ __aicore__ void segment_min_2201(
    GM_ADDR src, GM_ADDR indptr, GM_ADDR out, GM_ADDR argOut, SegmentTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DispatchSegmentDType<SegmentReduce::SegmentMin>(src, indptr, out, argOut, tiling);
}
