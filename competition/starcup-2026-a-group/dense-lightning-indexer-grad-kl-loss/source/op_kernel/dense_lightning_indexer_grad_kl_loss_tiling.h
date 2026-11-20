#pragma once
#include <cstdint>
struct DenseLightningIndexerGradKlLossTilingData {
uint32_t bSize, s1Size, s2Size, n1Size, nidx1Size, kSize;
uint32_t dQuery, dQueryIndex, dRope;
uint32_t hasRope, weightsFp32, sparseMode, useSeqLens, seqLensIsI64;
uint32_t coreNum, totalTasks, tasksPerCore;
float scaleValue, invN1;
uint64_t wsOffsetP, wsOffsetI, wsOffsetScatter;
uint32_t reserved[6];
};
static_assert(sizeof(DenseLightningIndexerGradKlLossTilingData) == 128,
"Dense tiling ABI must remain 128 bytes");
