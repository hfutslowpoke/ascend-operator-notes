
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SegmentMinCsrTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, srcLength);
  TILING_DATA_FIELD_DEF(uint32_t, E_1);
  TILING_DATA_FIELD_DEF(uint32_t, nSegments);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, indptrPhysicalNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerAiv);
  TILING_DATA_FIELD_DEF(uint32_t, rowsLastAiv);
  TILING_DATA_FIELD_DEF(uint32_t, KloopTime);
  TILING_DATA_FIELD_DEF(uint32_t, hasOptionalOut);
  TILING_DATA_FIELD_DEF(uint32_t, indptrDimNum);
  TILING_DATA_FIELD_DEF(uint32_t, strideIndptr);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SegmentMinCsr, SegmentMinCsrTilingData)
}
