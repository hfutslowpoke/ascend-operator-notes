
#include "segment_min_csr_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
const uint32_t MAX_DEAL_NUM = 2048;

static uint64_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    if (value2 == 0)
    {
        return value1;
    }
    return (value1 + value2 - 1) / value2;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SegmentMinCsrTilingData tiling;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubLength;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto aivCanUseNum = ascendcPlatform.GetCoreNumAiv();
    auto dt = context->GetInputDesc(0)->GetDataType();
    if (dt == ge::DT_INT8)
    {
        sizeofdatatype = 1;
    }
    else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16)
    {
        sizeofdatatype = 2;
    }
    else
    {
        sizeofdatatype = 4;
    }

    const auto& srcShape = context->GetInputShape(0)->GetStorageShape();
    const auto& indptrShape = context->GetInputShape(1)->GetStorageShape();
    uint32_t srcDimNum = srcShape.GetDimNum();
    uint32_t indptrDimNum = indptrShape.GetDimNum();
    uint32_t srcTotalSize = srcShape.GetShapeSize();
    uint32_t indptrPhysicalNum = indptrShape.GetShapeSize();

    if (indptrDimNum < 1) {
        return ge::GRAPH_FAILED;
    }

    uint32_t E_1 = 1;
    for (uint32_t d = 0; d + 1 < indptrDimNum; ++d)
    {
        uint32_t s = srcShape.GetDim(d);
        uint32_t ip = indptrShape.GetDim(d);
        if (!(ip == s || ip == 1)) {
            return ge::GRAPH_FAILED;
        }
        E_1 *= s;
    }

    uint32_t M = srcShape.GetDim(indptrDimNum - 1);
    uint32_t indptrLastDim = indptrShape.GetDim(indptrDimNum - 1);
    uint32_t nSegments = indptrLastDim - 1;

    uint32_t K = 1;
    for (uint32_t d = indptrDimNum; d < srcDimNum; ++d)
    {
        K *= srcShape.GetDim(d);
    }

    uint32_t indptrPreProduct = 1;
    for (uint32_t d = 0; d + 1 < indptrDimNum; ++d)
    {
        indptrPreProduct *= indptrShape.GetDim(d);
    }
    uint32_t strideIndptr = (indptrPreProduct == E_1) ? indptrLastDim : 0;

    tiling.set_srcLength(srcTotalSize);
    tiling.set_E_1(E_1);
    tiling.set_M(M);
    tiling.set_nSegments(nSegments);
    tiling.set_K(K);
    tiling.set_indptrPhysicalNum(indptrPhysicalNum);
    tiling.set_indptrDimNum(indptrDimNum);
    tiling.set_strideIndptr(strideIndptr);

    auto optionalOut = context->GetOptionalInputShape(2);
    if (optionalOut != nullptr)
    {
        tiling.set_hasOptionalOut(1);
    }
    else
    {
        tiling.set_hasOptionalOut(0);
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;

    uint32_t aivNum = (aivCanUseNum < E_1) ? aivCanUseNum : E_1;
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t rowsPerAiv = GetCeilInt(E_1, aivNum);
    uint32_t rowsLastAiv = (E_1 % aivNum == 0) ? rowsPerAiv : (E_1 % aivNum);

    uint32_t coreDataNum = MAX_DEAL_NUM;
    uint32_t coreTailDataNum = (K % MAX_DEAL_NUM == 0) ? MAX_DEAL_NUM : (K % MAX_DEAL_NUM);

    uint32_t KloopTime = GetCeilInt(K, MAX_DEAL_NUM);

    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_rowsPerAiv(rowsPerAiv);
    tiling.set_rowsLastAiv(rowsLastAiv);
    tiling.set_KloopTime(KloopTime);
    tiling.set_coreDataNum(coreDataNum);
    tiling.set_coreTailDataNum(coreTailDataNum);
    tiling.set_aivNum(aivNum);

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* srcShape = context->GetInputShape(0);
    const gert::Shape* indptrShape = context->GetInputShape(1);
    uint32_t indptrDimNum = indptrShape->GetDimNum();

    if (indptrDimNum < 1) {
        return ge::GRAPH_FAILED;
    }

    uint32_t nSegments = indptrShape->GetDim(indptrDimNum - 1) - 1;

    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *srcShape;
    outShape->SetDim(indptrDimNum - 1, nSegments);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SegmentMinCsr : public OpDef {
public:
    explicit SegmentMinCsr(const char* name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("optional_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SegmentMinCsr);
}
