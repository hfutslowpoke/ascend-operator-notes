#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "../op_kernel/tiling_key_dense_lightning_indexer_grad_kl_loss.h"
namespace optiling {
constexpr uint64_t WS_ALIGN = 512;
constexpr uint64_t WS_GUARD = 8192;
static inline uint64_t AlignUp(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
auto *td = context->GetTilingData<DenseLightningIndexerGradKlLossTilingData>();
if (td == nullptr) return ge::GRAPH_FAILED;
if (auto *raw = context->GetRawTilingData(); raw != nullptr) raw->SetDataSize(sizeof(*td));
const gert::Tensor *q = context->GetRequiredInputTensor(0);
const gert::Tensor *k = context->GetRequiredInputTensor(1);
const gert::Tensor *qi = context->GetRequiredInputTensor(2);
const gert::Tensor *ki = context->GetRequiredInputTensor(3);
const gert::Tensor *w = context->GetRequiredInputTensor(4);
if (q == nullptr || k == nullptr || qi == nullptr || ki == nullptr || w == nullptr) return ge::GRAPH_FAILED;
auto qs = q->GetShape().GetStorageShape();
auto ks = k->GetShape().GetStorageShape();
auto qis = qi->GetShape().GetStorageShape();
auto kis = ki->GetShape().GetStorageShape();
auto ws = w->GetShape().GetStorageShape();
if (qs.GetDimNum() != 4 || ks.GetDimNum() != 4 || qis.GetDimNum() != 4 || kis.GetDimNum() != 4 || ws.GetDimNum() != 3) return ge::GRAPH_FAILED;
td->bSize = static_cast<uint32_t>(qs.GetDim(0));
td->s1Size = static_cast<uint32_t>(qs.GetDim(1));
td->n1Size = static_cast<uint32_t>(qs.GetDim(2));
td->dQuery = static_cast<uint32_t>(qs.GetDim(3));
td->s2Size = static_cast<uint32_t>(ks.GetDim(1));
td->nidx1Size = static_cast<uint32_t>(qis.GetDim(2));
td->dQueryIndex = static_cast<uint32_t>(qis.GetDim(3));
td->kSize = td->s2Size;
td->dRope = 0;
td->hasRope = 0;
td->weightsFp32 = w->GetDataType() == ge::DT_FLOAT ? 1u : 0u;
td->sparseMode = 0;
td->useSeqLens = 0;
td->seqLensIsI64 = 0;
td->totalTasks = td->bSize * td->s1Size;
const uint32_t weightRowBytes = td->nidx1Size * (td->weightsFp32 ? 4u : 2u);
uint32_t groupRows = 1u;
while ((groupRows * weightRowBytes) % 32u != 0u) ++groupRows;
const uint32_t groups = (td->totalTasks + groupRows - 1u) / groupRows;
uint32_t hwCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
if (hwCores == 0u) hwCores = 1u;
uint32_t cores = hwCores;
if (cores > groups) cores = groups;
if (cores == 0u) cores = 1u;
const uint32_t dtype = static_cast<uint32_t>(q->GetDataType());
for (auto &v : td->reserved) v = 0;
const bool p7ExactPublic =
dtype == static_cast<uint32_t>(ge::DT_FLOAT16) && td->weightsFp32 == 0u &&
td->bSize == 1u && td->s1Size == 1u && td->s2Size == 8u && td->kSize == td->s2Size &&
td->n1Size == 128u && td->nidx1Size == 64u &&
td->dQuery == 128u && td->dQueryIndex == 128u &&
td->hasRope == 0u && td->sparseMode == 0u && td->useSeqLens == 0u &&
td->totalTasks == 1u && hwCores >= 8u;
const bool headSplitDomain =
td->totalTasks <= 2u && td->s2Size <= 8u && td->hasRope == 0u &&
td->dQuery == 128u && td->dQueryIndex == 128u &&
(td->n1Size == 32u || td->n1Size == 64u || td->n1Size == 128u) &&
td->nidx1Size >= 8u && td->nidx1Size <= 64u && (td->nidx1Size % 8u) == 0u;
uint32_t headParts = 0u;
if (headSplitDomain) {
if (p7ExactPublic) headParts = 8u;
for (uint32_t parts = 4u; parts >= 2u && headParts == 0u; --parts) {
if ((td->n1Size % parts) != 0u || (td->nidx1Size % parts) != 0u) continue;
if (((td->n1Size / parts) % 8u) != 0u) continue;
if ((td->nidx1Size / parts) < 8u) continue;
if (((td->nidx1Size / parts) % 8u) != 0u) continue;
if (static_cast<uint64_t>(parts) * td->totalTasks > hwCores) continue;
headParts = parts;
break;
}
}
if (headParts >= 2u) {
td->reserved[0] = 1u;
td->coreNum = td->totalTasks * headParts;
td->tasksPerCore = 1u;
} else {
const uint32_t groupsPerCore = (groups + cores - 1u) / cores;
td->tasksPerCore = groupsPerCore * groupRows;
td->coreNum = (td->totalTasks + td->tasksPerCore - 1u) / td->tasksPerCore;
if (td->totalTasks <= 2u && td->s2Size <= 8u && td->hasRope == 0u &&
td->dQuery == td->dQueryIndex && hwCores >= td->totalTasks && td->totalTasks > 1u) {
td->tasksPerCore = 1u;
td->coreNum = td->totalTasks;
}
}
td->scaleValue = 0.0884f;
if (const auto *attrs = context->GetAttrs(); attrs != nullptr) {
if (const float *v = attrs->GetFloat(0); v != nullptr) td->scaleValue = *v;
}
td->invN1 = td->n1Size == 0 ? 0.0f : 1.0f / static_cast<float>(td->n1Size);
const uint64_t taskElems = static_cast<uint64_t>(td->totalTasks) * td->s2Size;
uint64_t off = 0;
td->wsOffsetP = off; off += AlignUp(taskElems * sizeof(float), WS_ALIGN);
td->wsOffsetI = off; off += AlignUp(taskElems * sizeof(float), WS_ALIGN);
td->wsOffsetScatter = off;
off += AlignUp(static_cast<uint64_t>(td->bSize) * td->s2Size * td->dQueryIndex * sizeof(float), WS_ALIGN);
if (td->reserved[0] == 1u) {
off = AlignUp(off + static_cast<uint64_t>(td->coreNum) * 32u, WS_ALIGN);
td->reserved[1] = static_cast<uint32_t>(off);
off += AlignUp(static_cast<uint64_t>(td->coreNum) * 16u * sizeof(float), WS_ALIGN);
td->reserved[2] = static_cast<uint32_t>(off);
off += AlignUp(static_cast<uint64_t>(td->coreNum) * 16u * sizeof(float), WS_ALIGN);
td->reserved[3] = static_cast<uint32_t>(off);
off += AlignUp(static_cast<uint64_t>(td->totalTasks) * td->coreNum * td->s2Size *
td->dQueryIndex * sizeof(float),
WS_ALIGN);
}
size_t *workspace = context->GetWorkspaceSizes(1);
workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize() + off + WS_GUARD);
const bool fuseBig = td->s2Size <= 8u && td->nidx1Size > 16u &&
td->nidx1Size <= 64u && td->dQueryIndex == 128u &&
(td->n1Size == 32u || td->n1Size == 64u || td->n1Size == 128u);
const bool smallIndexHeads = td->nidx1Size >= 1u && td->nidx1Size <= 8u;
const bool fullIndexHeadsLargeMain = td->nidx1Size > 8u && td->nidx1Size <= 16u &&
td->n1Size > 64u && td->n1Size <= 128u;
const bool wideFuse = td->s2Size > 12u && td->s2Size <= 16u &&
(smallIndexHeads || fullIndexHeadsLargeMain) &&
td->dQueryIndex == 128u;
const bool width64 = (td->dQueryIndex == 64u);
const bool oneTaskPerCore =
td->coreNum == td->totalTasks &&
(td->tasksPerCore == 1u || td->totalTasks == 1u);
const bool leanPublic =
dtype == static_cast<uint32_t>(ge::DT_FLOAT16) &&
td->weightsFp32 == 0u && td->hasRope == 0u && td->kSize == td->s2Size &&
td->n1Size == 32u && td->nidx1Size == 8u && td->s2Size == 4u && td->s1Size == 1u &&
td->dQuery == td->dQueryIndex && (td->dQuery == 64u || td->dQuery == 128u) &&
td->totalTasks >= 1u && td->totalTasks <= 2u && td->totalTasks == td->bSize &&
td->coreNum >= 1u && td->coreNum <= 2u &&
oneTaskPerCore && !fuseBig && !wideFuse;
const bool weightsMatchDtype =
td->weightsFp32 == ((dtype == static_cast<uint32_t>(ge::DT_FLOAT)) ? 1u : 0u);
const uint32_t headPartsFromCore =
(td->totalTasks != 0u && td->reserved[0] == 1u) ? td->coreNum / td->totalTasks : 0u;
const bool headPublicCommon =
td->bSize == 1u && td->s1Size == td->totalTasks && td->hasRope == 0u &&
td->sparseMode == 0u && td->useSeqLens == 0u && td->kSize == td->s2Size &&
td->dQuery == 128u && td->dQueryIndex == 128u && weightsMatchDtype &&
td->reserved[0] == 1u && td->tasksPerCore == 1u &&
!wideFuse && !width64;
const bool headPublicA =
headPublicCommon && td->coreNum == 4u &&
dtype == static_cast<uint32_t>(ge::DT_FLOAT16) &&
td->weightsFp32 == 0u && td->s1Size == 2u && td->s2Size == 8u && td->n1Size == 64u &&
td->nidx1Size == 16u && td->totalTasks == 2u && headPartsFromCore == 2u && !fuseBig;
const bool headPublicB =
headPublicCommon && td->coreNum == 4u &&
dtype == static_cast<uint32_t>(ge::DT_FLOAT) &&
td->weightsFp32 == 1u && td->s1Size == 1u && td->s2Size == 4u && td->n1Size == 64u &&
td->nidx1Size == 32u && td->totalTasks == 1u && headPartsFromCore == 4u && fuseBig;
const bool headPublicC =
headPublicCommon && td->coreNum == 8u &&
dtype == static_cast<uint32_t>(ge::DT_FLOAT16) &&
td->weightsFp32 == 0u && td->s1Size == 1u && td->s2Size == 8u && td->n1Size == 128u &&
td->nidx1Size == 64u && td->totalTasks == 1u && headPartsFromCore == 8u && fuseBig;
const bool headPublic = headPublicA || headPublicB || headPublicC;
const bool selectedWideFuse = wideFuse || (leanPublic && td->bSize == 2u);
ASCENDC_TPL_SEL_PARAM(context, dtype, fuseBig, selectedWideFuse, width64, leanPublic, headPublic);
if (headPublic || (leanPublic && !width64)) {
if (context->SetScheduleMode(1u) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
}
context->SetBlockDim(td->coreNum);
return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
const gert::Shape *qi = context->GetInputShape(2);
const gert::Shape *ki = context->GetInputShape(3);
const gert::Shape *w = context->GetInputShape(4);
if (qi == nullptr || ki == nullptr || w == nullptr) return GRAPH_FAILED;
*context->GetOutputShape(0) = *qi;
*context->GetOutputShape(1) = *ki;
*context->GetOutputShape(2) = *w;
gert::Shape loss; loss.AppendDim(1); *context->GetOutputShape(3) = loss;
return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
context->SetOutputDataType(0, context->GetInputDataType(2));
context->SetOutputDataType(1, context->GetInputDataType(3));
context->SetOutputDataType(2, context->GetInputDataType(4));
context->SetOutputDataType(3, ge::DT_FLOAT);
return GRAPH_SUCCESS;
}
}
namespace ops {
class DenseLightningIndexerGradKlLoss : public OpDef {
public:
explicit DenseLightningIndexerGradKlLoss(const char *name) : OpDef(name) {
const std::vector<ge::DataType> dt = {ge::DT_FLOAT16, ge::DT_FLOAT};
const std::vector<ge::Format> fmt = {ge::FORMAT_ND, ge::FORMAT_ND};
this->Input("query").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Input("key").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Input("query_index").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Input("key_index").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Input("weights").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Output("d_query_index").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Output("d_key_index").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Output("d_weights").ParamType(REQUIRED).DataType(dt).Format(fmt);
this->Output("loss").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(fmt);
this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0884);
this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}
};
OP_ADD(DenseLightningIndexerGradKlLoss);
}
