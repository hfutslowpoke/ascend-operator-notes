#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include "../op_kernel/sparse_softmax_tiling.h"
#include "../op_kernel/tiling_key_sparse_softmax.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
    auto platform=platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    auto *src=ctx->GetRequiredInputTensor(0);auto *index=ctx->GetOptionalInputTensor(1);auto *ptr=ctx->GetOptionalInputTensor(2);
    auto *td=ctx->GetTilingData<SparseSoftmaxTilingData>();if(!src||!td||(!index&&!ptr))return ge::GRAPH_FAILED;
    if(auto *raw=ctx->GetRawTilingData();raw)raw->SetDataSize(sizeof(*td));
    auto shape=src->GetShape().GetStorageShape();const uint32_t rank=shape.GetDimNum();if(rank==0)return ge::GRAPH_FAILED;
    int64_t dim=0;float eps=1e-16f;if(auto *attrs=ctx->GetAttrs();attrs){if(auto*v=attrs->GetInt(0))dim=*v;if(auto*v=attrs->GetFloat(1))eps=*v;}
    if(dim<0)dim+=rank;if(dim<0||dim>=rank)return ge::GRAPH_FAILED;
    uint64_t inner=1,outer=1;for(uint32_t i=0;i<rank;++i){uint64_t d=shape.GetDim(i);if(i<(uint32_t)dim)outer*=d;else if(i>(uint32_t)dim)inner*=d;}
    td->axisLength=static_cast<uint32_t>(shape.GetDim(dim));td->innerSize=static_cast<uint32_t>(inner);td->tasks=static_cast<uint32_t>(outer*inner);
    // Match torch_geometric.utils.softmax: when both optional descriptors are
    // supplied, CSR ptr is the authoritative fast path and index is ignored.
    td->mode=ptr?2u:1u;td->ptrLength=ptr?static_cast<uint32_t>(ptr->GetShapeSize()):0u;td->eps=eps;
    td->indexBroadcast=(index&&index->GetShapeSize()==td->axisLength)?1u:0u;
    uint32_t cores=static_cast<uint32_t>(platform.GetCoreNumAiv());if(cores==0)cores=1;if(td->tasks<cores)cores=static_cast<uint32_t>(td->tasks);if(inner>1)cores=1;if(inner==1&&cores<8)cores=8;if(cores==0)cores=1;
    uint64_t path=0;
    // 64-bit ids: the runtime id dtype alone selects the scalar 64-bit entries
    // (PATH13 = index, PATH14 = ptr) and pad0 carries that marker to the kernel.
    // ia32 ids keep the whole routing chain below unchanged.
    const ge::DataType idDtype=ptr?ptr->GetDataType():(index?index->GetDataType():ge::DT_INT32);
    td->pad0=(idDtype==ge::DT_INT64)?(ptr?2u:1u):0u;
    if(idDtype==ge::DT_INT64)path=ptr?14u:13u;
    else
    // axisLength above the CAP window (>4096): the streaming route keeps its group state in
    // the user workspace and covers any axisLength. Every axisLength <= 4096 keeps the
    // routing and the workspace below unchanged.
    if(td->axisLength>4096u)path=10;
    else
    // Raw-UB fixed entries. PATH6/PATH7 select the raw-UB index routes
    // (index 模式、tasks=1、axis=8、inner=1、广播、fp32/fp16；以及 index 模式、
    // tasks=2、axis=4、inner=1、广播、fp32); PATH8/PATH9 are the raw-UB ptr routes
    // (ptr 模式、tasks=1、axis=8、inner=1、ptrLength=5；以及 ptr 模式、tasks=2、
    // axis=4、inner=1、ptrLength=3；均 fp32). Other domains take the generic chain.
    if(td->mode==1u&&td->tasks==1u&&td->axisLength==8u&&td->innerSize==1u&&td->indexBroadcast==1u&&td->eps==1e-16f)path=6;
    else if(td->mode==1u&&td->tasks==2u&&td->axisLength==4u&&td->innerSize==1u&&td->indexBroadcast==1u&&td->eps==1e-16f&&src->GetDataType()==ge::DT_FLOAT)path=7;
    else if(td->mode==2u&&td->tasks==1u&&td->axisLength==8u&&td->innerSize==1u&&td->indexBroadcast==0u&&td->ptrLength==5u&&td->eps==1e-16f&&src->GetDataType()==ge::DT_FLOAT)path=8;
    else if(td->mode==2u&&td->tasks==2u&&td->axisLength==4u&&td->innerSize==1u&&td->indexBroadcast==0u&&td->ptrLength==3u&&td->eps==1e-16f&&src->GetDataType()==ge::DT_FLOAT)path=9;
    else
    // Small-shape family (inner==1, default eps): every remaining 1<=axis<=8 index / CSR
    // case goes to the run-time-length raw route, so the small-shape region is covered as
    // a family instead of a few pinned sizes. Larger axes keep the generic chain below.
    if((src->GetDataType()==ge::DT_FLOAT16||src->GetDataType()==ge::DT_FLOAT)&&td->mode==1u&&td->tasks<=8u&&td->axisLength>=1u&&td->axisLength<=8u&&td->innerSize==1u&&td->indexBroadcast==1u&&td->eps==1e-16f)path=11;
    else if(src->GetDataType()==ge::DT_FLOAT&&td->mode==2u&&td->tasks<=8u&&td->axisLength>=1u&&td->axisLength<=8u&&td->innerSize==1u&&td->ptrLength>=2u&&td->ptrLength<=49u&&td->eps==1e-16f)path=12;
    else
    // Minimal instantiation for the index, inner==1, 8<axis<=16 domain
    // (one AllPairs16 pass per row).
    if(td->mode==1u&&td->innerSize==1u&&td->axisLength>8u&&td->axisLength<=16u)path=5;
    else // Minimal instantiation for the ptr, inner==1, axis<=8, 2<=ptrLength<=49
    // domain, so its object carries only the small bitmask reduction chain.
    if(td->mode==2u&&td->innerSize==1u&&td->axisLength<=8u&&td->ptrLength>=2u&&td->ptrLength<=49u)path=4;
    else if(td->mode==1u&&td->innerSize==1u&&td->axisLength<=8u)path=1;
    else if(td->mode==2u&&td->innerSize==1u&&td->axisLength<=8u)path=3;
    else if(td->mode==2u&&td->innerSize==1u&&td->axisLength<=48u)path=2;
    else if(td->mode==1u&&td->innerSize==1u&&td->axisLength>8u&&td->axisLength<=48u)path=3;
    uint32_t dtype=static_cast<uint32_t>(src->GetDataType());ASCENDC_TPL_SEL_PARAM(ctx,path,dtype);ctx->SetBlockDim(cores);
    uint64_t wsSize=platform.GetLibApiWorkSpaceSize();
    if(path==10u)wsSize+=static_cast<uint64_t>(cores)*LARGE_AXIS_SLOT_BYTES;   // one group-table slot per block
    ctx->GetWorkspaceSizes(1)[0]=wsSize;return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c){auto*s=c->GetInputShape(0);if(!s)return GRAPH_FAILED;*c->GetOutputShape(0)=*s;return GRAPH_SUCCESS;}
static graphStatus InferDataType(gert::InferDataTypeContext*c){c->SetOutputDataType(0,c->GetInputDataType(0));return GRAPH_SUCCESS;}
}
namespace ops {
class SparseSoftmax:public OpDef{public:explicit SparseSoftmax(const char*n):OpDef(n){
 // Positional dtype combos. Entries 0/1 keep the ia32 id lists unchanged; entries
 // 2..7 add the 64-bit id variants (index-only, ptr-only and both). src/out are
 // float16/float32: bfloat16 is intentionally absent because the AscendC toolchain
 // used for this build rejects bfloat16<->float32 Cast in device code
 // ("not support bf16 type cast"), so no bfloat16 kernel can be produced here.
 const std::vector<ge::DataType> f={ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT};
 const std::vector<ge::DataType> i={ge::DT_INT32,ge::DT_INT32,ge::DT_INT64,ge::DT_INT64,ge::DT_INT32,ge::DT_INT32,ge::DT_INT64,ge::DT_INT64};
 const std::vector<ge::DataType> p={ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64};
 const std::vector<ge::Format> fmt={ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND};
 this->Input("src").ParamType(REQUIRED).DataType(f).Format(fmt);this->Input("index").ParamType(OPTIONAL).DataType(i).Format(fmt);this->Input("ptr").ParamType(OPTIONAL).DataType(p).Format(fmt);this->Output("out").ParamType(REQUIRED).DataType(f).Format(fmt);this->Attr("dim").AttrType(OPTIONAL).Int();this->Attr("eps").AttrType(OPTIONAL).Float(1e-16);this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");}};OP_ADD(SparseSoftmax);
}
