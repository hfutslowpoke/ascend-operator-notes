#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
#include "sparse_flash_attention_grad_tiling.h"
ASCENDC_TPL_ARGS_DECL(SparseFlashAttentionGrad,ASCENDC_TPL_DATATYPE_DECL(DT_QUERY,C_DT_FLOAT16,C_DT_BF16),ASCENDC_TPL_BOOL_DECL(DUP_PATH,0,1),ASCENDC_TPL_BOOL_DECL(HAS_ROPE,0,1),ASCENDC_TPL_BOOL_DECL(HAS_VALUE,0,1),ASCENDC_TPL_BOOL_DECL(CUBE_PATH,0,1),ASCENDC_TPL_BOOL_DECL(FW_PATH,0,1),ASCENDC_TPL_BOOL_DECL(COUNT_RANK,0,1),ASCENDC_TPL_BOOL_DECL(BATCH_WINDOW,0,1),ASCENDC_TPL_BOOL_DECL(OPT_SCORE,0,1),ASCENDC_TPL_BOOL_DECL(OPT_WINDOW,0,1),ASCENDC_TPL_BOOL_DECL(BYTE_COUNT,0,1),ASCENDC_TPL_BOOL_DECL(HR_GRAPH_ENTRY,0,1),ASCENDC_TPL_BOOL_DECL(C10_OPT_ENTRY,0,1),ASCENDC_TPL_BOOL_DECL(C10_WB_ENTRY,0,1),);
ASCENDC_TPL_SEL(
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagLegacyTiling96),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,0),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,1),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,1),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,0),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,1),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,1),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagLegacyTiling100),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,0),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,1),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,1),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,0),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,1),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagLegacyTiling100),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,1),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,1),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,1),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,1),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,0),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,0),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,0),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,0),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,1),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,1),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,1),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,1),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,1),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,0),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagLegacyTiling100),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,1),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,1),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,1),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,1),ASCENDC_TPL_BOOL_SEL(FW_PATH,1),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,0),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,1),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagCompactTiling8),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_FLOAT16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,0),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,0),ASCENDC_TPL_BOOL_SEL(FW_PATH,0),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,0),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagGeneralTiling64),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_QUERY,C_DT_BF16),ASCENDC_TPL_BOOL_SEL(DUP_PATH,0),ASCENDC_TPL_BOOL_SEL(HAS_ROPE,0),ASCENDC_TPL_BOOL_SEL(HAS_VALUE,0),ASCENDC_TPL_BOOL_SEL(CUBE_PATH,0),ASCENDC_TPL_BOOL_SEL(FW_PATH,0),ASCENDC_TPL_BOOL_SEL(COUNT_RANK,0),ASCENDC_TPL_BOOL_SEL(BATCH_WINDOW,0),ASCENDC_TPL_BOOL_SEL(OPT_SCORE,0),ASCENDC_TPL_BOOL_SEL(OPT_WINDOW,0),ASCENDC_TPL_BOOL_SEL(BYTE_COUNT,0),ASCENDC_TPL_BOOL_SEL(HR_GRAPH_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_OPT_ENTRY,0),ASCENDC_TPL_BOOL_SEL(C10_WB_ENTRY,0),ASCENDC_TPL_TILING_STRUCT_SEL(SfagGeneralTiling64),ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY))
);

#ifndef SFAG_ALL15_TOKEN_DICTIONARY
#define SFAG_ALL15_TOKEN_DICTIONARY
#define J00 >low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,W0n Jbm,W3s){Jd0(W3p,JI2>W0p||W3p,T>W0p,"PairFull output must be FP32 scratch or the input dtype")W0R as=Jcz%2,ao=as W1p,bo=as*65536;Jb7 acp{1,W2q(TA?k:m),W2q(TA?m:k),0,W2q(Jcl),W0e
#define J01 ;W13(W3p,JI2>W0p)fp.JcT=W5h Z1B W1b W13(W3p,W1z)fp.JcT=W5h Je8 W1b fp.JcT=W5h Z2G;if(out W3A()W4u)JbT<Out,JI2,Jax>(out,c,fp);W13(!W3p,JI2>W0p){if(Jbq&&Jbm W3A()W4u){JaG rp=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax
#define J12 W02 Jbh>(5+bs);JJN[bs]W3N;if(W34 0&&JbS>0)W1q nk=JbS<128?JbS:128,ns=1-as W3v Jd6=TA?W1n(k)*Jcl:k;Jb7 nap{1,W2q(TA?nk:m),W2q(TA?m:nk),0,W2q(Jcl),W2q(TA?nk:m),1,0};Jb7 nbp{1,W0e
#define J23 ++Jcz;if(JcH){for(Jaj i=0;i<2;++i){if(W4F i]){W1X>(i);W4F i]W5u if(Z0v[i]){W0A>(3+i);Z0v[i]W5u if(JJN[i]){W0A>(5+i);JJN[i]W5u}W4P>(JaY)W3h>(JaY);W3Z
#define J34 ){W5i{1,W2q(k),64,0,64,W2q(k),1,0}W0O(b1[bo],JcC,rp);}Jd3();W0W>(JaY)W2D>(JaY);}if(Z0v[as]){W0A>(3+as);Z0v[as]W5u W13(TA){Jan p{0,W1U>(k/16),1,0,W0e
#define J45 ;}Jan p{0,8,W2q(k/16),0,0,W4g(Jaj ki=0;ki<k/16;++ki)Jas(b2[b2o+ki W14*16],b1[bo W5C W14*k+W2N],p);W2E>(JaY)W3C>(JaY);W31;mm.m=m;mm.n=W1M.k=k;mm.JbQ=JI6;mm.Jci W0s W0b Jc0
#define J56 )W0O(b1[ns*W5O,bg[W1n(k)W1t,nbp);if(Je6){W5i{1,W2q(nk),64,0,64,W2q(nk),1,0}W0O(b1[ns*W5O,JcC[W1n(k)*64],rp);}W0W>(JbO);JbA W3N;}}W3j>(as);W4G]W3N;W1W
#define J67 1,0};Jb7 bcp{1,W2q(k),512,0,512,W2q(k),1,0};if(JbA){W4Q>(JbO);JbA W1B if(Jbp){Jd3();W0W>(JaY)W2D>(JaY);}W2a if(W4G]){W1X>(as);W4G]W5u JaU(a1[ao],JJS,acp);if
#define J78 512,W2q(k),1,0};if(JbA){W4Q>(JbO);JbA W1B if(Jbp){Jd3();W0W>(JaY)W2D>(JaY);}W2a if(W4G]){W1X>(as);W4G]W5u JaU(a1[ao],JJS,acp);if(JeR)JaU(a1[ao+W2b,low,acp)W0O(b1[bo
#define J89 JIv)W1q within=(JIX+j)%t.n1 W0R remain=t.n1-within W0R copies=JIv-j<remain?JIv-j:remain W0R JIk=(JIX+j)/t.n1-Je2;Jdf cp{1,1,W2q(nr/8),W4g(Jaj c=0;c<nr;c+=64)Copy(mult[j*nr+c],JcJ[JIk*nr+c],JaM(64),W1U>(copies),cp
#define Jhx W2e Jcj(W0B>src,Jaj JIv,Jaj nr){JIb stage1=JJ9 W5Q>();JcX()W5r(stage1,src,W1U>(JIv/8),Jbv{1,8})W0o();Jbv cp{1,W2q(nr/8)W4f(W1e=0;col<nr;W5H)Brcb(JJP[col],stage1,W1U>(JIv),cp)W0o();JcF()
#define Jiy ];JII W1E(128);W1d>gm;gm W3I(W4n*>(Jc6+32))W0O(JII,gm[Jch],128)W0b JJE=Header();W13(FIRST){W1d>headerG;headerG W3I(W4n*>(Jc6))W0O(JJE,headerG,8);}JJZ();W07 Z0Q>(Jao)W10 Z0Q>(Jao);W13(FIRST){JcE=W5b>(JJE.Jco
#define Jjz W03 Jf5(W0n dst,W0B>f,W27,W1e,Jaj nr,Jaj JIv,JI7 Z2u)W1q JIQ=JIv*nr;JJH(raw,f,W1F Jdo,JIQ)W2v();if(Z2u){JJH(JJP,raw,W1G,JIQ)W2v();Sub(JJP,f,JJP,JIQ)W2v();}JbU cp{W2q(JIv),W2q(nr/16),0,W2q((t.s2
#define JkA s2<=W0H t.s2%W1I;}W11 JaH(){Jb1 W0P!W0F W0U>=8&&r>=256&&r%256==0&&t.s2<=W0H t.s2%W1I;}W11 Jds(){Jb1 W0U>=8&&r>=W5s r%W1I&&t.s2 W5z t.s2<=W5P t.s2%W1I;}W11 Je0(){Jb1 W0P!W0F W0U<=2&&r W5z r%128
#define JlB W0o();Jat(ones,1.0f,16)W0o();Jat(Z0P[16],0.0f,48)W0o();Jbj(JbH,Z0P,0.0f,W1f GT,Jaj(64))W0o();Div(Z0P,ones,Z0P,JaQ(16))W0o();JbG(Z0P,JbH,Z0P,0.0f,W3y,16)W0o()W2m(Jfn,Jfn,W0E,16);W0i
#define JmC >&mm,Jaj Jdi,Jaj JdV);W11 JfW(){{}W25 W4t Z2X(W27,W1e,Jaj nr,Jaj JIv);W11 Jga(W27,W1e,Jaj Z1X){if(!JfW())W25 W0x(Jaj at=0;at<Z1X;at+=32){W08 Jd6=((row+at)/32*(t.s2 W47)W4O)*8;if(Je3.Jco(Jd6)!=0)W25;}W5j
#define JnD -row<Jbs?r-row:Jbs,nr=t.W4z<JIH?t.W4z:JIH,JIA=(mr+1)/2 W0R JIL=sub*JIA,end=JIL+JIA<mr?JIL+JIA:mr W3v ro=W1n(JIe)*Jd7*4+W2X W0x(Jaj JI6=JIL;JI6<end;JI6+=32)W1q JIv=end-JI6<32?end-JI6:32 W0R JIQ=JIv*nr W3v JIX=W0d
#define JoE mp.JbQ=JIG==0;mp.Jci W59(c,a2[ao],b2[bo],mp);W1W>(3 W5D);W3V]W3N;}for(Jaj JIg=0;JIg<2;++JIg)if(W3V])W0A>(3 W5D);W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=m;fp.JJz=n;fp.JbK=m;fp.JaB
#define JpF W0l();JJH(JII,ix,W1G,n)W0o();Jbj(JIJ,JII,JI2(W0V>(t.s2)),W1f LT,A)W0o();JbG(JII,JIJ,JII,W0E,W3y,A)W0o();Jbj(JIJ,JII,0.0f,W1H,A)W0o();JbG(JII,JIJ,JII,W0E,W3y,A);W0i
#define JqG pos+=JJf*4;ring W3I(W4m*>(JcU+pos));pos+=t.Jed>0?Jbi()*W1n(t.JJD)*262144ull:Jbi()*((Jbw()||(W4w)W49())?1048576ull:524288ull);JIR W3I(W4m*>(JcU+pos));pos+=JJL*4;Jee W3I(W4m*>(JcU+pos));Je3 W3I(W4n*>(JcU
#define JrH Jcx())W1q W4h(at/Jcx())*64,mr=r-row<64?r-row:64 W0R JIr=at%Jcx(),JIL=JIr*qchunk,end=JIL+qchunk<t.s2?JIL+qchunk:t.s2;mm.Je1(0u)W0x(W1e=JIL;col<end;W2U 128)W1q kr=end W4y<128?end W4y:W0D W08 wo=wb+W1n(row)*W5B W0b Z2v
#define JsI >Jcm=val?dv:dk;if(Jcm W3A()W4u)Jcm=Jcm[out];W0n Jbm=dkr;W0X Jbq=W0F!val&&dkr W3A()W4u;if(Jbq)Jbm=dkr[(out/512)*64]W0x(Jaj W4h 0;row<r;W4M 128)W1q kr=r-row<128?r-row:W0D W08 wo=wb+W1n(row)*W5B W3v qo=(W0d
#define JtJ ()*2,vi=Jaz()W0x(W27=W1n(vi)*16;row<JJL;W4M vc*16){JaU(JIf,wq W5w W1t,8192)W2v()W0x(Jaj JIr=1;JIr<Jcx();W2M){JaU(JJP,Jee[(JIr-1)*JJf W17 W1t,8192)W2v();Add(JIf,JIf,JJP,8192)W2v();}JJH(raw,JIf,W1F Jdo,8192)W2v
#define JuK W01 Jd7=(Jbw()||(W4w)W49())W53:32768 W0R JeM=(Jbw()||(W4w)W49())?32768:16384 W0R JIe=Jaz()/2,sub=Je9();W3Y=0;W0X JaJ=JbL()||Jbw()W3v Jcp=JaJ?JIU*JIe/Jbi():JIe W3v Jdj=JaJ?JIU*(JIe+1)/Jbi():JIU W0R Jcy=JaJ
#define JvL JJ9 W5Q>()[2304]W0b JbH=JJ9 W2n<Jar>()[W0K;Jat(ones,1.0f,JIv);Jat(JaI[JIv],0.0f,64-JIv)W0o();Jbj(JbH,JaI,0.0f,W1f GT,Jaj(64))W0o();Div(JaI,ones,JaI,W0V>(JIv))W0o();JbG(JaI,JbH,JaI,0.0f,W3y,JIv);W0i
#define JwM ,W3D,Jaj JJU,JI7 ovA,JI7 ovB,W0n pa,W0n pb)W44 Z2Y(Jaj JIg,W3J out,Jaj m,Jaj n,Jaj k,Jaj Z1Y,W1J,W1N,Jaj Z2d,Jaj Jbb);JaR<W40>W03 HrPairKV(W4p,W0n low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,W0n Jbm,W3s,JI7 isValue
#define JxN .JaR Jdp W4c,JHZ>(gs[wo],gs[JcW+wo],ek[(W1n(b)*W5B)W1t,W34 0?wq[(W1n(b)*W18)W1t:Jee[(JIr-1)*JJf+(W1n(b)*W18)W1t,mr,kr,t.s2,col==JIL,col+kr==end,end W4y-kr,dkr,JHZ)W1b mm.JaR Z1I W4c
#define JyO kr W3I(W2I T*>(JIo));mx W3I(W4m*>(Z1w));sm W3I(W4m*>(sums));JbB W3I(W4n*>(Z1m));dq W3I(W2I T*>(gradQuery));dk W3I(W2I T*>(gradKey));dv W3I(W2I T*>(gradValue));dqr W3I(W2I T*>(Z3v));dkr W3I(W2I T
#define JzP Jaj k,W3D,Jaj JJU,JI7 ovA,JI7 ovB,W0n pa,W0n pb)W44 SqCompute(Jaj JIg,W3J out,Jaj m,Jaj n,Jaj k,Jaj Z1Y,W1J,W1N,Jaj Z2d)W44 SqDrain();JaR<JI7 TA,JI7 TB>W03 Block(W0n ag,W5a,W3J out,Jaj m,Jaj n,Jaj k,W3D,Jaj JJU,Jaj Z1Y,W1J,W1N,JI7 reuseB
#define JAQ ()W3v Jcp=JaJ?JIU*JIe/Jbi():JIe W3v Jdj=JaJ?JIU*(JIe+1)/Jbi():JIU W0R Jcy=JaJ?1:Jbi()W0x(W33=W5g<Jdj W4d+=Jcy,++JJ2)W1q JIg=JJ2%4;Jav<2,Jc8>(W3B)W0R W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b W2s/(rt*kt)W0R mr=
#define JBR ,2048)W0o();JJH(JII,JJ8[z*W4e,W1G,2048)W0o();Jbj(JIJ,JII,JI2(W0V>(s)),W1f LT,2048)W0o();JbG(JII,JIJ,JII,W0E,W3y,2048)W0o();Jbj(JIJ,JII,0.0f,W1H,2048)W0o();JbG(JII,JIJ,JII,W0E,W3y
#define JCS W2E>(JaY)W3C>(JaY);W31;mm.m=W1M.n=W1M.k=W1M.JbQ=(W4C);mm.Jci W59(c[JIA*halfElems],a2,b2,mm);W1W>(JbM);JdM W3N;W3j>(JaY)W10 Jaw>(JaY);}}JcM();W4P>(JaY)W3h
#define JDT W2E>(JaY)W3C>(JaY);W31;mm.m=m;mm.n=n;mm.k=k;mm.JbQ=(W4C);mm.Jci W59(c,a2,b2,mm);W1W>(JbM);JdM W3N;W3j>(JaY)W10 Jaw>(JaY);}JcM();W4P>(JaY)W3h>(JaY);JbT
#define JEU W3v Jcp=JaJ?JIU*JIe/Jbi():JIe W3v Jdj=JaJ?JIU*(JIe+1)/Jbi():JIU W0R Jcy=JaJ?1:Jbi()W0x(W33=W5g<Jdj W4d+=Jcy,++JJ2)W1q JIg=JJ2%4;if(JJ2>=4)Jav<2,Jc8>(W3B+1)W0R W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b W2s/(rt*kt
#define JFX Jdr(JbZ,JHZ)[Jct?JJ5/2:JJ5],Jdr(JbZ,JIj)[Jct?JJ5/2:JJ5],JbN,JIR,mx,sm,gs,ps,t,JaM(JJ7)*16,col,Jdl,Jct);W07 JcV>(JaK)W10 JcV>(JaK);++JcK W5G;JcN(84+JJ7,1,W0V>(JcK W5G));if(JcK W5G==4)++Z1Q;Z1n W3N;JfE=Jc3<Jfp&&
#define JGY :JIH W3v qo=(W1n(b)*W18)W0y W3v ko=(W1n(b)*W5B)W0y W0R Jd7=(Jbw()||(W4w)W49())W53:32768 W3v ro=W1n(JIe)*Jd7*4+W2X;if(Jbw()){W5U(qr[(W1n(b)*W18)*64],kr[(W0d
#define J9a )/JIH W3v JIU=W1n(W0U)*rt*kt W0R JIe=Jaz();W3Y=0;JaM Jea=~JaM(0);W0X JaJ=JbL()||Jbw()W3v Jcp=JaJ?JIU*JIe/Jbi():JIe W3v Jdj=JaJ?JIU*(JIe+1)/Jbi():JIU W0R Jcy=JaJ?1:Jbi()W0x(W33=W5g<Jdj W4d+=Jcy,++JJ2)W1q JIg=JJ2
#define Jab JJH(JbW,raw,W1G,8192)W2v()W0O(raw,o[W3F,8192)W2v();JJH(JcQ,raw,W1G,8192)W2v();Mul(JbW,JbW,JcQ,8192)W2v()W0x(W4H=0;j<16;++j){Z1C(JaI,JbW[j W1t,ft,512)W2v();JaI.Jd4(16+j,JaI.Jco(0));}JIY()W0O(JIR[JI6
#define Jbc s1*t.n1;JJL=W1n(W0U)*r;Z1Z=W1n(W0U)*t.s1;rowEnd=JJL;JcW=Jbk?W1n(r)*t.s2:JJL*t.s2;JJf=JJL W0y;kelems=W1n(W0U)*t.s2 W0y W3v Jc7=W1n(W0U)*t.s1 W0R vc=Jbi()*2;Jau=Jc7<vc?W5b>(vc/Jc7
#define Jcd Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W1n(W0U)*rt*kt W0R Jd7=((W4w)W49())W53:32768 W0R JeM=((W4w)W49())?32768:16384 W0R JIe=Jaz()/2,sub=Je9();W3Y=0;W0X JaJ=JbL()W3v Jcp
#define Jde W3T>(),JIJ,JaE W3T>(),rank W3T>(),W3x JdR,n)W0o();}Jcc(JaE,JaQ(1),n)W0o();Mins(JaE,JaE,W0V>(n-1),n)W0o();JdF(JaE,JaE,JaQ(2),n)W0o()W0b JJO=JbR W3O W2T>();JJO W1E
#define Jef ,JIv)W0O(JaI,sm[JIX],JIv);if(t.n1>=4){W08 Je2=JIX/t.n1,Z2e=(JIX+JIv-1)/t.n1 W0R JJt=Z2e-Je2+1 W0x(W4H=0;j<JJt;++j)JaU(Jc2[j*nr],JbN[(Je2+j)*W5B],nr)W2v()W0b JcJ=tb W5Q>();JJH(JcJ,Jc2,W1G,JJt
#define Jfg JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W1n(W0U)*rt*kt W0R Jd7=((W4w)W49())W53:32768 W0R JeM=((W4w)W49())?32768:16384 W0R JIe=Jaz()/2,sub=Je9();W3Y=0;W0X JaJ=JbL
#define Jgh if(Z0K&&W4b Z0K)W1q JIL=(JIe-Z0K)*fill W0x(Jaj i=JIL;i<JIL+fill&&i<filled;++i)W1q JIi=JJX?(i/kvpt)*per+qpt+i%kvpt:(i/qpt)*per+i%qpt;JdX(mm,JIi);}}for(Jaj i=filled+JIe;i<sc;i+=JJJ)W1q JIi=JJX?(i/kvpt)*per+qpt+i%kvpt:(i/qpt)*per
#define Jhf W13(Z2H)Jas(a2[ao+8192+mi*k*16],a1[ao+8192+mi*k*16],p);}W2a Jay p{0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(W21=0;mi<m/W4L){Jc4(a2[ao+mi*k*16],a1[ao+W1x],p);W13(Z2H)Jc4(a2[ao+8192
#define Jhg ();Jh4(Z2f,Z15);Jh4(Z2g,Z0P);Jh4(Z2h,Jfn);JgW=JIX;Jdu W3N;}Muls(JIf,JIf,t.JJq,8192)W0o();Jhe W4c>(JIf,JIf,Z2f);Exp(JIf,JIf,8192)W0o();Jhe<W1j(JIf,JIf,Z2g);ApplyCounts16NZ(JIf,JIf,JbE);Jhe W4c>(dp,dp,Z2h);Mul(dp,dp,JIf,8192);W0i
#define Jhh val&&Jga(W1n(b)*W18,col,kr))mm.JaR Jdp<JIj,JHZ>((val?ps:gs)[wo],(val?ps:gs)[JcW+wo],(val?go:eq)[qo],Jcm,mr,kr,t.s2,row==0,W0P row+kr==r,r-row-kr,Jbm,Jbq)W1b mm.JaR Z1I<W1j((val?ps:gs)[wo],(val?ps:gs)[JcW
#define Jhi if(JfE&&Jc3<Jfp&&vid+40*Jc3<64)W1q JJ7=vid+40*Jc3;av.Z3d(t,JbB,JbN,go,o,JIR,JJ7);W07 JcV>(JaK)W10 JcV>(JaK);Z1e[Jc3]=av.Jg5();JcN(84+JJ7,0,W0V>(Z1e[Jc3]+1));++Jc3;JfE W1B JcS;}JI7 Z1n W0s W0x(Jaj JIW=0;JIW<Jc3;++JIW)
#define Jhj Je0())Jb1 W2O(Jdt())Jb1 W0D W08 JIU=W1n(W0U)*((r+63)/64)*((t.s2+255)/256);Jb1 t.s2%256==0&&JIU>=Jbi()?256:W0D}W2t JcB(){if(Jbw())Jb1 W0D{}if(JaH())Jb1 W2O(Jds())Jb1 W2O(Je0())Jb1 W2O(Jdt())Jb1 W0D W08
#define Jhk 2);W0X JJX=qu>=ku W0R lc=W0U*(JJX?qpt:kvpt),sc=W0U*(JJX?kvpt:qpt)W0R JJJ=Jbi(),JIe=Jaz(),Z0K=lc%JJJ W0R fill=(JJX?qu:ku)/(JJX?ku:qu)W0R cap=Z0K?(JJJ-Z0K)*fill:0,filled=sc<cap?sc:cap W0x(Jaj i=JIe;i<lc;i+=JJJ)W1q JIi=JJX
#define Jhl W3J JeQ,W3J Z0L,JI7 JJQ,Jaj JaB)W44 JK3(W0n qg,W3W,W5I,W0n vg,W3J JeQ,W3J Z0L,JI7 JJQ,Jaj JaB)W44 Z44(W0n qg,W3W,W5I,W0n vg,W3J JeQ,W3J Z0L,Jaj JaB)W44 JeA(W0n qg,W3W
#define Jhm for(W21=0;mi<m/W4L){Jas(a2[ao+mi*k*16],a1[ao+mi*k*16],p);Jas(a2[ao+8192+mi*k*16],a1[ao+8192+mi*k*16],p);}W2a Jay p{0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(W21=0;mi<m/W4L
#define Jhn Jbb],bcp);Jd3();W0W>(JaY)W2D>(JaY);JcM();Jay ap{0,8,W2q(m/16),0,0,JHZ,W4g(W21=0;mi<m/W4L)Jc4(a2[mi*k*16],a1[W1x],ap);W52{0,W1U>(n*k/256),1,0,0,JHZ,0};Jc4(b2
#define Jho W0b JII=tb W5Q>();JII W1E(A)W0b JJl=tb W2n<Jaj>()[A];JJl W1E(A)W0b JbX=W5S>();JbX W1E(2*Q)W0b JJA=W5S>()[W1m;JJA W1E(2*Q)W0b JIJ=fb W2n<Jar>()[W45;JIJ W1E((A/8+31)/32*32);Jat(JII,
#define Jhp JaU(b1,JcC,rp);}W0W>(JaY)W0O(a1[tail],qg[448],tcp);W0W>(JbO);}W4Q>(JaY);if(JIq==1){W0W>(Jbx)W2D>(Jbx);}JI7 W1g 2]={JHZ,JHZ W4f(Jaj JIG=0;JIG<8;++JIG)W1q JIg=JIG%2,ao=W1P 8192,bo=JIg
#define Jhq 0,0,JHZ,0};Jc4(b2[bo],b1[JIG*kp*n],bp);W2E>(JaY)W3C>(JaY);if(JIq==0&&JIG==7){W3j>(JaY)W10 Jaw>(JaY)W0O(b1,vg,bcp);W0W>(JaY)W0O(a1[tail],gog[448],tcp);W0W>(JbO);}W5K;mp
#define Jhr W2e Z45(Jaj Jdi,Jaj JdV)W1q JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W1n(JdV)*rt*kt W0R Jd7=(Jbw()||(W4w)W49())W53:32768 W0R JeM=(Jbw()||(W4w)W49())?32768:16384 W0R JIe=Jaz
#define Jhs );Jat(rank[s],JaQ(32760),s);Jcc(Jd2,0.0f,s)W0o()W0x(Jaj bit=2048;bit>0;bit/=2){Adds(JaE,rank,W0V>(bit*8),n)W0o();Jcv(JbR,JbX,JaE W3O<Jaj>(),0,n)W0o();Z1i(JIJ,JbR,Jd2,W1H,s)W0o();Z1i(JIJ
#define Jhu W2t ScoreChunk448(Jbe<T>&mm,Jaj Jdi,Jaj JdV)W1q JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W1n(JdV)*rt*kt W0R JIe=Jaz();W3Y=0;JaM Jea=~JaM(0)W3v Jcp=JIU*JIe/Jbi()W3v Jdj=JIU*(JIe+1)/Jbi()W0R Jcy
#define Jhv Mul(dst,src,broadcast,JaM(64),Jar(128),cp)W0o()W4t Jdg(W0n dst,W0B>src,Jb0<T>raw,JaM JIX,W1e,JI7 Z3O){JJH(raw,src,W1F Jdo,8192);if(Z3O){W07 Jcw>(JbM)W10 Jcw>(JbM);}W07 Jck>(JbD)W10 Jck>(JbD)W3v halfBase=((JIX
#define Jhw JbY=0;fp.W30 W13(W3p,JI2>W0p)fp.JcT=W5h Z1B W1b W13(W3p,W1z)fp.JcT=W5h Je8 W1b fp.JcT=W5h Z2G;Jcq();if(out W3A()W4u)JbT<Out,JI2,Jax>(out[JdP W1s],c,fp);W13(!W3p,JI2>W0p){if(Jbq&&JdP==0&&Jbm W3A()W4u){JaG rp
#define J03 W2Z W48=Jbl<W0k*>(W5k());W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>()W0R done=W5b>(addr[1]);JeX[W1P 4+j]=done;if(done<JcK)JcK=done;if(done>0)W1q JJt=W5b>(addr[0]-1);if(JJt>Z11)Z11=JJt;}}if(JcK>Jdn W3f]){JcN
#define J04 s2+127)W47);Jb1 JbF()==256||JIU>=Jbi()?64:32 W4t JdS(Jbe<T>&mm)W1q JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W1n(W0U)*rt*kt W0R JIe=Jaz();W3Y=0;JaM Jea=~JaM(0);W0X JaJ=JbL()||Jbw
#define J05 ;W07 JcI>(Jao)W10 JcI>(Jao);JJH(JIf,Jfo,W1G,8192);JJH(dp,Z1x,W1G,8192)W0o()W0b JbE=raw W3T>();JbE W1E(1024);JJH(JbE,grid,W1G,1024)W0o();if(Z1J){JIb JbH=grid W3O<Jar>()[W4e;JbH W1E(8)W0b ones W2C
#define J06 JaR<JI7 MULT>W03 Jhe(W0B>dst,W0B>src,W0B>JIs){JI0 Jbz cp{1,1,1,8,8,0};W13(MULT)Mul(dst,src,JIs,JaM(64),Jar(128),cp)W1b Add(dst,src,JIs,JaM(64),Jar(128),cp)W0o()W4t ApplyCounts16NZ(W0B>dst,W0B>src,W0B>JbE){JIb broadcast
#define J07 =0;pass<passes;++pass)W1q JbZ=Z2Z(JIe,pass),JId=JbZ/4 W3v qi=JaM(JbZ)*W0D JI7 Jg8 W0s W0R groupCount=(Jgs&&pass==1)?3u:4u W0x(Jaj JIk=0;JIk<groupCount;++JIk){if(!JfB(JbZ,JIk))for(Jaj child=0;child<2;++child)W1q W4x JIk W0y+child W1s W3v ki=JaM(JId)*2048 W0z W3v JJ5
#define J08 JIf,JIf,JJP,JIQ);Exp(JIf,JIf,JIQ)W0b ones=JJ9 W5Q>()[2304]W0b JbH=JJ9 W2n<Jar>()[W0K;Jat(ones,1.0f,JIv);Jat(JaI[JIv],0.0f,64-JIv);Jbj(JbH,JaI,0.0f,W1f GT,Jaj(64));Div(JaI,ones,JaI,W0V>(JIv));JbG(JaI,JbH,JaI,0.0f,W3y,JIv);Jcj(JaI
#define J09 W1p;if(W3V]){W0A>(3 W5D);W3V]W5u if(JIG==7)W4Q>(JbO)W0R src=JIG==7?tail:JIq*abase+JIG*kp*m;Jay ap{0,4,8,0,0,JHZ,W4g(W21=0;mi<8 W4K)Jc4(a2[ao+mi*kp*16],a1[src+W1x],ap);W52{0
#define J0a dp,JIf,JIQ)W2m(dp,dp,t.JJq,JIQ);W07 Jck>(JbD)W10 Jck>(JbD);Jf5(gs,dp,JIX,col,nr,JIv,JHZ);Jf5(ps,JIf,JIX,col,nr,JIv,!Z01);}Jbg<2,JeU>(W3B+1);}W2e JeY(Jaj Jdi,Jaj JdV);W11 JgN(){Jb1 r>=W5s r<=W0H t.s2>=2048;W25
#define J0b )[W0K;JII W1E(128);W1d>keyG;keyG W3I(W4n*>(Jc6+32))W0O(JII,keyG[Jch],128);W07 Z0Q>(Jao)W10 Z0Q>(Jao)W1O W1g 2]={JHZ,JHZ};Jdh<W1y()W0x(Jaj JcO=0;JcO<Jfw;++JcO)W1q JIg=JcO%2,ub=JcO%2;if(W1g ub]){W06 Jff
#define J0C go[W3F,8192)W2v();JJH(JbW,raw,W1G,8192)W2v()W0O(raw,o[W3F,8192)W2v();JJH(JcQ,raw,W1G,8192)W2v();Mul(JbW,JbW,JcQ,8192)W2v()W0x(W4H=0;j<16;++j){Z1C(JaI[j],JbW[j W1t,ft,512);}JIY()W0O(JIR[JI6],JaI
#define J0D /40+1;Jaj JcK[7]={0},JeV[7]={0},Z1e[7]={0};Jaj Jc3=0,Z1Q=0 W1O JfE W1B Jaj leaderCache[8]={0},leaderPrefix[2]={0}W0R leaderJ=vid%4,leaderBase=10*leaderJ+vid/4;W0X secondLeader=leaderBase<24;Z0H(Z1Q<Jfp||leaderPrefix[0]<4||(secondLeader&&leaderPrefix[1]<4)){Jh3(leaderBase,leaderJ,JcK,leaderCache,leaderPrefix
#define J0E /2,sub=Je9();W3Y=0 W3v Jcp=JIU*JIe/Jbi()W3v Jdj=JIU*(JIe+1)/Jbi()W0R Jcy=1 W0x(W33=W5g<Jdj W4d+=Jcy,++JJ2)W1q JIg=JJ2%t.JJD;Jav<2,Jc8>(W3B)W0R W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b=Jdi+JIi/(rt*kt)W0R mr
#define J0F );j+=copies;}W0l();W2a for(W4H=0;j<JIv;++j)JaU(Jc2[j*nr],JbN[((JIX+j)/t.n1)*W5B],nr)W2v();JJH(mult,Jc2,W1G,JIQ)W0o();}Muls(JIf,JIf,t.JJq,JIQ)W0o()W2m(ft,ft,W0E,JIv)W0o();
#define J0G (dp,dp,JIf,SPAN)W0o()W2m(dp,dp,JJq,SPAN)W0o();Jbj(slotMask,mult,0.0f,W1f GT,SPAN)W0o();JbG(JIf,slotMask,JIf,0.0f,W3y,SPAN)W0o();JbG(dp,slotMask,dp,0.0f,W3y,SPAN);JJZ();Pack(gsHigh,gsLow,dp,JJP);Pack(psHigh,psLow,JIf,JJP)W4t
#define J0H W3v qo=(W1n(b)*W18)W0y W3v ko=(W1n(b)*W5B)W0y W0R Jd7=(Jbw()||(W4w)W49())W53:32768 W3v ro=W1n(JIe)*Jd7*t.JJD+W2X;if(Jbw()){W5U(qr[(W1n(b)*W18)*64],kr[(W0d
#define J0I W08 ro=W1n(JIe)*Jd7*t.JJD+W2X W0x(Jaj JI6=JIL;JI6<end;JI6+=32)W1q JIv=end-JI6<32?end-JI6:32 W0R JIQ=JIv*nr W3v JIX=W1n(b)*W18+JI6 W0O(JIf,W2y+JI6*nr],JIQ)W0O(dp,W2y+JeM+JI6*nr],JIQ);W07 JcI>(Jao)W10
#define J0J *64],JIj,JIj);JdT.JdZ(q[qi W1t,go[qi W1t,k[ki W1t,v[ki W1t,Jdr(JbZ,JHZ)[Jct?JJ5/2:JJ5],Jdr(JbZ,JIj)[Jct?JJ5/2:JJ5],!Jg8,2048,Jdl,Jct);Jg8 W3N;}W07 Z0q>(Jao)W10 Z0q>(Jao);Jel=pass*4+JIk+1;Jdy(JIe,W0V>(Jel
#define J0K Jed;Jgp=W1n(b2)*t.s1;Z1Z=Jgp+W1n(n2)*t.s1;Z2w=W1n(b2)*r;rowEnd=Z2w+W1n(n2)*r;JJ1.Reset();JdJ();if(JdA<T,ROPE,Z01,Jdq,Jbk>(t))JeG<T>(JJ1,JbB,JbN,go,o,JIR,t,Jgp/2,Z1Z/2)W1b{Jem();Jf4();}JcD<JIj
#define J0L W03 Jej(Jbe<T>&mm,Jaj Jdi,Jaj JdV){if(JfW())W3E,Jda::JgC,JcL::Jca>(Je3)W0R rt=(r+63)/64,kt=(t.s2+63)/64 W0R qpt=rt*Jcx(),kvpt=kt*(Z01?2:1),per=qpt+kvpt W0R qu=(t.s2+Jcx()W14-1)/(Jcx()W14),ku=((r+127)/
#define J0M >=4){W08 Je2=JIX/t.n1,Z2e=(JIX+JIv-1)/t.n1 W0R JJt=Z2e-Je2+1 W0x(W4H=0;j<JJt;++j)JaU(Jc2[j*nr],JbN[(Je2+j)*W5B],nr);W07 JcI>(Jao)W10 JcI>(Jao)W0b JcJ=tb W5Q>();JJH(JcJ,Jc2,W1G,JJt*nr);Jaj
#define J0N JbW,JbW,JcQ,8192)W2v()W0b Jdk=ft;Jdk W1E(128)W0b delta16=JaI;delta16 W1E(16);JcX();Jeu(Jdk,JbW,JaQ(64),JaQ(128),JaQ(1),JaQ(1),JaQ(8))W0o();Jeu(delta16,Jdk,JaQ(8),JaQ(16),JaQ(1),JaQ(1),JaQ(1))W0o();JcF();W07 Jck
#define J0O {Jaq Jaj m=128,n=256,kp=64,abase=57344,tail=114688;Jb7 qcp{1,128,448,0,512,128,1,0};Jb7 bcp{1,256,512,0,512,256,1,0};Jb7 tcp{1,128,64,0,512,128,1,0};if(JJQ){JaU(a1,qg,qcp)W0O(a1[abase],gog,qcp);if(JK6){W5i{1,128,64,0,64,128
#define J0P W1E(64)W0b inv=W5R>()[12416];inv W1E(64)W0b ones=W5R>()[12480];ones W1E(64)W0b JbH=W2o<Jar>()[50176];JbH W1E(32)W0b JIu=W2o W2T>()[12800];JIu W1E(128)W0b countF=W5R>()[13056];countF W1E(128)W0b slotMask=W2o
#define J0Q W2Z W48=Jbl<W0k*>(W5k());W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>();Jb1 W48 W4t Jdy(Jaj id,JaQ JIa){JIb JK5=Jdd[JaM(id)*32];W2Z W48=Jbl<W0k*>(W5k());W48=JIa;Jbd<W2G>();W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<JcR
#define J0R W45;ft=fw[512];JaI=ft[512];JJ1.JaF(tb,16384);raw=tb W2n<T>();JJ1.JaF(JJ9,32768);Jc2=JJ9 W2V>()W4t PrepareBatch(Jaj JId);Z0Y:W03 JIw(Jam JIB,Jam key,Jam JIa,Jam Z1m,Jam Z1t,Jam JcP,Jam Z1w,Jam sums,Jam Jfe,Jam JIo,Jam gradQuery,Jam gradKey,Jam gradValue,Jam Z3v,Jam JgV,Jam JcU,W2j&data
#define J0S )JcD<W1j();if(Z3e()){W1q n0=W0U<t.Jed?W0U:t.Jed;Jgp=0;Z1Z=W1n(n0)*t.s1;Z2w=0;rowEnd=W1n(n0)*r;if(JdA<T,ROPE,Z01,Jdq,Jbk>(t))JeG<T>(JJ1,JbB,JbN,go,o,JIR,t,0,Z1Z/2)W1b{Jem();Jf4();}JcD<JIj
#define J0T Jcj(JaI,JIv,nr)W0o();Mul(JIf,JIf,JJP,JIQ)W0o();Mul(JIf,JIf,mult,JIQ)W0o()W2m(fw,fw,W0E,JIv)W0o();Jcj(fw,JIv,nr)W0o();Add(dp,dp,JJP,JIQ)W0o();Mul(dp,dp,JIf,JIQ)W0o()W2m(dp,dp
#define J0U ;W0X JJX=qu>=ku W0R lc=JdV*(JJX?qpt:kvpt),sc=JdV*(JJX?kvpt:qpt),JJJ=Jbi(),JIe=Jaz(),Z0K=lc%JJJ W0R fill=(JJX?qu:ku)/(JJX?ku:qu)W0R cap=Z0K?(JJJ-Z0K)*fill:0,filled=sc<cap?sc:cap W0x(Jaj i=JIe;i<lc;i+=JJJ)W1q JIi=(Jdi+i/(JJX?qpt
#define J0V W40>W03 Z1I(W4p,W0n low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,W0n Jbm,W3s);JaR<JI7 TA,JI7 Z2H W3N,W40>W03 Jdp(W4p,W0n low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,W0n Jbm,W3s
#define J0W }for(Jaj b=0;b<W0U;b+=t.Jed)W1q n=W0U-b<t.Jed?W0U-b:t.Jed;JJ1.Reset();Jei();Z45(b,n);W07 Jck>(Jcf)W10 Jck>(Jcf);Jbg<2,JeU>(12);if(b+n<W0U)W1q b2=b+n W0R n2=W0U-b2<t.Jed?W0U
#define J0X JIb JIf W2C>(0,8192),dp W2C>(32768,8192)W0b raw W1o<T>(135424,8192)W0b grid W1o W2T>(151808,1024)W0b Z2f W2C>(155904,64)W0b Z2g W2C>(157952,64)W0b Z2h W2C>(160000,64)W0b Jfn W2C>(162048,16)W0b Z15 W2C>(162304,16)W0b Z0P W2C>(162560,64)
#define J0Y ){JIb JIf=W5R>();JIf W1E(SPAN)W0b dp=W5R>()[W4e;dp W1E(SPAN)W0b JJP=W5R>()[W1m;JJP W1E(SPAN)W0b mult=W5R>()[6144];mult W1E(SPAN)W0b fw=W5R>()[12288];fw W1E(64)W0b ft=W5R>()[12352
#define J0Z W07 Jce>(JaY)W10 Jce>(JaY)W4t JK3(W0n qg,W3W,W5I,W0n vg,W3J JeQ,W3J Z0L,JI7 JJQ,Jaj JaB)W44 Z44(W0n qg,W3W,W5I,W0n vg,W3J JeQ,W3J Z0L,Jaj JaB){Jd3();JcM();Jb7 kvp{1,128
#define J10 8192),JcQ W2C>(32768,8192)W0b raw W1o<T>(65536,8192)W0b ft W2C>(81920,512),JaI W2C>(83968,32)W3v JI6=JJ7*16 W0O(raw,go[W3F,8192)W2v();JJH(JbW,raw,W1G,8192)W2v()W0O(raw,o[W3F,8192)W2v();JJH(JcQ,raw,W1G,8192)W2v
#define J11 JbE[JIk W0y+Z0S],JaM(64),Jar(8),cp)W0o()W4t PackHigh16(W0n dst,W0B>src,Jb0<T>raw,W2j&t,JaM JIX,W1e,JI7 Z3O){JJH(raw,src,W1F Jdo,8192);if(Z3O){W07 Jcw>(JbM)W10 Jcw>(JbM);}W07 Jck>(JbD)W10 Jck>(JbD);JbU
#define J0c %kvpt);JdX(mm,JIi);}if(Z0K&&W4b Z0K)W1q JIL=(JIe-Z0K)*fill W0x(Jaj i=JIL;i<JIL+fill&&i<filled;++i)W1q JIi=(Jdi+i/(JJX?kvpt:qpt))*per+(JJX?qpt+i%kvpt:i%qpt);JdX(mm,JIi);}}for(Jaj i=filled+JIe;i<sc;i+=JJJ)W1q JIi=(Jdi+i/(JJX?kvpt
#define J0d )W0x(Jaj W4h 0;row<16;++row){JI0 JaQ key=Jgc W5w];if(key>=0&&W5b>(key)<Z0T)JaU((JcO%2?vg:kg)[W1n(key)W1t,Z12 W5w W1t,512);}W07 Jff>(3+ub);W1g ub]W3N;}for(Jaj ub=0;ub<2;++ub)if(W1g ub])W06 Jff>(3+ub)
#define J0e raw,cp)W2v();}for(Jaj h=0;h<16;++h)Jat(raw[h W1t,W0c T>(0),64)W2v();}JaU(dq W5w W1t,raw,8192)W2v();}}Z0Y:W03 JIw(Jam JIB,Jam key,Jam JIa,Jam Z1m,Jam Z1t,Jam JcP,Jam Z1w,Jam sums,Jam Jfe,Jam JIo,Jam gradQuery,Jam gradKey,Jam gradValue,Jam Z3v,Jam JgV,Jam JcU,W2j&data
#define J0g W07 Z0y>(Jcf)W10 Z0y>(Jcf);}Z0Y:W2t Jg5()JI0{Jb1 Jgz W4t ZeroPack16x512(W0n gs,W0n ps,W2j&t,JaM JIX,W1e){JIb zeros W2C>(0,8192)W0b raw W1o<T>(135424,8192);Jat(zeros,0.0f,8192)W0o();PackHigh16(gs,zeros,raw,t,JIX,col,JHZ);PackHigh16(ps,zeros
#define J0h );if(JdE()){if(Jga(W1n(b)*W18,col,mr))mm.JaR Jdp W4c,JHZ>(gs[wo],gs[JcW+wo],ek[(W1n(b)*W5B)W1t,dq[(W1n(b)*W18)W1t,mr,kr,t.s2,col==JIL,col+kr==end,end W4y-kr,dkr,JHZ)W1b mm.JaR Z1I W4c
#define J0j )W0o();W2a for(W4H=0;j<n;++j){JI0 JaQ p=ix.Jco(j);if(p>=0&&W5b>(p)<t.s2)JIu.Jd4(p,JIu.Jco(p)+1);}}JIY();if(Jau>1)Jdh W2T>()W0O(JbN[JIh*t.s2],JIu,t.s2)W2v();if(Jau>1)Jcq();}W2e
#define J0k Jbo=0;fp.JbY=0;fp.W30 W2u,JI2,Jax>(JeQ,c,fp);W2u,JI2,Jax>(Z0L,c[W0K,fp);W07 Jce>(JaY)W10 Jce>(JaY)W4t JeA(W0n qg,W3W,W5I,W0n vg,W3J JeQ,W3J Z0L,Jaj m,Jaj n,Jaj JaB)W1q k=128
#define J0l W0C if(JaJ){W5U(qr[(W1n(b)*W18)*64],kr[(W1n(b)*W5B)*64],ROPE,ROPE);mm.JfL(q[qo],go[qo],k[ko],v[ko],W2y],W2y+W0K,Jea!=qo,nr);Jea=qo;W0C if(JeK()){W5U(qr,kr,JHZ,JHZ);mm.Z44
#define J0n Adds(JaE,rank,W0V>(bit*8),t.s2)W0o();Jcv(fetchedHalf,halfSorted,JaE W3O<Jaj>(),0,t.s2)W0o();Z1i(JIJ,fetchedHalf,Jd2,W1H,t.s2)W0o();JbG(rank W3T>(),JIJ,JaE W3T>(),rank W3T>(),W3x JdR,t.s2);W0i
#define J0o (W1n(b)*W5B)W1t,W34 0?wq[(W1n(b)*W18)W1t:Jee[(JIr-1)*JJf+(W1n(b)*W18)W1t,mr,kr,t.s2,col==JIL,col+kr==end,end W4y-kr,dkr,JHZ);}}W0C W1q JIk=at-rt*Jcx();W0X val=W0P JIk>=kt;mm.Je1
#define J0p W03 Z3d(W2j&t,W1d>JbB,W1d>JbN,W0n go,W0n o,W3J JIR,JaM JJ7){JIY();Jdu W0s W0R s=t.s2,n=2*s W0b JJ8 W1o W2T>(135296,4096)W0b JbX W2C>(0,16384)W0b JJA W2C>(151680,4096)W0b JIJ W1o<Jar>(184448,1024)W0O(JJ8,JbB[JJ7*W1m
#define J0q ;W2u,JI2,Jax>(JeQ[128*JaB],c[halfElems],fp);W07 Jce>(JaY)W10 Jce>(JaY)W0x(Jaj Jbb=0;Jbb<W58+=k){JaU(b1,vg[Jbb],bcp);Jd3();W0W>(JaY)W2D>(JaY);JcM();Jc4(b2,b1,bp);W3j>(JaY)W10 Jaw>(JaY
#define J0r W07 JcI>(Jao)W10 JcI>(Jao);if(Jdl){JJH(JIf,Jfo,W1G,8192);JJH(dp,Z1x,W1G,8192)W0o();}JIb JbE=raw W3T>();JbE W1E(1024);JJH(JbE,grid,W1G,1024)W0o();if(Z1J){JIb JbH=grid W3O<Jar>()[W4e;JbH W1E(8)W0b
#define J0s for(JaM JIh=Jaz();JIh<Jc7;JIh+=vc){JaU(JbN[JIh*t.s2],JIu,t.s2)W2v();}}}if Je4{W0X Jd9=W1w!W0F r W5z t.s2>=8192;W0X JaJ=JbL()||Jbw();Jbe<T>mm;if(Jbw())mm.JeF(JJ1)W1b mm.Init(JJ1,Jd9||JaJ,JaJ);{if(Z3e()){Jaj d2Iter=0 W0x(Jaj
#define J0t }if Je4{W0X Jd9=W1w!W0F r W5z t.s2>=8192;W0X JaJ=JbL();Jc9<T>mm;mm.Init(JJ1,Jd9||JaJ,JaJ);{{JdS(mm);JcD W4c>();if(JaJ){JJ1.Reset();mm.Init(JJ1);}JeT(mm);}}}W4Z{if(Jau>1)JcD<W1j();if(JdA<T,ROPE,Z01,Jdq,Jbk>(t))JeG
#define J0u W2t JbF(){{}if(JaH())Jb1 256;if(Jds())Jb1 W2O(Je0())Jb1 W2O(Jdt())Jb1 W0D W08 JIU=W1n(W0U)*((r+63)/64)*((t.s2+255)/256);Jb1 t.s2%256==0&&JIU>=Jbi()?256:W0D}W2t JcB(){{}if(JaH())Jb1 W2O(Jds())Jb1 W0D
#define J0v ;JbN W3I(W4n*>(JcU+pos));pos+=W1n(W0U)*t.s1*t.s2*4;gs W3I(W2I T*>(JcU+pos));pos+=JcW*4;ps W3I(W2I T*>(JcU+pos));pos+=JcW*4;wq W3I(W4m*>(JcU+pos));pos+=JJf*4;ring W3I(W4m*>(JcU+pos));pos+=
#define J0x W1q vc=Jbi()*2,vi=Jaz()W3v Jc7=W1n(W0U)*t.s1 W0R JIQ=(t.Jbf+Jau*8-1)/(Jau*8)*8 W0x(W33=Jgp*Jau+vi W4d<Z1Z*Jau W4d+=vc){W08 JIh W2s/Jau W0R JJa=(JIi%Jau)*JIQ W0R n=t.Jbf-JJa<JIQ?t.Jbf-JJa:JIQ W0O(ix,JbB[JIh
#define J0y Jap,JbR,W1F Jdo,t.s2)W0o();Jcv(JbR,JbX,Jap W3O<Jaj>(),0,t.s2)W0o();Z1i(JIJ,JbR,Jd2,W1H,t.s2)W0o();JbG(rank,JIJ,JaE,rank,W3x JdR,t.s2)W0o();}Jcc(JbR,1.0f,t.s2)W0o();Mins(JbR,JbR,JI2(JaO
#define J0z Jbb+=k){JaU(b1,kg[Jbb],bcp);Jd3();W0W>(JaY)W2D>(JaY);JcM();Jc4(b2,b1,bp);W3j>(JaY)W10 Jaw>(JaY)W0x(Jaj JIA=0;JIA<2;++JIA){JaU(a1,qg[Jbb+JIA*halfRows W1t,acp);W0W>(JaY)W2D>(JaY);JcM
#define J0A ((t.s2+127)W47);Jb1 JbF()==256||JIU>=Jbi()?64:32 W4t JdS(Jc9<T>&mm)W1q JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W1n(W0U)*rt*kt W0R JIe=Jaz();W3Y=0;JaM Jea=~JaM(0);W0X JaJ=JbL
#define J0B JJH(JII,JJ8,W1G,2048)W0o();Jbj(JIJ,JII,JI2(W0V>(t.s2)),W1f LT,2048)W0o();JbG(JII,JIJ,JII,-1.f,W3y,2048)W0o();Jbj(JIJ,JII,0.f,W1H,2048)W0o();JbG(JII,JIJ,JII,-1.f,W3y,2048)W0o();W0N
#define J13 W1q vc=Jbi()*2,vi=Jaz()W3v Jc7=W1n(W0U)*t.s1 W0R JIQ=(t.Jbf+Jau*8-1)/(Jau*8)*8 W0x(W33=vi W4d<Jc7*Jau W4d+=vc){W08 JIh W2s/Jau W0R JJa=(JIi%Jau)*JIQ W0R n=t.Jbf-JJa<JIQ?t.Jbf-JJa:JIQ W0O(ix,JbB[JIh*t.Jbf
#define J14 W2t Jcx(){if(t.s2>W5P r<=64)Jb1 8;if(t.s2>W5P r>=512)Jb1 2;if(r<=128&&t.s2<=W0H W0U<=2&&t.s2%512==0)Jb1 4;Jb1 r<W5s t.s2>=2048?4:1 W4t JIY(){JaN<JdL>();}W11 JbC(){{}W25 W4t Jf1()W44 Jgg()W44 JeL
#define J15 W3N;if(JcH){JcM();Z30();W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=m;fp.JJz=n;fp.JbK=m;fp.JaB=Z1Y;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 W2u,JI2,Jax>(out,Jc0,fp);W07 Jce>(JaY)W10 Jce>(JaY);}W2e
#define J16 Jaj mr=r-row<Jbs?r-row:Jbs,nr=t.W4z<JIH?t.W4z:JIH W3v qo=(W1n(b)*W18)W0y W3v ko=(W1n(b)*W5B)W0y W0R Jd7=((W4w)W49())W53:32768 W3v ro=W1n(JIe)*Jd7*4+W2X;if(JaH()){mm.JeA
#define J17 JbQ=(W4C);mm.Jci W59(c,a2,b2,mm);Mmad(c[W0K,a2[W0K,b2[W0K,mm);W1W>(JbM);JdM W3N;W3j>(JaY)W10 Jaw>(JaY);}JcM();W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=W2r.JJz=W2r.JbK=W2r.JaB
#define J18 Z2g,Z0P,Jar(2),Jbv{1,8})W5r(Z2h,Jfn,Jar(2),Jbv{1,8})W0o();JcF();JgW=JIX;Jdu W3N;}Muls(JIf,JIf,t.JJq,8192)W0o();ApplyRow16 W4c>(JIf,JIf,Z2f);Exp(JIf,JIf,8192)W0o();ApplyRow16<W1j(JIf,JIf,Z2g);ApplyCounts16(JIf,JIf,JbE);ApplyRow16 W4c
#define J19 JI2,JI2,Jax>(Z0L[128*JaB],c[halfElems],fp);W07 Jce>(JaY)W10 Jce>(JaY);Jb1;}W3Z;fp.Z0E=m;fp.JJz=n;fp.JbK=m;fp.JaB=JaB;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA W0s W0x(Jaj Jbb=0;Jbb<W58+=k){Jb7 acp{1,W2q(m),W0e
#define J1a W03 JfC(Jb0<T>cacheA1,Jb0<T>streamB1,Jb0<T>sharedA2,Jb0<T>sharedB2,W0B>sharedC){JJN=cacheA1;JJN W1E(131072);a1=streamB1;a1 W1E(32768);b1=streamB1[W45;b1 W1E(65536);a2=sharedA2;a2 W1E(32768);b2=sharedB2;b2 W1E(32768);c=sharedC;c W1E(32768)W0x(Jaj JIK=0;JIK<2;++JIK){Z1j W2z]W1B Z16
#define J1b W03 JfL(W0n qg,W3W,W5I,W0n vg,W3J JeQ,W3J Z0L,JI7 JJQ,Jaj JaB){Jaq Jaj m=128,n=128,JIq=16384,tail=114688;Jb7 full{1,128,512,0,512,128,1,0};if(JJQ){JaU(a1,qg,full)W0O(a1[W5O,gog,full);if(JK6){W5i{1,128,64,0,
#define J1D W0m,JI2,Jax>(JeQ,c,fp);W07 Jce>(JaY)W10 Jce>(JaY)W0x(Jaj Jbb=0;Jbb<W58+=k){Jb7 acp{1,W2q(m),W2q(k),0,512,W2q(m),1,0};Jb7 bcp{1,W2q(n),W2q(k),0,512,W2q(n),1,0}W0O
#define J1E Jb6 t;Z0F JJ1;W5y Z17>fb;W5y Z17>tb,ib,JJ9;W0B>JIf,dp,JJP,mult,fw,ft,JaI;Jb0<T>raw;Jb0 W2T>ix,JIu,Jc2;W0n q,k,v,go,o,qr,kr,eq,ek,gs,ps,dq,dk,dv,dqr,dkr;W3J mx,sm,scores,dps,wq,wk,wv,ring,JIR;W1d>JbB,JbN,Je3;
#define J1G JIi/2 W0R JId=JbZ/4,JI6=(JbZ%4)*2 W0x(Jaj JIA=0;JIA<2;++JIA)W1q want=W16(JI6+JIA);if(!(Jdm[JId]&want)&&JdY(20+JbZ*2+JIA)>=4)Jdm[JId]|=want;}if((Jdm[JId]&(3u<<JI6))!=(3u<<JI6))JcS;Jew(mm,JIi);JJK|=bit;W5j;}Jaj Jfi=0 W0R Z2x
#define J1H Jaj i=0;i<2;++i)if(Z2y[i])W0A>(3+i);W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=m;fp.JJz=n;fp.JbK=m;fp.JaB=JaB;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 W2u,JI2,Jax>(JeQ,c,fp);W2u,JI2,Jax>(Z0L
#define J1I JaU(Z1x,Z1q,halfCp)W0O(Z1x[256],Z1q[512],halfCp);}W2a JbU inputCp{16,64,192,0}W0O(JIf,Z2I,inputCp)W0O(dp,ringDp,inputCp);}if(Z1J){JaU(Jfn,JIR[JIX],16)W0O(Z15,mx[JIX],16)W0O(Z0P,sm[JIX],16);}W08 Je2=JIX/8 W0x(Jaj JIk=0;JIk<2;++JIk)JaU(grid[JIk
#define J1J Jbb*n;Jc4(b2[off],b1[bo],bp);W2E>(JaY)W3C>(JaY);if(seq==7){W3j>(JaY)W10 Jaw>(JaY)W0O(b1[tail],vg[448],lastV);W0W>(JbO);}W31;mm.m=m;mm.n=n;mm.k=k;mm.JbQ=(W4C);mm.Jci W0s
#define J1K =ft[512];JJ1.JaF(tb,(Jgr==4u||Jgr==5u)?16416:(JbC()?32768:16384));raw=tb W2n<T>();JJ1.JaF(JJ9,(Jgr==4u||Jgr==5u)?32768:(Jgr==3u?49152:16384));Jc2=JJ9 W2V>();JJ1.JaF(ib,JbC()?16384:(t.Jbf+t.s2)*4);ix=ib W2V>();JIu=JbC
#define J1L Z2h);Mul(dp,dp,JIf,8192)W0o()W2m(dp,dp,t.JJq,8192)W0o();PackHigh16(gs,dp,raw,t,JIX,col,JHZ);PackHigh16(ps,JIf,raw,t,JIX,col,JIj);}private:W03 Jh4(W0B>dst,W0B>src){JI0 JbU cp{1,2,0,W4g(Jaj repeat=0;repeat<4;++repeat)JaU(dst[repeat
#define J1M d2Iter)Jg7(d2Iter);d2Iter=ScoreChunk448(mm,b,n);JaN<JdL>();Jav<2,Jc8>(12);Jbg<0,JdH>(11);Jav<2,Jc8>(11);JJ1.Reset();mm.Init(JJ1);Jej(mm,b,n);if(b+n<W0U){JJ1.Reset();if(Jbw())mm.JeF(JJ1)W1b mm.Init(JJ1,JIj,JIj);}}if(d2Iter)Jg7(d2Iter
#define J1N s2)W0b rank=JIu W3T>();rank W1E(t.s2)W0b Jd2=tb W5Q>();Jd2 W1E(t.s2)W0b Jap=JJ9 W2V>();Jap W1E(t.s2);Jcc(Jd2,0.0f,t.s2)W0o()W0x(Jaj bit=Q;bit>0;bit/=2){Adds(JaE,rank,JI2(W0V>(bit)),t.s2);W0i
#define J1O ,t.s2)W0o()W2m(JbR,JbR,4.0f,t.s2)W0o();JJH(Jap,JbR,W1F Jdo,t.s2)W0o();Jcv(JbR,rank,Jap W3O<Jaj>(),0,t.s2)W2v();JbR.Jd4(t.s2-1,0.0f)W2v();Sub(JaE,rank,JbR,t.s2)W0o();JJH(JIu,JaE,W1F Jdo
#define J1P ,ap);W52{0,W1U>(k*n/256),1,0,0,JHZ,0};Jc4(b2,b1[bOff],bp);W2E>(JaY)W3C>(JaY);W31;mm.m=m;mm.n=n;mm.k=k;mm.JbQ=JI6;mm.Jci W59(Jc0,a2,b2,mm);W1W>(JbM);JdM W3N;}W3j
#define J1R mm.JdZ(q[qo],go[qo],k[ko],v[ko],W2y],W2y+W45,Jea!=qo,nr);Jea=qo;}Jbg<2,JdH>(W3B);}W5U(qr,kr,JHZ,JHZ);Jb1 JJ2 W4t Jg7(W3Y){if(JJ2==0)Jb1 W0R JJa=JJ2>t.JJD?JJ2-t.JJD:0 W0x(Jaj i=JJa;i<JJ2;++i)Jav
#define J1S JaF(tb,JbC()?32768:16384);raw=tb W2n<T>();JJ1.JaF(JJ9,16384);Jc2=JJ9 W2V>();JJ1.JaF(ib,JbC()?16384:(t.Jbf+t.s2)*4);ix=ib W2V>();JIu=JbC()?ix:ix[t.Jbf];{}if(Jau>1){Jat(JIu,0,t.s2)W2v()W0x(JaM JIh=Jaz
#define J1T W0O(b1,JcC,rp);}Jb7 firstV{1,128,384,0,512,128,1,0}W0O(b1[W5O,vg,firstV);Jd3();W0W>(JaY)W2D>(JaY);Jb7 lastV{1,128,64,0,512,128,1,0}W0O(b1[tail],vg[384],lastV);W0W>(JbO)W1O Z2y[2]={JHZ,JHZ W4f(Jaj seq
#define J1U for(Jaj Jbb=0;Jbb<W58+=128){Jb7 qp{1,128,128,0,512,128,1,0}W0O(a1,qg[Jbb],qp)W0O(a1[W0K,gog[Jbb],qp);if(W4C&&JK6){W5i{1,128,64,0,64,128,1,0}W0O(a1,Z0X,rp);}W0W>(JaY)W2D>(JaY);JcM();Jay ap
#define J1V Jaq(TA)W1q JIG=Z0f W47,bit=W16 JIG;if(!(Jfs&bit)){bcp.dstNzC0Stride=512 W0O(JJN[JaM(Z0f)*16],bg[JaM(Z0f)W0y+JdP W1s],bcp);if(Z20&&JdP==0){W5i{1,128,64,0,64,512,1,0}W0O(JJN[JaM(Z0f)*16],JcC[JaM(Z0f)*64],rp);}Jfs|=bit;}W2a JaU
#define J1W nr,JIv,!Z01);}Jbg<2,JeU>(W3B+1);}}W11 Z3e(){{if(t.Jed==0)W25;if(Jau!=1)W25;if(r%128!=0||t.s2%256!=0||t.s1%2!=0)W25 W3v per=W1n(t.Jed)*(r W47)*(t.s2/256);Jb1 per<=W1n(t.JJD)*Jbi();}W25
#define J1X JaM(Z0f)W0y+JdP W1s],bcp);if(Z20&&JdP==0){W5i{1,128,64,0,64,128,1,0}W0O(b1 W2z*W45,JcC[JaM(Z0f)*64],rp);}}W2e Z3f(){for(Jaj JIK=0;JIK<2;++JIK){if(Z1j W2z]){W1X>(JIK);Z1j W2z]W5u if(Z16 W2z]){JaS<Jal
#define J1Y 32768 W2D>(2 W5D)W0b Jc0=c[Z2d];if(n%32==0&&n>=64)W1q nH=n/2,as=sliceStep&1u,ao=as*8192;sliceStep+=1;if(Z0N[as]){W0A>(as==0?Jcf:JbD);Z0N[as]W5u Jay ap{0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(Jaj
#define J1Z ,bpH);W2E>(JaY)W3C>(JaY);JbI mh;mh.m=m;mh.n=nH;mh.k=k;mh.JbQ=JI6;mh.Jci W59(Jc0[h*m*nH],a2[ao],b2[h*(k*nH)],mh);W1W>(h==0?Jao:JaK);Z0B[h]W3N;}W1W>(as==0?Jcf:JbD);Z0N
#define J20 ];mult=f[12288];fw=f[W0K;ft=fw[512];JaI=ft[512];JJ1.JaF(tb,JbC()?32768:16384);raw=tb W2n<T>();JJ1.JaF(JJ9,16384);Jc2=JJ9 W2V>();JJ1.JaF(ib,JbC()?16384:(t.Jbf+t.s2)*4);ix=ib W2V>();JIu=JbC()?ix:ix[t.Jbf
#define J21 W03 JdX(Jc9<T>&mm,W33)W1q rt=(r+63)/64,kt=(t.s2+63)/64 W0R per=rt*Jcx()+kt*(Z01?2:1)W0R qchunk=((t.s2+Jcx()W14-1)/(Jcx()W14))*W0D W01 b W2s/per,at W2s%per W3v wb=Jbk?0ull:W1n(b)*r*t.s2;if(
#define J22 JI7 isDp=(seq<6)?JI7(seq%2):(seq>=7)W0R Jbb=(seq<6)?(seq/2)W14:(seq==8?448:384)W0R k=seq>=7?64:128,JIg=isDp?1:0,off=W1P JIq;if(Z2y W3f]){W0A>(3 W5D);Z2y W3f]W5u if(seq>=7)W4Q>(JbO);Jay ap{0,W1U
#define J1d Jcv(JJO,rank,JaE W3O<Jaj>(),0,n)W2v();JJO.Jd4(s-1,JaQ(-8));JJO.Jd4(n-1,JaQ(32760))W2v();Sub(JaE,rank,JJO,n)W0o();JdO(JaE,JaE,JaQ(3),n)W2v()W0O(JbN[(2*JJ7)*s],JaE,s)W0O(JbN[(2*JJ7+1)*s],JaE
#define J1e W0N,W1j(JbX,JII,JJl,JJA,JaQ(Q/32))W2v();JbX W1E(4*Q);Jat(JbX[2*Q],W0E,2*Q)W0o()W0b JaE=W5S>()[W2b;JaE W1E(t.s2)W0b JbR=W5S>()[12288];JbR W1E(t.s2);JIJ=fb W2n<Jar>()[W5O;JIJ W1E
#define J1f )*W18)W0y W0b Jfy=qr;if(!val)Jfy=qr[(W1n(b)*W18)*64];W5U(qr,Jfy,JHZ,W0F!val);if(val&&JgN())mm.JaR Jdp<JIj,JHZ,JHZ>(ps[wo],ps[JcW+wo],go[qo],Jcm,mr,kr,t.s2,row==0,W0P row+kr==r,r-row-kr,Jbm,Jbq)W1b if
#define J1g Jc4(a2[mi W14*16],a1[W1x],ap);Jc4(a2[16384+mi W14*16],a1[16384+W1x],ap);}W52{0,64,1,0,0,JHZ,0};Jc4(b2,b1[Jbb W14],bp);Jc4(b2[W0K,b1[65536+Jbb W14],bp);W2E>(JaY)W3C>(JaY);W31;mm.m=W1M
#define J1j Jaq Jaj m=256,n=128,kp=64,qplane=57344,tail=114688;Jb7 qcp{1,128,448,0,512,128,1,0};Jb7 kcp{1,256,512,0,512,256,1,0};Jb7 tcp{1,128,64,0,512,128,1,0};if(JJQ){JaU(b1,qg,qcp)W0O(b1[qplane],gog,qcp);if(overlayQ){W5i{1,128,64,0,64,128,
#define J1k {Jd0(Z18(T)==2,"Grad128 requires a two-byte input type");Jd0(65536+131072==196608,"DQ stream L1 byte accounting");Jd0(196608<=245760,"DQ exceeds retired ScoreB1");Jd0(512 W1s*2==262144,"Persistent B L1 byte accounting");Jd0(262144+196608<=507904,"Borrowed views exceed Score L1");Jd0(2 W14 W14*2==65536,"Grad128 L0A/B byte accounting");Jd0(128 W1s*4==131072,"Grad128 L0C byte accounting");Jb0<T>a1,b1,JJN,a2,b2;W0B>c W1O Z1j[2]={JHZ,JHZ}W1O Z16[2]={JHZ,JHZ}
#define J1l W2e JK2(Jaj id,JaQ v);W00 JaQ C8Read(Jaj id);W11 C8Panel(Jaj p,Jaj&seen)W44 Z3g(Jaj p)W44 Z3h(Jc9<T>&mm)W44 Z3i(Jc9<T>&mm)W44 Z3j()W44 JfA()W44 Jf4()W1q vc=Jbi()*2 W0b JbW=W5S>()W0b JcQ=JbW[W2b W0x(JaM JI6=JaO
#define J1m id)*32];W2Z W48=Jbl<W0k*>(W5k());W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>();JI0 JaQ JcK=addr[0];if(JcK>0)Z11=W5b>(addr[1]);Jb1 JcK;}W2t Jfk(W32){Jb1 W35 8?1u:2u;}W2t Z4m(W32,Jaj JIW){Jb1 JIe+JIW*12u W4t Jew(JcZ
#define J1n W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>();Jb1 addr[word]W4t JcN(Jaj id,Jaj word,JaQ JIa){JIb JK5=Jdd[JaM(id)*32];W2Z W48=Jbl<W0k*>(W5k());addr[word]=JIa;Jbd<W2G>();W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>()W4t JgM(Jaj
#define J1o wo],gs[JcW+wo],ek[(W1n(b)*W5B)W1t,W34 0?wq[(W1n(b)*W18)W1t:Jee[(JIr-1)*JJf+(W1n(b)*W18)W1t,mr,kr,t.s2,col==JIL,col+kr==end,end W4y-kr,dkr,JHZ);}W0C W1q JIk=at-rt*Jcx();W0X
#define J1q W00 W1d>Jeh(Jaj id){W1d>JK5;JK5 W3I(W4n*>(Jbl<Jb3 T*>(ps[JcW]W3A()))+JaM(id)*32u);Jb1 JK5 W4t Jh9(Jaj id,JaQ JIa){JIb JK5=Jeh(id);W2Z W48=Jbl<W0k*>(W5k());W48=JIa;Jbd<W2G>();W3E,Jda::Jc1,JcL::Jca>(JK5)
#define J1r ];W2Z W48=Jbl<W0k*>(W5k());W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>();addr[0]=0;addr[1]=0;Jbd<W2G>();W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>()W4t Jey(W32,Jaj Jel,Jaj*JeX,Jaj*Jdn){for(Jaj pass=0;pass<Z2z(JIe
#define J1s W08 qp=(W1n(b)*W18)*64 W3v kp=(W1n(b)*W5B)*64 W0x(Jaj c=0;c<9;++c){if(c<8)W1q JIq=c&1,ff=(c>>1)*W2O(JIq)mm.SqPrefetch(1,go[qo+ff],v[ko+ff],mr,nr,128,512,512,JHZ,JHZ,qr[qp],kr[kp
#define J1t 16384 W44 DmaReady(){W0W>(JaY)W2D>(JaY)W4t Jes(){W2E>(JaY)W3C>(JaY)W4t MathDone(JI7 releaseMte2 W3N){W1W>(JbM);W0A>(JbM);if(releaseMte2){W3j>(JaY)W10 Jaw>(JaY);}W2e MulAt(Jaj co,Jaj ao,Jaj
#define J1u 20+Z19,1,W0V>(Z11));JcN(20+Z19,0,W0V>(JcK));Jdn W3f]=JcK;}W2e Jh3(Jaj JJ5,Jaj piece,Jaj*ownPrefix,Jaj*JeX,Jaj*Jdn){if(ownPrefix[piece]>Jdn[0])Jf9(JJ5,0,JeX,Jdn);if(JJ5<24&&ownPrefix[piece+4]>Jdn[1])Jf9(JJ5+40,1,JeX,Jdn)W4t Jfq(){JeN
#define J1y JaN<JdL>();JcZ<T>Z1k;Z1k.JfC(JdT.JeI(),JdT.Jf7(),JdT.BorrowA2(),JdT.BorrowB2(),JdT.BorrowC());Z0H(JJK!=dqAll||Jcs[0]!=65535u||(Z2x==2&&Jcs[1]!=65535u)){Jgt(Z1k,JIe,JJK,Jcs,Jdm,Jcg,Z2A);}Z1k.Jd3();JaN<JdL>()W4t Jf9(Jaj Z19,Jaj JIg,Jaj*JeX,Jaj*Jdn){if
#define J1A W07 Jce>(JaY)W10 Jce>(JaY)W4t JJu(W0n qg,W3W,W5I,W3J JeQ,W3J Z0L,Jaj m,Jaj n,W1J,W1N)W44 SqPrefetch(Jaj JIg,W0n ag,W5a,Jaj m,Jaj n,Jaj k,W3D,Jaj JJU,JI7 ovA,JI7 ovB,W0n pa,W0n pb)W1q aOff=JIg
#define J1B >=W0V>(JfR)){mask|=W16 j;}}Jb1 mask;}W11 Z3w(JdG<T>&mm,W32,Jaj Z21,Jaj&JJK,JaM*Jcs,Jaj*Jdm,Jaj*Jcg,JI7 JeJ,JaM JgL=0,JI7 Jg6 W0s){if(!JeJ){for(Jaj JIW=0;JIW<Z21;++JIW)W1q bits=3u<<(JIW*2);if((JJK&bits)==bits)JcS W0R JbZ=SpatialDQOwnerRow(JIe,JIW)W0R
#define J1C W0b JbW=W4T>(a,65536,4096),JcQ=W4T>(a,81920,4096);W5c encoded;encoded W3I(W2I JIA*>(Jbl<W0k*>(JbN W3A())));W1d>JJe;JJe W3I(W4n*>(tokenFlags));W09 JaM>tr;if(Jek)tr W3I(W2I JaM*>(Jek))W0b initial=JaL W2T>(a,98304,2048)W0b initialGO=JaL
#define J24 T>(135424,8192);Jat(zeros,0.0f,8192)W0o();Jdg(gs,zeros,raw,JIX,col,JHZ);Jdg(ps,zeros,raw,JIX,col,JIj)W4t Jh1(W3J Z2I,W3J ringDp,W1d>JbN,W3J JIR,W3J mx,W3J sm,W0n gs,W0n ps,W2j&t,JaM JIX,W1e,JI7 Jdl W0s,JI7 Jct W0s
#define J25 JcT=W5h Je8 W0b JcP=JIq?od:os;W5c Jfj;Jfj W3I(W2I JIA*>(Jbl<W1Y*>(JcP W3A())));JbT<JIA,JI2,CFG_NZ>(Jfj,c,fp);W07 Jce>(JaY)W10 Jce>(JaY);}W2e Jh0(W0n qg,W3W,W5I,W0n vg,W0n krg,W3J os,Jak
#define J26 40)Jdy(id,0)W0x(Jaj JJ7=Jaz();JJ7<256;JJ7+=40)JgM(JJ7);}JcD W4c>();if Je4{{if(((t.JbP>>20)&3u)==3u)JeW()W1b Jfv();}}W4Z{{if(((t.JbP>>20)&3u)==3u)Jfq()W1b Jfu();}}JcD W4c>();W4Z{JdI();}}W11 JdE(){{}W25
#define J27 ;Jey(JIe,Jel,JeX,Jdn);}if(Jgs&&pass==1){W07 Z0q>(Jao)W10 Z0q>(Jao);Jel=pass*4+4;Jdy(JIe,W0V>(Jel));Jey(JIe,Jel,JeX,Jdn);}}JaN<JdL>()W1O Z2J W1B Z0H(!Z2J||JJK!=dqAll||Jcs[0]!=kvAll||Jcs[1]!=kvAll){Jey(JIe,Jel,JeX,Jdn);Z2J W3N W0x
#define J28 255u)!=255u)JcS W0x(Jaj Jbb=0;Jbb<2;++Jbb){if(Jcs[Jbb]&Z1R)JcS;if(JeJ&&Jcu()-JgL>=3200)W25;Z46(mm,JHZ,Z0G>=16,b,(Z0G%16)W14,Jbb,Jg6&&!(Jcg[Z31]&256u));Jcs[Jbb]|=Z1R;if(JeJ)mm.Jd3();W5j;}}W25 W4t RunSpatialIC()W44 JdI()W1q vid=Jaz
#define J29 fp.Jbo=0;fp.JbY=0;fp.W30 W13(W3p,JI2>W0p)fp.JcT=W5h Z1B W1b W13(W3p,W1z)fp.JcT=W5h Je8 W1b fp.JcT=W5h Z2G;W13(DIRECT_DQ){Jd0(!TA&&W3p,T>W0p,"H-F2 direct DQ is non-transposed and final dtype only");if(out W3A()W4u){if(dqFeatureZero){JIb tail=fp;tail.JJz=192;JbT<Out,JI2,Jax>(out
#define J2a )W1b mm.SqPrefetch(0,q[qo+ff],k[ko+ff],mr,nr,128,512,512,W0F ff==0,W0F ff==0,qr[qp],kr[kp]);}if(c>0)W1q p=c-1,JIq=p&1,ff=(p>>1)*W2O(JIq)mm.SqCompute(1,W2y+W0K,mr,nr,128,nr,ff==0,ff==384,16384)W1b mm.SqCompute
#define J2b }W01 maxParts=t.Jbf/256;if(Jau>maxParts)Jau=maxParts;if(Jau==0)Jau=1;if(JbC())Jau=2;q W3I(W2I T*>(JIB));k W3I(W2I T*>(key));v W3I(W2I T*>(JIa));go W3I(W2I T*>(Z1t));o W3I(W2I T*>(JcP));qr W3I(W2I T*>(Jfe
#define J2c W06 JcV>(JaK);Z1e[Jc3]=av.Jg5();JcN(84+JJ7,0,W0V>(Z1e[Jc3]+1));++Jc3;}}JaN<JdL>()W4t Z46(JdG<T>&mm,JI7 JIB,JI7 JIa,Jaj JId,Jaj row,Jaj Jbb,JI7 Z32 W0s){if(JIB){W08 qi=JaM(JId)W0y W17 W3v ki=JaM(JId)*2048;Jaj Z1K=4,Z1L=4;JdU
#define J2d ==0?Z1M:Jfl)[W1n(key)W1t,512);}JJZ()W0O((JIq==0?Jf0:Jen)[W3F,raw,8192);JJZ();}W2e WeightsTileCached(Jam Jc6,Jaj JaD,Jaj Jch,JI2 JJq,W3J scoreG,W3J dpG,W3J Z1w,W3J sums,W3J JIR,W0n gsHigh,W0n gsLow,W0n psHigh,W0n psLow,JI7 JfM
#define J2E (JdM){W0A>(JbM);JdM W5u}Z0Y:W03 Jd3(){if(Jbp){W06 Jce>(JaY);Jbp W5u W2e Jcn(W0n a,W0n b,JI7 pa,JI7 pb){Z0X=a;JcC=b;JK6=pa;Je6=pb;}Jaj JeR=1 W44 Je1(Jaj g){JeR=g W4t Init(Z0F&p,JI7 Jd9 W0s,JI7 JaJ W0s
#define J2G JJ8,W1G,JaD)W0o();Jbj(mask,JII,0.0f,W1H,JaD)W0o();JbG(JII,mask,JII,W0E,W3y,JaD)W0o();Jbj(mask,JII,JI2(W0V>(Z0T)),W1f LT,JaD)W0o();JbG(JII,mask,JII,W0E,W3y,JaD)W0o()W0b JJl=JJ8 W3O<Jaj
#define J2H <passes*2;++JIg)Z2J=Z2J&&Jdn W3f]>=4;if(JJK!=dqAll||Jcs[0]!=kvAll||Jcs[1]!=kvAll)Z3w(Z1k,JIe,Z21,JJK,Jcs,Jdm,Jcg,JHZ,0,JIj);}Z1k.Jd3();JaN<JdL>();}W11 Jfz(){Jb1 Z3x()&&((t.JbP>>20)&3u)==3u&&((t.JbP>>22)&1u)W0S;W25;}W2t JgD(W32
#define J2I W02 Z0y>(Jcf)W10 Z0y>(Jcf);Jat(halves,W0c T>(0),8192);W07 Jck>(JbD)W10 Jck>(JbD);JbU scatter{128,4,0,28}W0O(dq[qi W1t,halves,scatter);JaN<JdL>();}W00 JaQ Z3P(Jaj id,Jaj word){JIb JK5=Jdd[JaM(id)*32];W2Z W48=Jbl<W0k*>(W5k
#define J2J ;JcN(84+JJ7,0,W0V>(Z1e[Jc3]+1));++Jc3;}}JaN<JdL>()W4t Jfv(){W0X Jgs=((t.JbP>>22)&1u)!=0;W0X Jdl=((t.JbP>>20)&1u)!=0;W0X Jct=Jdl&&((t.JbP>>21)&1u)!=0 W0R JIe=Jaz(),passes=Z2z(JIe)W0R Z21=Z47(JIe),kvCount=Z3Q(JIe
#define J2K W02 Jck>(JaK)W0O(JIR[JaM(JIh)*8],delta8,8);W07 JcV>(JaK)W10 JcV>(JaK);if(Jek)tr.Jd4(tw+3,Jcu())W0b JK5=JJe[JaM(100+JIh)*32];W2Z*dst=Jbl<W0k*>(W5k());*dst=1;Jbd<W2G>();W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd
#define J2L W0g W2T>(),JaQ(0),80)W0o();Z1i(run,JII,nextKey,W1f NE,JaD)W0o();Jbj(validMask,JII,0.0f,W1H,JaD)W0o();And(run W3O<Jb5>(),run W3O<Jb5>(),validMask W3O<Jb5>(),JaD/16)W0o()W0R Z0u=JaD/8 W0b pattern=W2o<Jaj
#define J2M (JJK&Z1R)JcS;if(JeJ&&Jcu()-JgL>=3200)W25;Z46(mm,JIj,JHZ,b,(JbZ%4)W14,Jbb);JJK|=Z1R;if(JeJ)mm.Jd3();W5j;}}}W01 kvCount=Z3Q(JIe);Jaj Jfi=0 W0x(Jaj JIW=0;JIW<kvCount;++JIW){W08 Z1R=JaM(1)<<JIW;if((Jcs[0]&Jcs[1]&Z1R)==Z1R)JcS W0R JIi=SpatialKVTask
#define J2N >(Jao)W10 V_S>(Jao)W0R u=W5b>(GetGatherMaskRemainCount());JJZ()W0R JIJ=u?W5b>(ends.Jco(u-1)):0;JJH(unique,compact,W1F Jdo,JaD)W0o();Jcc(JJ8,JaQ(7),JaD)W0o();JdF(JJ8,JJ8,JaQ(2),JaD)W0o();Jcv(JIu,allEnds,JJ8 W3O<Jaj
#define J2S W0g<Jaj>(),off W3O<Jaj>(),0,Z0u)W0o()W0b compact=JbX;compact W1E(JaD)W0b allEnds=W2o W2T>()[W2b;allEnds W1E(JaD+8)W0b ends=allEnds[8];ends W1E(JaD);Jat(compact,W0E,JaD);Jat(allEnds,JaQ(0),JaD+8);Jcc(JJ8,JaQ(1),JaD)W0o();JI0 JdW gp
#define J2T W0o();Jcv(JbR,JbX,JaE W3O<Jaj>(),0,t.s2)W0o();Z1i(JIJ,JbR,JII,W1H,t.s2)W0o();JbG(JIu W3T>(),JIJ,JaE W3T>(),JIu W3T>(),W3x JdR,t.s2)W0o();}Jcc(Jap,JaQ(1),t.s2);JaN
#define J2V .JbP>>20)&1u)!=0;W0X Jct=Jdl&&((t.JbP>>21)&1u)!=0 W0R JIe=Jaz(),passes=Z2z(JIe);Jaj Jel=0;Jaj Jdm[8]={0},Jcg[32]={0};Jaj JJK=0,Jcs[2]={0,0},Z2A=32 W0R dqAll=W35 8?255u:0u W0R Z2x=Jfk(JIe);JeH<T>JdT;JdT.JeF(JJ1)W0x(Jaj pass
#define J30 Jfm,2048,!JIa,!JIa,Jbb);}}W2t Jex(Jaj JId,Jaj mask,Jaj JfR,JI7 Jg6 W0s){for(W4H=0;j<8;++j){if(mask&(W16 j))JcS;if(Jg6){Jaj Z11=4;JI0 JaQ JcK=JdU(20+JId*8+j,Z11);if(JcK>=W0V>(JfR)){mask|=W16 j;if(JcK>4||Z11>=JfR)mask|=256u;}W0C if(JdY
#define J33 ;Jcv(JII,JbX,Jap W3O<Jaj>(),0,2048)W0o();JJH(code,JII,W1F CAST_TRUNC,2048)W0o();Adds(code,code,JdD(16384),2048)W0o();Jat(code[W4e,JdD(16383),1024)W0o();W07 Jck>(Jao)W10 Jck>(Jao)W0O(encoded[JaM(JIh)*3072],code W3O
#define J2e JJ9 W5Q>()[512]W0b JbH=JJ9 W2n<Jar>()[W1m;Jat(ones,1.f,64)W0o();Jbj(JbH,JaI,0.f,W1f GT,Jaj(64))W0o();Div(JaI,ones,JaI,JaQ(64))W0o();JbG(JaI,JbH,JaI,0.f,W3y,64)W0o();C9QueuedRows128<W1j(JIf,JaI);C9QueuedCounts128(JIf
#define J2h p.JcT=W5h Je8;p.W30 p.JfN W1B p.unitFlag=0 W0x(JaM at=W1n(cid)*32768;at<elements;at+=W1n(JJJ)*32768){JbT<JIA,JI2,Jax>(dk[at],c,p);JbT<JIA,JI2,Jax>(dv[at],c,p);}EndFix();W07 Z0q>(Jao)W10 Z0q>(Jao)W4t C4CapturePBeforeNextScore(Jak
#define J2k Jar>()[W5O;slotMask W1E(256)W0b maxCache=W5R>()[18432];maxCache W1E(SPAN)W0b invCache=W5R>()[20480];invCache W1E(SPAN)W0b deltaCache=W5R>()[22528];deltaCache W1E(SPAN);W1d>countG;countG W3I(W4n*>(Jc6+32+4*JaD))W0O(JIf,scoreG[sub*SPAN],SPAN)W0O(dp
#define J2l JIW,JbZ=JJ7/8 W0R producer=JbZ<20?JbZ:JbZ-12 W0R pass=JbZ<20?0:1,JIk=JcK W5G W0R JfR=pass*4+JIk+1;if(JeV W5G<JfR)JeV W5G=W5b>(JdY(producer));if(JeV W5G>=JfR)W1q W4x JIk W0y W3v JJ5=JaM(JIk)*65536+JaM(JJ7%8)*4096;if(JIk>=Z1e W5G)av.Jgk
#define J2m JJS,16384);W2a JaU(Jf0[W3F,JJS,16384)W0O(Jen[W3F,low,16384);}W07 JcV>(ev);W1g b]W3N;}for(Jaj b=0;b<2u;++b)if(W1g b])W06 JcV>(gEv[b]);}JaR<JI7 FIRST>W03 Z48(Jam Jc6,Jaj Jch,Jaj Z0T,W0n Z1M,W0n Jfl,W0n Jf0,Jak
#define J2p ){Jd0(W3p,JI2>W0p||W3p,T>W0p,"C2PairFullWide128 output must be FP32 scratch or the input dtype")W0R as=Jcz%2,ao=as W1p,bo=as*32768;Jb7 acp{1,W2q(TA?k:m),W2q(TA?m:k),0,W2q(Jcl),W2q(TA?k:m),1,0};Jb7 bcp{1,W2q(k),256,0,512,W2q(k),
#define J2q JcK W5G>=4)JcS W0R JJ7=vid+40*JIW,JbZ=JJ7/8 W0R producer=JbZ<20?JbZ:JbZ-12 W0R pass=JbZ<20?0:1,JIk=JcK W5G W0R JfR=pass*4+JIk+1;if(JeV W5G<JfR)JeV W5G=W5b>(JdY(producer));if(JeV W5G<JfR)JcS W0R W4x JIk W0y W3v JJ5=JaM(JJ7%8)*16*2048
#define J2r *64],(JIa?dv:dk)[ki W1t,dkr[ki*64],Z32?128:512,2048,!JIa,!JIa,Jbb,tag);}W11 Jgt(JcZ<T>&mm,W32,Jaj&JJK,Jaj*Jcs,Jaj*Jdm,Jaj*Jcg,Jaj&Z2A){if(W35 8)for(Jaj JIW=0;JIW<8;++JIW)W1q bit=W16 JIW;if(JJK&bit)JcS W0R JIi=JIe+JIW*8,JbZ
#define J2u ,W1G,Q)W0o();Jat(JJl,Jaj(0),Q)W0o();Jbj(JIJ,JII,JI2(W0V>(t.s2)),W1f LT,Q)W0o();JbG(JII,JIJ,JII,W0E,W3y,Q)W0o();Jbj(JIJ,JII,0.0f,W1H,Q)W0o();JbG(JII,JIJ,JII,W0E,W3y
#define J2y 64+64;JI6+=16)W1q b=Z0I++&1u;JI0 Jb5 ev=W2q(4u+b);if(W1g b]){W06 JcV>(ev);W1g b]W5u JIb raw=W2o<T>()[b?16384:0];raw W1E(8192)W0R Jep=Jch+JI6 W0R JbS=JcE>Jep?JcE-Jep:0 W0R JJL=JbS<16?JbS:16;JaQ JeZ[16]W0x(Jaj
#define J2z W20],JJS[col+128u],acp);W0W>(JbO);}W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=W2r.JJz=256;fp.JbK=W2r.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JcT=W5h Je8;if(out W3A()W4u)JbT<T,JI2,Jax>(out[W0d
#define J2A initialOut=JaL<T>(a,114688,4096)W0O(initial,JbB[JaM(JI6)*W4e,2048)W0O(initialGO,go[JaM(JI6)*W1m,4096)W0O(initialOut,o[JaM(JI6)*W1m,4096);W07 JcI>(JaY);Jat(JJl,Jaj(0),2048)W0o();Jcc(Jap,JaQ(0),2048)W0o();JdF(Jap,Jap,JaQ(3),2048);JaN
#define J2B Jd4(tw+2,Jcu())W0b rawGO=JaL<T>(a,JcO+8192,4096),rawOut=JaL<T>(a,JcO+16384,4096);JJH(JbW,rawGO,W1G,4096)W0o();JJH(JcQ,rawOut,W1G,4096)W0o();W07 Jcw>(s);packetRead[s]W3N;Mul(JbW,JbW,JcQ,4096)W0o();JcX();Jeu(Jdk,JbW
#define J2C fp.Z0E=m;fp.JJz=256;fp.JbK=m;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 W13(W3p,JI2>W0p)fp.JcT=W5h Z1B W1b W13(W3p,W1z)fp.JcT=W5h Je8 W1b fp.JcT=W5h Z2G;if(out W3A()W4u)JbT<Out,JI2,Jax>(out,c,fp);W07
#define J35 W06 JcV>(nev);W1g nb]W5u W01 Jhb=Jch+nfirst,nRemaining=JcE>Jhb?JcE-Jhb:0,nRows=nRemaining<32?nRemaining:32;JaQ nKeys[32]W0x(W4H=0;j<nRows;++j)nKeys[j]=JII.Jco(nfirst+j)W0b nlow=W2o<T>()[nbase];nlow W1E(32768)W0b nhigh=W2o<T>()[nbase+W0K;nhigh W1E(16384
#define J36 bo],p)W4t JfZ(){W4P>(JaY)W3h>(JaY)W4t Write(W3J out,Jaj co,Jaj n){JaG p;p.Z0E=32;p.JJz=n;p.JbK=32;p.JaB=n;p.JJg=1;p.Jbo=0;p.JbY=0;p.W30 W2u,JI2,Jax>(out,c[co],p)W4t EndFix
#define J37 W0l();Jbj(JbH,inv,0.0f,W1f GT,Jaj(64))W0o();Div(inv,ones,inv,JaQ(16))W0o();JbG(inv,JbH,inv,0.0f,W3y,16)W0o();Jg9(inv,invCache);}Mul(JIf,JIf,invCache,SPAN)W0o();Mul(JIf,JIf,mult,SPAN)W0o();if(!JfM){Muls
#define J38 Jaj kOff=Jfa?0u:8192u W0R vOff=Jfa?8192u:0u;Jaj W4h 0;Z0H(row<16u){JI0 JaQ key=Jgc W5w];if(key<0||W5b>(key)>=Z0T){++row;JcS;}Jaj run=1u;Z0H(row+run<16u&&Jgc W5w+run]==key+W0V>(run)&&W5b>(key+W0V>(run))<Z0T)++run;if(run==1u){Jeo(dst[W0d
#define J39 h2GBusy W39]){W0A>(4 W5C);h2GBusy W39]W5u JI0 Jan bp{0,8,8,0,0,W4g(Jaj ki=0;ki<8;++ki)Jas(b2[b2o+ki*W4e,b1[bo W5C*JIq+W2N],bp);W2E>(JaY)W3C>(JaY);if(JI6&&JfU){W06 Z0U>(Jbx);JfU W5u W31
#define J3a W03 Jf1()W1q Q=4096,vc=Jbi()*2 W3v Jc7=W1n(W0U)*t.s1 W0b JII=tb W5Q>();JII W1E(Q)W0b JJ8=tb W2V>()[Q];JJ8 W1E(Q)W0b JJl=JJ8 W3O<Jaj>();JJl W1E(Q)W0b JbX=W5S>();JbX W1E(2*Q)W0b JJA=fb
#define J3b JIb JJ8=W2o W2T>()[W4e;JJ8 W1E(JaD)W0b JbX=W5R>()[W1m;JbX W1E(2*JaD)W0b JJA=W5R>()[W2b;JJA W1E(2*JaD)W0b unique=W2o W2T>()[12288];unique W1E(JaD)W0b JIu=W2o W2T>()[14336];JIu W1E(JaD)W0b mask=W2o<Jar
#define J3d W4Q>(JbO);Jd3();W0W>(JaY)W2D>(JaY);if(amath W2z]){W0A>(3u+JIK);amath W2z]W5u Jan ap{0,8,1,0,7,W4g(W21=0;mi<8u W4K)Jas(a2[ao+mi W1k],a1[ao+mi W1k],ap);W2E>(JaY);JaS<Jal
#define J3e JaU(a1,prefixK,rp);}W0W>(JaY)W0O(b1[tail],qg[448],tcp);W0W>(JbO);}W4Q>(JaY);if(JIq==1){W0W>(Jbx)W2D>(Jbx);}JI7 W1g 2]={JHZ,JHZ W4f(Jaj JIG=0;JIG<8;++JIG)W1q JIg=JIG%2,ao=JIg W1p,bo=JIg
#define J3F Z3y);Jb1 raw W4t Jfu(){JeN<T>av;av.Init(JJ1);W0X Jdl=((t.JbP>>20)&1u)!=0;W0X Jct=Jdl&&((t.JbP>>21)&1u)!=0 W0R vid=Jaz(),Jfp=(255-vid)/40+1;Jaj JcK[7]={0},JeV[7]={0},Z1e[7]={0};Jaj Jc3=0,Z1Q=0 W1O JfE W1B Z0H
#define J3H 128<Jfm){CopyK<TA>(JJS,bg,JcC,Z0f+128,1-as,Jcl,Z20,JdP);W0W>(JbO);}}W3j>(as);Z1j[as]W3N;W1W>(3+as);Z16[as]W3N;}Z3f();W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=W2r.JJz=256;fp.JbK=W2r.JaB
#define J3I PrepareDeltaToken(W0n Z1t,W0n JcP,W3J JIR){JIb JbW=W5R>();JbW W1E(8192)W0b JcQ=W5R>()[W2b;JcQ W1E(8192)W0b raw=W2o<T>()[W45;raw W1E(8192)W0b JaI=W5R>()[20480];JaI W1E(64)W0O(raw,Z1t[sub*W2b,8192);JJZ();
#define J3K W3E,Jda::Jc1,JcL::Jca>(a);}Jbd<W2G>()W0x(W4H=0;j<N;++j)if(Jcr&(W16 j))W1q t0=tile+j W4A,id=t0*2u W0b a=st.Jg1[(id/2u)W0w]W0b b=st.Jg1[(id/2u)W0w+8u];W2Z*p0=Jbl<W0k*>(a W3A());W2Z*p1=Jbl
#define J3M W0b JbX=W4T>(a,0,4096),JJA=W4T>(a,16384,4096)W0b JII=W4T>(a,32768,2048)W0b JJl=JaL<Jaj>(a,40960,2048)W0b Jap=JaL W2T>(a,49152,2048)W0b JIJ=JaL<Jar>(a,57344,1024)W0b Jdk=W4T>(a,58368,64),delta8=W4T>(a,58624,8)W0b code=JaL<JdD>(a
#define J3N }Jcc(Jap,JaQ(1),t.s2)W0o();Mins(Jap,Jap,W0V>(t.s2-1),t.s2)W0o();JdF(Jap,Jap,JaQ(2),t.s2)W0o()W0b Jet=JbR W3O W2T>();Jet W1E(t.s2);Jcv(Jet,rank,Jap W3O<Jaj>(),0,t.s2)W2v();Jet.Jd4(
#define J3O W01 src=JIG==7?tail:JIq*qplane+JIG*kp*n;W52{0,32,1,0,0,JHZ,0};Jc4(b2[bo],b1[src],bp);W2E>(JaY)W3C>(JaY);if(JIq==0&&JIG==7){W3j>(JaY)W10 Jaw>(JaY)W0O(a1,vg,kcp);W0W>(JaY)W0O(b1[tail
#define J3R ];JI0 Jb5 ev=gEv[b];if(JIk+1u<4u)W1q nfirst=sub*64+(JIk+1u)W4A,nb=(JIk+1u)%3u,nbase=gBase[nb];JI0 Jb5 nev=gEv[nb];if(W1g nb]){W06 JcV>(nev);W1g nb]W5u W01 Jhb=Jch+nfirst,nRemaining=JcE>Jhb?JcE-Jhb:0,nRows=nRemaining<16?nRemaining:16;JaQ nKeys[16]W0x(W4H
#define J3S JaL<T>(a,65536,8192)W0b ft=W4T>(a,81920,512),JaI=W4T>(a,83968,32)W3v JI6=JJ7*16 W0O(raw,go[W3F,8192);Jdb();JJH(JbW,raw,W1G,8192);Jdb()W0O(raw,o[W3F,8192);Jdb();JJH(JcQ,raw,W1G,8192);Jdb();Mul(JbW,JbW,JcQ,8192
#define J3T *8192,bOff=W1P 32768;Jb7 acp{1,W2q(m),W2q(k),0,W2q(Jcl),W2q(m),1,0};Jb7 bcp{1,W2q(n),W2q(k),0,W2q(JJU),W2q(n),1,0};if(Jfr W3f]){W1X>(2 W5D);Jfr W3f]W5u JaU(a1
#define J3V );if(Jbq&&feat W0S&&Jbm W3A()W4u){JaG rp=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax>(Jbm[W1n(col)W2R],c,rp);}W07 Jce>(JaY);Jbp W3N;}for(Jaj JIK=0;JIK<2u;++JIK){if(l1 W2z])W1X>(JIK);if(amath W2z])W0A>(3u+JIK
#define J3X ];JI0 Jb5 nev=dEv[nb];if(W1g nb]){W06 Jff>(nev);W1g nb]W5u Jav<2,Jc8>(8);Jav<2,Jc8>(10)W0b nlow=W2o<W1y()[nbase];nlow W1E(16384)W0b nhigh=W2o<W1y()[nbase+W2b;nhigh W1E(8192)W0b nkraw=Jfa?nlow:nhigh W0b nvraw=Jfa?nhigh:nlow W0O(nkraw,ring
#define J3Y dqFeatureZero&&Jbm W3A()W4u){JIb rp=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax>(Jbm,c,rp);}W2a if(out W3A()W4u)JbT<Out,JI2,Jax>(out,c,fp);W13(!W3p,JI2>W0p){if(Jbq&&Jbm W3A()W4u){JIb rp=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax>(Jbm
#define J3Z ;W1d>countG;countG W3I(W4n*>(Jc6+32+4*JaD))W0O(JIf,scoreG[sub*SPAN],SPAN)W0O(dp,dpG[sub*SPAN],SPAN)W0O(fw,JIR[sub*16],16)W0O(ft,Z1w[sub*16],16)W0O(inv,sums[sub*16],16)W0O(JIu,countG[Jch],128);JJZ();JJH(countF,JIu,W1G,128);JaN
#define J40 W1y rh;rh W3I(W2I::JIA*>(Jbl<W1Y*>(ring W3A())))W3v hs=ro*2ull+JI6*nr W3v hd=(ro+JeM)*2ull+JI6*nr W0O(raw,rh[hs],JIQ)W2v();JJH(JIf,raw,W1G,JIQ)W0o()W0O(raw,rh[hd],JIQ)W2v();JJH(dp,raw,W1G,JIQ);W0i
#define J42 -row<Jbs?r-row:Jbs,nr=t.W4z<JIH?t.W4z:JIH,JIA=(mr+1)/2 W0R JIL=sub*JIA,end=JIL+JIA<mr?JIL+JIA:mr W3v ro=W1n(JIe)*Jd7*4+W2X W0x(Jaj JI6=JIL;JI6<end;JI6+=64)W1q JIv=end-JI6<64?end-JI6:64 W0R JIQ=JIv*nr W3v JIX=W0d
#define J43 ;JIJ W1E(512)W0b JaE=fb W2V>()[W0K;JaE W1E(t.s2)W0b JbR=W5S>()[20480];JbR W1E(t.s2)W0b Jap=JJ9 W2V>();Jap W1E(t.s2);JIu W1E(t.s2)W0x(W33=Jaz()W4d<Jc7*2 W4d+=vc){W08 JIh W2s/2 W0O(JJ8,JbB[JIh*t.Jbf
#define J44 ){Jdb();JJ1.Reset();W5y Z17>a;JJ1.JaF(a,185472)W0R s=t.s2,n=2*s,vc=Jbi()*2 W3v Jc7=W1n(W0U)*t.s1 W0b JJ8=JaL W2T>(a,135296,4096)W0b JbX=W4T>(a,0,16384)W0b JJA=W4T>(a,151680,4096)W0b JIJ=JaL<Jar>(a,184448,1024)
#define J3f ;}W01 bo=JIa?BV:0;Jes();MulAt(0,0,bo,512,32,JIj);MathDone((pipeMode&4u)W0S);if(reuse)Jav<2,JdH>(Z2K);JfZ();W3Z;fp.Z0E=32;fp.JJz=W4Y.JbK=32;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcT=W5h Je8;fp.W30 fp.JfN W1B fp.unitFlag
#define J3g W4H=0;j<n;++j){JI0 JaQ p=ix.Jco(j);if(p>=0&&W5b>(p)<t.s2)JIu.Jd4(p,JIu.Jco(p)+1);}}JIY();if(Jau>1)Jdh W2T>()W0O(JbN[JIh*t.s2],JIu,t.s2)W2v();if(Jau>1)Jcq();}W2e JeL()W44 JdB()W44
#define J3h W0o();JdF(Jap,Jap,JaQ(2),t.s2)W0o()W0b JJO=JbR W3O W2T>();JJO W1E(t.s2);Jcv(JJO,JIu,Jap W3O<Jaj>(),0,t.s2)W2v();JJO.Jd4(t.s2-1,JaQ(-8))W2v();Sub(JaE,JIu,JJO,t.s2)W0o();JdO(JIu,JaE,JaQ
#define J3i ,2048)W0o();W0N,W1j(dst,JII,JJl,JJA,JaQ(64));Jdb();Jat(JbX[z*8192+W1m,W0E,4096)W0o();}JIb rank=JaL W2T>(a,69760,n)W0b JaE=JaL W2T>(a,102528,n)W0b JbR=W4T>(a,135296,n)W0b Jd2=W4T>(a,168064,s);Jat(rank,JaQ(-
#define J3j ,Z1L)W0R Z11=Z1K>Z1L?Z1K:Z1L W0R Jfm=Z11?Z11 W0y:W1M.JaR Run W4c>(gs[qi*W4e,k[ki W1t,kr[ki*64],dq[qi W1t,dkr,Jfm,2048,JIj,JHZ,Jbb);W2a W08 qi=JaM(JId)W0y W3v ki=JaM(JId)*2048 W17 W0b out=JIa?dv[ki W1t:dk[ki W1t W0R Jfm=Z32
#define J3k (bb,JaJ?245760:(Jd9?131072:262144));p.JaF(a2b,65536);p.JaF(b2b,65536);p.JaF(cb,131072);a1=ab W2n<T>();b1=bb W2n<T>();a2=a2b W2n<T>();b2=b2b W2n<T>();c=cb W5Q>()W4t JfL(W0n qg,W3W,W5I,Jak
#define J3l W1E(Z0u)W0b off=W2o W2T>()[10504];off W1E(Z0u)W0b tmp=W2o W2T>()[10760];tmp W1E(Z0u);Jcc(off,JaQ(0),Z0u)W0o();JdO(tmp,off,JaQ(3),Z0u)W0o()W2m(tmp,tmp,JaQ(24),Z0u)W0o();JdF(off,off,JaQ(2),Z0u);JaN
#define J3m W2q(n/2),1,0}W0O(b1[bOff],pb,p);if(n%32==0&&n>=64)JaU(b1[bOff+(n/2)*k],pb[W1n(n/2)*64],p);}Jd3();W0W>(2 W5D)W4t SqCompute(Jaj JIg,W3J out,Jaj m,Jaj n,Jaj k,Jaj Z1Y,W1J,W1N,Jaj Z2d)W1q
#define J3o countF[col],JaM(64),Jar(16),cp)W0o()W2m(JIf,JIf,JJq,SPAN)W0o();if(!JfM){Muls(ft,ft,W0E,16)W0o();Jg9(ft,maxCache);}Add(JIf,JIf,maxCache,SPAN)W0o();Exp(JIf,JIf,SPAN)W0o();if(!JfM){Jat(ones,1.0f,16);Jat(inv
#define J3p JI0 JaQ key=nKeys[j];Jeo(nlow[j W1t,src[W1n(key)W1t,cp,pad);}W07 JcG>(nev);}W06 JcG>(ev)W0b low=W2o<T>()[JJ5];low W1E(32768)W0b JJS=W2o<T>()[JJ5+W0K;JJS W1E(16384);if(Jfa){JaU(Jf0[W3F,low,16384)W0O(Jen
#define J3s Jcv(JJO,rank,JaE W3O<Jaj>(),0,n);Jdb();JJO.Jd4(s-1,JaQ(-8));JJO.Jd4(n-1,JaQ(32760));Jdb();Sub(JaE,rank,JJO,n)W0o();JdO(JaE,JaE,JaQ(3),n);Jdb()W0O(JbN[(2*JJ7)*s],JaE,s)W0O(JbN[(2*JJ7+1)*s],JaE
#define J3t );if(Jbq&&feat W0S){W5i{1,128,64,0,64,128,1,0}W0O(b1,JcK,rp);}W0W>(JaY)W2D>(JaY)W0x(Jaj JIr=0;JIr<2u;W2M){Jan bp{0,8,8,0,0,W4g(Jaj ki=0;ki<8u;++ki)Jas(b2 W39 W20+ki W1k],b1 W39 W20+ki W12
#define J3u .s2,t.W4z-128,Jbm,JHZ,f==0,f==0?256u:512u);}W0C W1q W4x((at-64)/2)*W0D W08 out=(JaM(b)*W5B)W0y+f;W0n Jcm=dk;if(Jcm W3A()W4u)Jcm=Jcm[out];W0n Jbm=dkr;W0X Jbq=f==0&&dkr W3A()W4u;if(Jbq)Jbm=dkr[(JaM
#define J3v for(Jaj JIr=0;JIr<2;W2M)W1q bs=JIr,b2o=bs W1p;if(bMath[bs]){W0A>(5+bs);bMath[bs]W5u W13(TA){Jan bp{0,8,32,0,0,W4g(Jaj ki=0;ki<8;++ki)Jas(b2[b2o+ki*W4e,JJN W39*65536+JaM(Z0f)*16+W2N],bp);W2a Jan bp
#define J3w Jat(JJS,W0c T>(0),16384);W07 Jcw>(gEv[0])W10 Jcw>(gEv[0]);}for(W4H=0;j<JJL;++j){JI0 JaQ key=JeZ[j];Jeo(low[j W1t,src[W1n(key)W1t,cp,pad);}W07 JcG>(gEv[0]);}for(Jaj JIk=0;JIk<2u;++JIk)W1q JI6=sub
#define J3y (Jc9<T,W1j&mm)W1q JIe=Jaz()W0x(W3Y=0;JJ2<2;++JJ2)W1q W4x(JIe-4)W1s+JJ2 W14,JIg=JJ2 W3v ko=W1n(col)W0y,kp=W1n(col)*64,ro=W1n(JIe)*131072+W1P 32768 W0x(Jaj c=0;c<9;++c){if(c<8)W1q JIq=c&1,ff=
#define J3z Jdw[rawAt],8192)W0O(dp,Jdw[4194304ull+rawAt],8192)W0O(fw,JIR W5w],64)W0O(ft,mx W5w],64)W0O(JaI,sm W5w],64)W2v();Jdx(tw+3)W2m(JIf,JIf,t.JJq,8192)W0o()W2m(ft,ft,-1.f,64)W0o();C9QueuedRows128 W4c>(JIf,ft);Exp(JIf,JIf,8192);JaN
#define J3A W2e JdX(Jbe<T>&mm,W33)W1q rt=(r+63)/64,kt=(t.s2+63)/64 W0R per=rt*Jcx()+kt*(Z01?2:1)W0R qchunk=((t.s2+Jcx()W14-1)/(Jcx()W14))*W0D W01 b W2s/per,at W2s%per W3v wb=Jbk?0ull:W1n(b)*r*t.s2;if
#define J3D W2s%W0D W01 f=((at%2)^((at/4)%2))W1s W3v wb=JaM(b)*r*t.s2;if(at<64)W1q W4h(at/2)*W0D W08 JJY=JaM(b)*W18;W0n Jcm=dq;if(Jcm W3A()W4u)Jcm=dq[JJY W0y+f];W0n Jbm=dqr;if(Jbm W3A()W4u)Jbm=dqr[JJY
#define J3E packetRead[ns]){W06 Jcw>(ns);packetRead[ns]W5u JIb ni=JaL W2T>(a,np,2048)W0b ng=JaL<T>(a,np+8192,4096),no=JaL<T>(a,np+16384,4096)W0O(ni,JbB[JaM(nt)*W4e,2048)W0O(ng,go[JaM(nt)*W1m,4096)W0O(no,o[JaM(nt)*W1m,4096);W07 JcI
#define J48 W2n<T>()[JJ5];low W1E(16384)W0b JJS=W2o<T>()[JJ5+W2b;JJS W1E(8192);if(Jfa){JaU(Jf0[W3F,low,8192)W0O(Jen[W3F,JJS,8192);W2a JaU(Jf0[W3F,JJS,8192)W0O(Jen[W3F,low,8192);}W07 JcV>(ev);W1g b]W3N;}for
#define J4a ,dpG[sub*SPAN],SPAN);if(!JfM){JaU(fw,JIR[sub*16],16)W0O(ft,Z1w[sub*16],16)W0O(inv,sums[sub*16],16);}JaU(JIu,countG[Jch],128);JJZ();JJH(countF,JIu,W1G,128)W0o();Jdf cp{1,1,16,W4g(W1e=0;col<W0D W5H)Copy(mult[col
#define J4b ;JbX W1E(2*Q);W0N,W1j(JbX,JII,JJl,JJA,JaQ(Q/32))W2v();JbX W1E(4*Q);Jat(JbX[2*Q],W0E,2*Q)W0o();Jcc(JII,0.0f,t.s2);Jat(JIu,JaQ(-8),t.s2)W0o()W0x(Jaj bit=Q;bit>0;bit/=2){Adds(JaE,JIu,W0V
#define J4c if(dqr W3A()W4u){JbU cp{16,4,28,0}W0O(dqr W5w*64],raw,cp)W2v();}for(Jaj h=0;h<16;++h)Jat(raw[h W1t,W0c T>(0),64)W2v();}JaU(dq W5w W1t,raw,8192)W2v();}W2e JdJ(){JJ1.JaF(fb,JbC()?99072:(4*4096+1024+32
#define J4d {Jb7 p{1,W2q(m),64,0,64,W2q(m),1,0}W0O(a1[aOff],pa,p);}Jb7 bcpH{1,W2q(n/2),W2q(k),0,W2q(JJU),W2q(n/2),1,0};(JIm)bcp W0O(b1[bOff],bg,bcpH);if(n%32==0&&n>=64)JaU(b1[bOff
#define J4f ();W0N,W1j(JbX,JII,JJl,JJA,JaQ(Q/32))W2v();{JaQ Je7=-1,run=0 W0x(W4H=0;j<Q;++j){JI0 JaQ p=W0V>(JbX.Jco(2*j));if(p<0)Z0R;if(p==Je7)++run W1b{if(run)JIu.Jd4(Je7,run);Je7=p;run=1;}}if(run)JIu.Jd4(Je7,run
#define J4G ko+Jbb],W2y],mr,nr,128,512,512,nr,W4C,W3z 384,JHZ,0);W5U(qr,kr,JHZ,W0F!W0P W4C);mm.JaR Block W4c,W1j(go[qo+Jbb],(Z01?v:k)[ko+Jbb],W2y+W0K,mr,nr,128,512,512,nr,W4C,W3z 384,!Z01,16384);}}Jbg<2,JdH>(JIg
#define J4H =JbZ/4,bit=(JbZ%4)*2 W0x(Jaj JIA=0;JIA<2;++JIA)if(!(Jdm[b]&(W16(bit+JIA)))&&JdY(20+JbZ*2+JIA)>=4)Jdm[b]|=W16(bit+JIA);if((Jdm[b]&(3u<<bit))!=(3u<<bit))JcS W0x(Jaj Jbb=0;Jbb<2;++Jbb)W1q Z1R=W16(JIW*2+Jbb);if
#define J4I JcD<W1j();JJ1.Reset();JJ1.JaF(fb,(4*8192+1024+64)*4)W0b f=W5S>();JIf=f;dp=f[W2b;JJP=f[W0K;mult=f[24576];fw=f[W45;ft=fw[512];JaI=ft[512];JJ1.JaF(tb,16384);raw=tb W2n<T>();JJ1.JaF(JJ9,32768);Jc2=JJ9 W2n<
#define J4J W13(TA){JI0 Jan ap{0,8,1,0,7,W4g(W21=0;mi<8 W4K){Jas(a2[gao+mi*W4e,a1[ao+mi*W4e,ap);}W2a JI0 Jay ap{0,8,8,0,0,JHZ,W4g(W21=0;mi<8 W4K){Jc4(a2[gao+mi*W4e,a1[ao+W1x],ap);}}for(Jaj JIr
#define J4K ,Jcm,mr,kr,t.s2,row==0,W0P row+kr==r,r-row-kr,Jbm,Jbq);}{}}W2e JeT(Jc9<T>&mm)W1q rt=(r+63)/64,kt=(t.s2+63)/64 W0R qpt=rt*Jcx(),kvpt=kt*(Z01?2:1),per=qpt+kvpt W0R qu=(t.s2+Jcx()W14-1)/(Jcx()W14
#define J4L JIy{Jay p{0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(W21=0;mi<m/W4L)Jc4(a2[ao+mi*k*16],a1[ao+W1x],p);}for(Jaj JIr=0;JIr<2;W2M)W1q bs=JIr%2,b2o=bs W1p;if(JJN[bs]){W0A>(5+bs);JJN
#define J4M JaV>()W1O packetRead[2]={JHZ,JHZ W4f(W4H=0;j<n;++j)W1q s=j%2,JIh=JI6+j,tw=96+j*6,JcO=98304+s*24576;if(Jek){tr.Jd4(tw,JIh);tr.Jd4(tw+1,Jcu());}W06 JcI>(s);if(j+1<n)W1q ns=(j+1)%2,np=98304+ns*24576
#define J4N W4g(W21=0;mi<m/W4L){Jas(a2[ao+mi*k*16],a1[ao+mi*k*16],p);if(FN==512||JeR)Jas(a2[ao+8192+mi*k*16],a1[ao+8192+mi*k*16],p);}W2a Jay p{0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(W21
#define J4O =0;mi<m/W4L)Jc4(a2[ao+mi*k*16],a1[aOff+W1x],ap);Jay bpH{0,W1U>(k*nH/256),1,0,0,JHZ,W4g(Jaj h=0;h<2;++h){if(Z0B[h]){W0A>(h==0?Jao:JaK);Z0B[h]W5u Jc4(b2[h*(k*nH)],b1[bOff
#define J4P /W4L){Jc4(a2[ao+mi*k*16],a1[ao+W1x],p);if(FN==512||JeR)Jc4(a2[ao+8192+mi*k*16],a1[ao+8192+W1x],p);}}for(Jaj JIr=0;JIr<FN/W0D W2M)W1q bs=JIr%2,b2o=bs W1p;if(JJN[bs]){W0A>(5+bs);JJN[bs]W0s
#define J4Q Jb7 p{1,32,512,0,512,32,1,0}W0O(a1,q,p)W0O(a1[GO],go,p);DmaReady();if(qResident){Jay qp{0,8,2,0,0,JHZ,W4g(Jaj f=0;f<W4X f+=128)for(W21=0;mi<2 W4K)Jc4(a2[QRES+f*32+mi W14*16],a1[f*32+W1x],qp);
#define J4R if(JeR)Mmad(Jc0,a2[ao+W2b,b2[b2o],mm);JaW<J12>(nk),512,0,512,W2q(nk),1,0};if(W50]){W1X>(ns);W50]W5u JaU(a1[
#define J4S ]W3N;J23;fp.Z0E=m;fp.JJz=W4Y.JbK=m;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA W0s J01>(Jbm,c,rp);}}W07 Jce>(JaY);Jbp W3N;Jcz
#define J4T nap J56>(3+as);Z0v[as]W3N;J23;fp.Z0E=m;fp.JJz=W4Y.JbK=m;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA W0s J01>(Jbm,c,rp);}}JaW<Jal
#define J4U ;W2M)W1q bs=JIr%2,b2o=bs W1p;if(JJN[bs]){W0A>(5+bs);JJN[bs]W0s J45=c W39*m W14];Mmad(Jc0,a2[ao],b2[b2o],mm);mm.JbQ W1B if
#define J4V <4;W2M)W1q bs=JIr%2,b2o=bs W1p;if(JJN[bs]){W0A>(5+bs);JJN[bs]W0s J45=c W39*m W14];Mmad(Jc0,a2[ao],b2[b2o],mm);mm.JbQ W0s
#define J4W ,512,W2q(nk),1,0};if(W50]){W1X>(ns);W50]W5u JaU(a1[ns*W0K,JJS[Jd6],nap);if(JeR)JaU(a1[ns W1p+W2b,low[Jd6],nap
#define J4X JaZ(W4n*>(JcU+pos));pos+=W1n(W0U)*t.s1*t.s2*4;gs W3I(W2I T*>(JcU+pos));pos+=JcW*4;ps W3I(W2I T*>(JcU+pos));pos+=JcW
#define J4Y 16)}W0O(dst W5w*W5B],raw,cp)W2v();if(Z2u){JJH(raw,JJP,W1F Jdo,JIQ)W2v()W0O(dst[JcW W17*W5B],raw,cp)W2v();}Jhx W4t Jgb
#define J4Z W03 JJu(W0n qg,W3W,W5I,W3J JeQ,W3J Z0L,Jaj m,Jaj n,W1J,W1N)W44 SqPrefetch(Jaj JIg,W0n ag,W5a,Jaj m,Jaj n,JzP W0s,Jaj Z2d
#define J50 &&Jau<t.Jbf/512)Jau=t.Jbf/512 W1b if(t.s2>2048&&t.Jbf>2048&&t.Jbf%2048==0&&Jau<t.Jbf/2048)Jau=t.Jbf/2048;}W01 maxParts=t.Jbf/256;if(Jau>maxParts)Jau=maxParts;if(Jau
#define J51 .JaF(a2b,65536);p.JaF(b2b,65536);p.JaF(cb,131072);a1=ab W2n<T>();b1=bb W2n<T>();a2=a2b W2n<T>();b2=b2b W2n<T>();c=cb W2n
#define J52 JaR<JI7 TA,W40>W03 Z1I(W4p,W0a J00>(TA?k:m),J67(JeR)JaU(a1[ao+W2b,low,acp)W0O(b1[bo],bg,bcp);if(Je6 J34>(k/16-1),0};Jhm)
#define J53 (JeR)JaU(a1[ao+W2b,low,acp)W0O(b1[bo],bg,bcp);if(Je6 J34>(k/16-1),W4g(W21=0;mi<m/W4L){Jas(a2[ao+mi*k*16],a1[ao+mi
#define J54 Jaj JdV);W11 JfW()W44 Z2X(W27,W1e,Jaj nr,Jaj JIv);W11 Jga(W27,W1e,Jaj Z1X)W44 Jf5(W0n dst,W0B>f,W27,W1e,Jaj nr,Jaj JIv,JI7 Z2u)W44
#define J55 >W03 Run(W4p,W5a,W0n JcC,W5d>out,W0n Jbm,Jaj Jfm,W3D,JI7 Z20,W3s,Jaj JdP,Jaj Z2B){Jd0(W3p,JI2>W0p||W3p,T>W0p,"Grad128 output must be FP32 scratch or input dtype");W13
#define J4g W07 JcI>(Jao)W10 JcI>(Jao);JJH(mult,Jc2,W1G,JIQ);}Muls(JIf,JIf,t.JJq,JIQ)W2m(ft,ft,W0E,JIv);Jcj(ft,JIv,nr);Add(J08,JIv,nr);Mul(JIf
#define J4i JaV>();}J06 W2C>(65536,8192);JcX()W5r(broadcast,JbE,Jar(64),Jbv{2,16})W5r(broadcast[8],JbE[512],Jar(64),Jbv{2,16})W0o();JcF();JI0 Jbz cp
#define J4j for(W33=W5g<Jdj W4d+=Jcy,++JJ2)W1q JIg=JJ2%t.JJD;if(JJ2>=t.JJD)Jav<2,Jc8>(W3B+1)W0R W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b=Jdi+JIi
#define J4k (JdA<T,ROPE,Z01,Jdq,Jbk>(t))JeG<T>(JJ1,JbB,JbN,go,o,JIR,t,0,W1n(W0U)*t.s1/2)W1b{Jem();Jf4();}JcD<W1j();JJ1.Reset();Jei
#define J4l JIv)W0O(ft,mx[JIX]Jef*nr)W0o();W4H=0;Z0H(j<J89 J0F Jcj(ft,JIv,nr)W0o();Add(JIf,JIf,JJP,JIQ)W0o();Exp(JIf,JIf,JIQ);W0i
#define J4m )J0O,1,0}W0O(a1,Z0X,rp);}}for(Jaj JIq=0;JIq<2;++JIq){if(JIq==0){JaU(b1,kg,bcp);if(Je6){W5i{1,256,64,0,64,256,1,0};Jhp J09,64,
#define J4n ;}Jbg<2,JdH>(W3B);}W01 JJa=JJ2>4?JJ2-4:0 W0x(Jaj i=JJa;i<JJ2;++i)Jav<2,Jc8>((i%4)*2+1);W5U(qr,kr,JHZ,JHZ)W4t Jdz(
#define J4o W01 rt=(r+63)/64,kt=(t.s2+63)/64 W0R qpt=rt*Jcx(),kvpt=kt*(Z01?2:1),per=qpt+kvpt W0R qu=(t.s2+Jcx()W14-1)/(Jcx()W14),ku=
#define J4p )*(((W4w)W49())?1048576ull:524288ull);JIR W3I(W4m*>(JcU+pos));pos+=JJL*4;Jee W3I(W4m*>(JcU+pos));Je3 W3I(W4n*>(JcU+pos+(Jcx
#define J4q ();Jab],JaI[16],16)W2v();}}W11 Jdt(){Jb1 W0P!W0F W0U>=8&&r>=W5s r%W1I&&t.JkA==0&&t.s2 W5z t.s2<=2048&&t.s2%W1I;}W11 JbL
#define J4r Z1i(mask,JbR W3O<W1y(),Jd2 W3O<W1y(),W1H,N)W0o();JbG(Z0C W3T>(),mask,JaE W3T>(),Z0C W3T>(),W3x JdR,N);JaN
#define J4s JaZ(W2I T*>(JIB));k W3I(W2I T*>(key));v W3I(W2I T*>(JIa));go W3I(W2I T*>(Z1t));o W3I(W2I T*>(JcP));qr W3I(W2I
#define J4t ;if(t.s2>W5P r>=512)Jb1 2;if(r<=128&&t.s2<=W0H W0U<=2&&t.s2%512==0)Jb1 4;Jb1 r<W5s t.s2>=2048?4:1 W4t JIY(){JaN<JdL>();}W11 JbC
#define J4u W01 JIL=sub*JIA,end=JIL+JIA<mr?JIL+JIA:mr;J0I JcI>(Jao)W0O(fw,JIR[JIX],JIv)W0O(ft,mx[JIX],JIv)W0O(JaI,sm[JIX],JIv);if(t.n1 J0M j=0;Z0H(j<J89
#define J4v >W03 CopyK(W4p,W5a,W0n JcC,Jaj Z0f,Jaj JIK,W3D,JI7 Z20,Jaj JdP){if(Z1j W2z]){W1X>(JIK);Z1j W2z]W5u W08 aOffset=TA?JaM(Z0f)*Jcl:JaM
#define J4w Jaj mr=r JnD>(b)*W18+JI6 W0O(JIf,W2y+JI6*nr],JIQ)W0O(dp,W2y+JeM+JI6*nr],JIQ)W2v()W0O(fw,JIR[JIX],JIv)W0O(ft,mx[JIX]Jef*nr);JaN
#define J4x ;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 W13(W3p,JI2>W0p)fp.JcT=W5h Z1B W1b W13(W3p,W1z)fp.JcT=W5h Je8 W1b fp.JcT=W5h Z2G
#define J4y ,ek[(W1n(b)*W5B)W1t,dq[(W1n(b)*W18)W1t,mr,kr,t.s2,col==JIL,col+kr==end,end W4y-kr,dkr,JHZ);W2a if(Jga(W0d
#define J4z W01 vc=Jbi()*2 W0b JbW=W5S>()W0b JcQ=JbW[W2b W0x(JaM JI6=Z2w+W1n(Jaz())*16;JI6<rowEnd;JI6+=vc*16){JaU(raw,J0C,16)W2v();}}W11 Jdt
#define J4C .JbY=0;fp.W30 W2u,JI2,Jax>(JeQ,c,fp)J0q)W0x(Jaj JIA=0;JIA<2;++JIA){JaU(a1,gog[Jbb+JIA*halfRows W1t,acp);W0W>(JaY)W2D>(JaY
#define J4D W2t JbF(){if(Jbw())Jb1 256;{}if(JaH())Jb1 256;if(Jds())Jb1 W2O(Jhj JIU=W1n(W0U)*((r+63)/64)*((t.J04()JEU)W0R mr=r-row<Jbs
#define J4F /qpt)*per+i%qpt:(i/kvpt)*per+qpt+i%kvpt;JdX(mm,JIi);}{}Jgh+i%qpt;JdX(mm,JIi);}mm.Jd3();{}W2e Jej(Jc9<T>&mm,Jaj Jdi,Jaj JdV)W44 Z25
#define J57 .J1N>()W2m(JbR,JaE,8.0f,t.s2)W0o();Adds(JbR,JbR,-8.0f,t.s2)W0o();JJH(J0y W2T>(t.s2-1))J1O,t.s2 J0j JdB()W44 Jem(){{JeL
#define J58 av.Init(JJ1);W0X Jdl=((t.JbP>>20)&1u)!=0;W0X Jct=Jdl&&((t.JbP>>21)&1u)!=0 W0R vid=Jaz(),Jfp=(255-vid)J0D);Jhi{if(JcK W5G<4)W1q JJ7=vid
#define J59 m;mp.n=n;mp.k=kp;JoE=JaB;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 W2u,JI2,Jax>(JIq?od:os,c,fp);W07 Jce>(JaY)W10 Jce>(JaY
#define J5a (W0B>dst,W0B>src,W0B>JIs){JcX();JI0 Jbz cp{1,1,0,32,32,1 W4f(W1e=0;col<256;W5H){W13(MULT)Mul(dst[col],src[col],JIs,JaM(64)
#define J5b W2e JdJ(){JJ1.JaF(fb,(Jgr==4u||Jgr==5u)?100352:(JbC()?99072:(4*4096+1024+32)*4))W0b f=W5S>();JIf=f;dp=f[W1m;JJP=f[W2b;mult=
#define J5c ],ap);JCS>(JaY);W2u,JI2,Jax>(Z0L,c,fp);JbT<J19>(k),0,512,W2q(m),1,0};Jb7 bcp{1,W2q(n),W2q(k),0,512,W0e
#define J5d W03 Jf1()W44 Jgg()W44 JeL()W44 JdB()W44 Jem()W44 JK2(Jaj id,JaQ v);W00 JaQ C8Read(Jaj id);W11 C8Panel(Jaj p,Jaj&seen)W44 Z3g(Jaj p);W00
#define J5f 63)/64 W0R per=rt*Jcx()+kt*(Z01?2:1)W0R qchunk=((t.s2+Jcx()W14-1)/(Jcx()W14))*W0D W01 b W2s/per,at W2s%per W3v wb=Jbk?0ull:W0d
#define J5g JIb rank W1o W2T>(69760,n)W0b JaE W1o W2T>(102528,n)W0b JbR W2C>(135296,n)W0b Jd2 W2C>(168064,s);Jat(rank,JaQ(-8),s Jhs[s/8],JbR[s],Jd2,W1H
#define J5H J11 cp{16,32,0,W2q((t.s2-512)/16)}W0O(dst[JIX*W5B],raw,cp);J0g,raw,t,JIX,col,JIj)W4t Init(Z0F&JJ1){JJ1.JaF(JIF,ARENA_BYTES);Jdu W0s
#define J5I [256],JfT[512],halfCp);J1I W1t,JbN[(Je2+JIk)*W5B],512);J0r ones W2C>(131584,16)W2m(Z15,Z15,W0E,16)JlB>();JcX()W5r(Z2f,Z15,Jar(2),Jbv
#define J5J W01 JIg=JJ2%4;Jav<2,Jc8>(W3B)W0R W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b W2s/(rt*kt)W0R mr=r JnD>(b)*W18+JI6 W0O(JIf,W2y+JI6
#define J5K .J17=JaB;fp.JJg=1;fp.J0k;if(m==256&&n==128)W1q halfRows=128,halfElems=halfRows*n;Jb7 acp{1,128,128,0,512,128,1,0};Jb7 bcp{1,128,128,0,512,128,1,0};Jay ap
#define J5L Jaj Jbb=0;Jbb<W4X J0z()W0x(W21=0;mi<8 W4K)Jc4(a2[mi*k*16],a1[W1x],ap);JCS>(JaY);W3Z;fp.Z0E=W2r.JJz=W2r.JbK=W2r.JaB=JaB
#define J5M (Je0())Jb1 W2O(Jdt())Jb1 W0D W08 JIU=W1n(W0U)*((r+63)/64)*J0A()JEU);JI0 J16(q[qo],go[qo],k[ko],v[ko],W2y],W2y+32768
#define J5N ap);W52{0,W1U>(n*k/256),1,0,0,JHZ,0}W0R bo=seq>=7?tail:(isDp W53:0)+J1J;Mmad(c[isDp?16384:0],a2[off],b2[off],mm);W1W
#define J5O Jay ap{0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(W21=0;mi<m/W4L)Jc4(a2[mi*k*16],a1[aOff+W1x]J1P>(2 W5D);Jfr W3f
#define J5P W31;mm.m=W1M.n=W1M.k=W1M.JbQ=Z0f==0;mm.Jci W59(c W39*W0K,a2[ao],b2[b2o],mm);W1W>(5+bs);bMath[bs]W3N;if(W34
#define J5R W03 ApplyRow(W0B>dst,W0B>src,W0B>JIs)W44 ApplyCounts(W0B>dst,W0B>src,W0B>JbE)W44 PackHigh(W0n dst,W0B>src,Jb0<T>raw,W2j&t,JaM JIX,Jaj
#define J5S W3v out=(W1n(b)*W5B)W0y;W0a JsI>(b J1f(!Jhh+wo],(val?go:eq)[qo],Jcm,mr,kr,t.s2,row==0,W0P row+kr==r,r-row-kr,Jbm,Jbq
#define J5T ,s)W0o();JbG(rank Jde(n);J1d[s],s)W2v()W0b JbW W2C>(0,J10();Mul(J0N>(7)W10 Jck>(7)W0O(JIR[JI6],delta16,16)W2v();}
#define J5U );}mm.Jd3();}J0L 128)*(Z01?1:2)J0U:kvpt))*per+(JJX?i%qpt:qpt+i J0c:qpt))*per+(JJX?qpt+i%kvpt:i%qpt);JdX(mm,JIi);}mm.Jd3()W4t Z25
#define J5V Jbg<2,Jc8>(recycle[0]);W07 JcV>(Jao)W10 JcV>(Jao);Jcq();}JaR<JI7 FIRST>W03 JfI(Jam Jc6,Jaj Jch,Jaj Z0T,W0n Z1M,W0n Jfl,W0n Jf0,W0n Jen
#define J5W ,dp,deltaCache,SPAN)W0o();Mul J0G WeightsTile(Jam Jc6,Jaj JaD,Jaj Jch,JI2 JJq,W3J scoreG,W3J dpG,W3J Z1w,W3J sums,W3J JIR,W0n gsHigh,W0n gsLow,W0n psHigh,Jak
#define J5X JaR<JJ0 T,JI7 ROPE,JI7 Z01,JI7 Jdq,JI7 Jbk>W11 JdA(W2j&t){W13(W55,W2g W0F W0P Jdq&&!Jbk){W08 Jc7=W1n(W0U)*t.s1;Jb1 t.n1==8&&t.Jbf==2048
#define J5Y .JaF(fb,(4*8192+1024+64)*4)W0b f=W5S>();JIf=f;dp=f[W2b;JJP=f[W0K;mult=f[24576];fw=f[W45;ft=fw[512];JaI=ft[512];JJ1.JaF
#define J5Z 3 W5D);Z2y W3f]W3N;}for(J1H,c[W0K,fp);J0Z,512,0,512,128,1,0}W0O(b1,kg,kvp)W0O(b1[W5O,vg,kvp);if(Je6){W5i{1,128,64,0,64,128
#define J60 (W3J Z2I,W3J ringDp,W1d>JbN,W3J JIR,W3J mx,W3J sm,W0n gs,W0n ps,W2j&t,JaM JIX,W1e,JI7 Jdl W0s,JI7 Jct W0s){(JIm)Jdl;(JIm)Jct
#define J62 JaU(a1,Z0X,rp);}}JaU(b1,kg,full);if(Je6){W5i{1,128,64,0,64,128,1,0}J1T=0;seq<9;++seq){JI0 J22>(k/16),8,0,0,JHZ,W4g(W21
#define J63 +JJa],n);Jat(JIu,0,t.s2)W2v();if(n>=64&&n<=2048){Jaj Q=64;Z0H(Q<n)Q*=2 W0R A=Q Jho W0E,A);Jat(JJl,Jaj(0),A);JpF>();J1e((
#define J64 (W4p,W0n low,W5a,W5d>out,W1J,W1N,Jaj JbS,W0n Jbm,W3s,JI7 dqFeatureZero W0s,Jaj bStride=512){Jd0(W3p,JI2>W0p||W3p,T>W0p,"H2PairFull256 output type");Jaq Jaj m
#define J66 (Z0F&JJ1,W1d>JbB,W1d>JbN,W0n go,W0n o,W3J JIR,W2j&t,Jam Jek,Jam tokenFlags){Jdb();JJ1.Reset();W5y Z17>a;JJ1.JaF(a,147456)W0R av=Jaz(
#define J5h ;Jcz=0;}}JaR<JI7 TA,W40,JI7 DQ_PREFIX_SPLIT W0s>W03 H3Pair128(W4p,W0n low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,Jaj featOff,W0n Jbm,W3s);
#define J5i )W0R mr=r-row<Jbs?r-row:Jbs,nr=t.W4z<JIH?t.W4z:JIH J0H>(b)*W5B)*64],ROPE,ROPE);J1R<2,Jc8>((i%t.JJD)*2+1);Jhr()J0E
#define J5k 16)*2048;JI0 JbU cp{4,128,0,896}W0O(dst[halfBase],raw,cp);W07 Z0y>(Jcf)W10 Z0y>(Jcf);}Z0Y:W03 Jgk(W0n gs,W0n ps,W2j&t,JaM JIX,W1e
#define J5l W03 JcM(){if(JdM){W0A>(JbM);JdM W5u}Z0Y:W03 Jd3(){if(Jbp){W06 Jce>(JaY);Jbp W5u W2e Jcn(W0n a,W0n b,JI7 pa,JI7 pb
#define J5n );W4Z{Z25();}}}W00 W1d>Z2i(){W1d>x W0b low=ps[JcW];x W3I(W4n*>(Jbl<Jb3 T*>(low W3A())));Jb1 x W4t Z33(W1d>JJe,Jaj i
#define J5o W0R Jeg=(Z0G%16)/4,Z31=b*4+Jeg;if((Jcg[Z31]&255u)!=255u&&!(Jfi&(W16 Z31))){Jcg[Z31]=Jex(b,Jcg[Z31],Jeg+1,Jg6);Jfi|=W16 Z31;}if((Jcg[Z31
#define J5p W1X>(2+s);Jfr[s]W5u}JaR<JI7 TA,JI7 TB>W03 Block(W0n ag,W5a,W3J out,Jaj m,Jaj n,Jaj k,W3D,Jaj JJU,Jaj Z1Y,W1J,W1N,JI7 reuseB W0s
#define J5r W01 id=84+Z19*4+j W0b JK5=Jdd[JaM(id)*32];J03(20+Z19,1,W0V>(Z11));JcN(20+Z19,0,W0V>(JcK));Jdn W3f]=JcK;}}}}W11 JfB(Jaj JbZ,Jaj JIk
#define J5t .s2/W0D W08 JIU=JaM(W0U)*rt*kt W0R Jd7=32768,JeM=16384 W0R JIe=Jaz()/2,sub=Je9(),JJJ=Jbi()W3v Jcp=JIU*JIe/JJJ,Jdj=JIU*(JIe+1)/JJJ;W3Y=0;JaM
#define J5u JbP&&Z3x()){if(Jfz())JfF()W1b RunSpatial();Jb1;}}W4Z{JdJ();if(Jau>1){Jat(JIu,0,t.s2)W2v();J0s b=0;b<W0U;b+=t.Jed)W1q n=W0U
#define J5v .JbA){W4Q>(JbO);mm.JbA W1B if(mm.Jbp){mm.Jd3();W0W>(JaY)W2D>(JaY);}W2a if(mm.W4G]){W1X>(as);mm.Jd5
#define J5w .Z3R(JFX vid+40*Jc3<64;Z0R;}if(Z1n)JcS;if(Jc3<Jfp)W1q JJ7=vid+40*Jc3;av.Z3d(t,JbB,JbN,go,o,JIR,JJ7);W07 JcV>(JaK)W10 JcV>(JaK);Z1e[Jc3
#define J5x JI7 JdM W1B JI7 Jbp W1B JI7 Z0v[2]={JHZ,JHZ},JJN[2]={JHZ,JHZ};Jaj Jcz=0 W1O JbA W1B JI7 W4F 2]={JHZ,JHZ};W0n Z0X,JcC W1O JK6 W0s,Je6 W1B JI7 Jfr[2]={JHZ
#define J5z (JaY)W3h>(JaY);W3Z;fp.Z0E=64;fp.JJz=W4Y.JbK=64;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JcT=W5h Z1B;W2u,JI2,Jax
#define J5A JaM(JIk)*65536+JaM(child)*32768;JdT W5T(qr[qi*64],kr[ki J0J));}if(Jgs&&pass==1){W07 Z0q>(Jao)W10 Z0q>(Jao);Jel=pass*4+4;Jdy(JIe,W0V>(Jel
#define J5C W06 JcG>(Jao);if(dqr W3A()W4u)JaU(dqr[qi*64],halves,8192);JaW<J2I());J1n JJ7){JIb JK5=Jdd[JaM(84+JJ7)*32 J1r);++pass){if(Jel<=pass*4)JcS W0R JbZ=Z2Z
#define J5E ;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 W13(W55,W1z)fp.JcT=W5h Je8 W1b fp.JcT=W5h Z2G;if(out W3A()W4u)JbT<T,JI2,Jax>(out
#define J5F W3O<Jaj>(),0,N)W0o();Sub(JaE,Z0C,JJO,N)W0o();JdO(JaE,JaE,JaQ(1),N)W0o();JJH(JbR,JaE,W1G,N)W0o();
#define J68 ;Jd3();W0W>(JaY)W2D>(JaY);}if(Z0v[as]){W0A>(3+as);Z0v[as]W5u W13(TA){Jan p{0,W1U>(k/16),1,0,JaO
#define J69 .s2)W2v()W0x(JaM JIh=Jaz();JIh<Jc7;JIh+=vc){JaU(JbN[JIh*t.s2],JIu,t.s2)W2v();}}}if Je4{W0X Jd9=W1w!W0F r W5z t.s2>=8192;W0X JaJ=JbL
#define J6c JaH())Jb1 256;if(Jds())Jb1 W2O(Je0())Jb1 W2O(Jdt())Jb1 W0D W08 JIU=W1n(W0U)*((r+63)/64)*((t.s2+255)/256);Jb1 t.s2%256==0&&JIU
#define J6d ;W1W>(3u W5D);W3V]W3N;}for(Jaj JIg=0;JIg<2u;++JIg)if(W3V])W0A>(3u W5D);W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E
#define J6e ,JaM(JJ7)*16,col)W1b av.Jh1(JFX vid+40*Jc3<64;Z0R;}}}if(Z1n)JcS;if(Jc3<Jfp)W1q JJ7=vid+40*Jc3;av.Z3d(t,JbB,JbN,go,o,JIR,JJ7);W07 JcV>(JaK
#define J6f )W4t EmitDQ(W3J out){JfZ();Write(out,DQ,512);EndFix()W4t EmitDQHalf(W0n out){JfZ();JaG p;p.Z0E=32;p.JJz=W4X p.JbK=32;p.JaB=W4X p.JJg=
#define J6g =Jbk?W1n(r)*t.s2:JJL*t.s2;JJf=JJL W0y;kelems=W1n(W0U)*t.s2 W0y W3v Jc7=W1n(W0U)*t.s1 W0R vc=Jbi()*2;Jau=Jc7<vc?JaO
#define J6J ;Jaj deltaVi=Jaz();if(t.Jge==1u)deltaVi=(deltaVi+8u)%vc W0b JbW=W5S>()W0b JcQ=JbW[W2b W0x(JaM JI6=W1n(deltaVi)*16;JI6<JJL;JI6+=vc*16){JaU(raw,go[JI6
#define J6L JaY)W3h>(JaY);W3Z;fp.Z0E=W2r.JJz=256;fp.JbK=W2r.JaB=W4Y.JJg=1;fp.Jbo=0;fp.Jhw=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax>(Jbm
#define J6M 320)W0b validMask=W2o<Jar>()[65856];validMask W1E(256);Jat(run.J2L>()[10248];pattern J3l W0h>();Sub(off,off,tmp,Z0u)W0o();Jcv(pattern,run.J2S{1,W2q(JaD
#define J6N ;Z0z Jaq Jaj QRES=16384,GO=16384,CH=32768,CL=36864;Z0z Jaq Jaj V=65536,ALOW=4096,BV=16384,DQ=J1t bo,Jaj n,Jaj k,W1J){JbI p;p.m=32;p.n=n;p.k=k;p.JbQ=JI6;p.Jci
#define J6O W03 Z30(){if(Z0N[0]){W0A>(Jcf);Z0N[0]W5u if(Z0N[1]){W0A>(JbD);Z0N[1]W5u if(Z0B[0]){W0A>(Jao);Z0B[0
#define J6P ];plow W1E(16384)W0b phigh=W2o<W1y()[dBase[0]+W2b;phigh W1E(8192)W0b pkraw=Jfa?plow:phigh W0b pvraw=Jfa?phigh:plow W0O(pkraw,W2x sub W1L],8192)W0O(pvraw,W2x 16384u W3P W1L
#define J6Q );W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=W2r.JJz=256;fp.JbK=W2r.JaB=W4Y.JJg=1;J29[64],c[64*m],tail);W2a JbT<Out,JI2,Jax
#define J6R >=32)Jb1;JJ1.Reset();JJ1.JaF(tb,8192*2)W0b halves=tb W2n<T>()W3v qi=JaM(vid)*W0D JbU gather{128,4,28,0}W0O(halves,dq[qi W1t,gather);W07 JcG>(Jao
#define J6T W1E(JaD);J3b>()[W5O;mask W1E(256)W0O(JJ8,inputRow,JaD);JJZ();JJH(JII,J2G>();Jat(JJl,Jaj(0),JaD)W0o();W0N,W1j(JbX,JII,JJl,JJA,JaQ(JaD
#define J6U W2n<T>();JJ1.JaF(JJ9,16384);Jc2=JJ9 W2V>();JJ1.JaF(ib,JbC()?16384:(t.Jbf+t.s2)*4);ix=ib W2V>();JIu=JbC()?ix:ix[t.Jbf
#define J6Z );}}}W4Z{if(Jau>1)JcD<W1j();if(JdA<T,ROPE,Z01,Jdq,Jbk>(t))JeG<T>(JJ1,JbB,JbN,go,o,JIR,t,0,W1n(W0U)*t.s1/2)W1b{Jem
#define J71 W2q(TA?m:nk),0,W2q(Jcl),W2q(TA?nk:m),1,0};Jb7 nbp{W2q(nk/16u),16,256,8192,512,16,1,4096};if(W50]){W1X
#define J72 bg[J1X::Jbh>(3+JIK);Z16 W2z]W5u if(bMath W2z]){W0A>(5+JIK);bMath W2z]W5u}}Z0Y:J1a W2z]W1B bMath W2z]W5u Jbp W1B Jev W1B Jeq=0;Jfs
#define J73 W0o();Jeu(delta8,Jdk,8,8,1,1,1)W0o();JcF();W07 Jck>(JaK);JaS<J2K<W2G>();}for(Jaj s=0;s<2;++s)if(packetRead[s])JaS
#define J74 ;Jfd rcp{1u,1024u,0u,0u,0u};JI0 J38>(key)W1Z],low W5w W1Z],cp);W2a rcp.blockCount=W2q(run);Jeo(kg[W1n(key)W1Z],low[kOff W17 W1Z],rcp);Jeo(vg
#define J75 ;Jdb()W0x(Jaj z=0;z<2;++z){JIb JII=W4T>(a,69760+z W1p,2048)W0b JJl=JaL<Jaj>(a,77952+z W1p,2048)W0b dst=JbX[z*W2b;dst W1E(4096);Jat(JJl,Jaj
#define J77 ,nbase=gBase[nb];JI0 Jb5 nev=gEv[nb];if(W1g nb]){J35);if(nRows<32){Jat(nlow,W0c T>(0),16384);Jat(nhigh,W0c T>(0),16384);W07 Jcw>(nev)W10 Jcw
#define J6j JaV>();Mins(Jap,Jap,W0V>(t.s2-1),t.s2)J3h(3),t.s2)W2v();Jdh W2T>()W0O(JbN[JIh*t.s2],JIu,t.s2)W2v();Jcq();}W2e Jgg
#define J6l kg,kcp);if(overlayK){W5i{1,256,64,0,64,256,1,0};J3e*8192;if(W3V]){W0A>(3 W5D);W3V]W5u if(JIG==7)W4Q>(JbO);Jay ap
#define J6n (Z1J){JaU(Jfn,View W0m>(163840+cacheItem W1s+Jaj(JIX%64)*4,16),16)W0O(Z15,mx[JIX],16)W0O(Z0P,sm[JIX],16);}for(Jaj JIk=0;JIk<2;++JIk)JaU(grid[JIk W0y
#define J6o W0l()J05>(131584,16)W2m(Z15,Z15,W0E,16)JlB>Jhg>()W2m(dp,dp,t.JJq,8192)W0o();Jdg(gs,dp,raw,JIX,col,JHZ);Jdg(ps,JIf,raw,JIX,col,JIj
#define J6p W08 out=(W1n(b)*W5B)W0y;W0a JsI>(b)*W18)W0y W0b Jfy=qr;if(!val)Jfy=qr[(W1n(b)*W18)*64];W5U(qr,Jfy,JHZ,ROPE
#define J6q >raw;Jb0 W2T>ix,JIu,Jc2;W0n q,k,v,go,o,qr,kr,eq,ek,gs,ps,dq,dk,dv,dqr,dkr;W3J mx,sm,scores,dps,wq,wk,wv,ring,JIR;W1d>JbB,JbN,Je3
#define J6r Jcj(fw,JIv,nr)W0o();Add(dp,dp,JJP,JIQ)W0o();}Mul(dp,dp,JIf,JIQ)W0o()W2m(dp,dp,t.JJq,JIQ)W0o()W2v();Jf5(gs
#define J6s JaF(cb,131072);a1=ab W2n<T>();b1=bb W2n<T>();a2=a2b W2n<T>();b2=b2b W2n<T>();c=cb W5Q>()W4t FillQGO(W0n q,Jak
#define J6u W1n(b)*W5B)*64];W5U(qr,Z2v,JHZ,ROPE);if(Jga(W1n(b)*W18,col,mr))mm JxN>(gs[J1o val=W0P JIk>=kt;mm.Je1(0u)W0R W4x(JIk
#define J6v W02 Jb9>(JbO);}W5K;mp.m=m;mp.n=n;mp.k=kp;JoE=256;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JfN W1B fp.unitFlag=0;fp.J25 W0m>od,JI7 JJQ
#define J6w for(W4H=0;j<JJL;++j){JI0 JaQ key=JeZ[j]W0O(raw[j W1t,(JIq J2d J0Y];ft J0P<J2k J4a],J3o[16],0.0f,48);J37(fw,fw,W0E,16)W0o();Jg9(fw,deltaCache
#define J6x ){JIb JII=W2o W2T>()[12928 Jiy(0));}JI7 W1g 2]={JHZ,JHZ};Jaj Z0I=0 W0x(Jaj JIq=0;JIq<2;++JIq)for(Jaj JI6=sub*64;JI6<sub*J2y j=0;j<JJL;++j)JeZ
#define J6z W1M.JbQ W3N;mm.Jci W59(c,a2[ao],b2,mm);Mmad(c[W0K,a2[ao],b2[W0K,mm);W1W>(3u+JIK);amath W2z]W3N;W3j>(JIK);l1 W2z
#define J6B 2;W2M)W1q b2o=JIr*JIq;if(J39;mm.m=m;mm.n=W1M.k=k;mm.JbQ=JI6;mm.Jci W0s W0b Jc0=c W39*JIq];Mmad(Jc0,a2[gao],b2[b2o],mm);W1W
#define J6C /2u!=0u W3v qi=JaM(b)W0w,ki=JaM(b)W15,wb=JaM(b)W0w W15 W0b out=JIa?dv:dk;if(out W3A()W4u)out=out[ki W1Z+feat]W0b rp=dkr;if(rp W3A()W4u
#define J6D ,JIf,JJP,JIQ);Mul(JIf,JIf,mult,JIQ)W2m(fw,fw,W0E,JIv);Jcj(fw,JIv,nr);Add(dp,dp,JJP,JIQ);Mul(dp,J0a;J3A(at<rt*JrH=W3K;Z2v=W3K[(W1n
#define J6E W0z],512)J05>(131584,16)W2m(Z15,Z15,W0E,16)JlB>Jhg>()W2m(dp,dp,t.JJq,8192)W0o();Jdg(gs,dp,raw,JIX,col,JHZ);Jdg(ps,JIf,raw,JIX,col,JIj
#define J6F J3M,59392,3072)J1C<T>(a,106496,4096),J2A<J4M,nt=JIh+1;if(J3E>(ns);}JIb JJ8=JaL W2T>(a,JcO,2048);J0B,W1j(JbX,JII,JJl,JJA,JaQ(64))W0o()J33<JIA
#define J6G J3z W0h>()W0b ones=J2e,col%256)W2m(fw,fw,-1.f,64)W0o();C9QueuedRows128 W4c>(dp,fw);Mul(dp,dp,JIf,8192)W0o()W2m(dp,dp,t.JJq,8192);W0i
#define J6H low W1E(32768)W0b JJS=W2o<T>()[gBase[0]+W0K;JJS W1E(16384);if(JJL<32){Jat(low,W0c T>(0),16384);J3w*64+JIk*32u,b=JIk%2u,JJ5=gBase[b];JI0 Jb5
#define J79 .Jed?W0U-b:t.Jed;if(J1M);W2a JdS(mm);JcD W4c>();if(JaJ){JJ1.Reset();mm.Init(JJ1);}JeT(mm);}}}W4Z{if(Jau>1 J0S>();J0W-b2:t.J0K
#define J7b Jg1[(id/2u)W0w]W0b b=st.Jg1[(id/2u)W0w+8u];J3K<W0k*>(b W3A());if(*p0==1&&*p1==1){Jcr&=~(W16 j);st.c3ReadySeen|=1ull<<t0;}}}Jbd<JcR::
#define J7d );mm.Je1(0u)W0x(Jaj W4h 0;row<r;W4M 128){W08 wo=wb+(JaM(row W47)*32 W4O)W1p W3v qo=(JaM(b)*W18)W0y+f;W5U(qr,W3K,JHZ,JHZ
#define J7e Jhn,b1,bp);JDT W0m,JI2,Jax>(Z0L,c,fp);J1A J3T[aOff],ag,acp);if(ovA)J4d+(n/2)*k],bg[W1n(n/2)*JJU],bcpH);if(ovB){Jb7 p{1,W0e
#define J7h Je1(0u)W0x(Jaj W4h 0;row<r;W4M 128){W08 wo=wb+(JaM(row W47)*32 W4O)W1p W3v qo=(JaM(b)*W18)W0y+f;W5U(qr,qr[(JaM(b)*W18
#define J7K Je1(0u)W0x(W1e=0;col<t.s2;W2U 128){W08 wo=wb+(JaM(row W47)*32 W4O)W1p;W5U(qr,W3K[(JaM(b)*W5B)*64],JHZ,JHZ);mm.JaR
#define J7L {W5U(qr[(W1n(b)*W18)*64],kr[(W1n(b)*W5B)*64],W0F W4C,W0F W4C);mm.JaR Block W4c,W1j(q[qo+Jbb],k[J4G*2
#define J7M W01 W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b W2s/(rt*kt)W0R mr=r-row<Jbs?r-row:Jbs,nr=t.W4z<JIH?t.W4z:JIH,JIA=(mr+1)/2 W0R
#define J7N J3a W5Q>()[2*Q];JJA W1E(2*Q)W0b JIJ=fb W2n<Jar>()[24576*4]J43+(JIi%2)*Q],Q)W2v();JJH(JII,JJ8 J2u,Q)W0o()J4b>(bit*
#define J7P ;Jb7 bcp{1,W2q(k),FN,0,J78],bg,bcp);if(Je6 J34>(k/16-1),J4N=0;mi<m J4P J45=c W39*m W14];Mmad(Jc0,a2[ao],b2[b2o],mm);mm.JbQ W0s
#define J7R Jat(JIu,0,t.s2)W2v();if(n>=64&&n<=2048){Jaj Q=(n+31)/32*32;{}W01 A=JdN?Q:(n+63)/64*64 Jho W0E,A);Jat(JJl,Jaj(0),A);JpF>J4f
#define J7S ;J2c(20+(JId*4 W17 W47)*2,Z1K);JdU(20+(JId*4 W17 W47)*2+1 J3j?128:W4X mm.JaR Run<W1j((JIa?ps:gs)[qi*2048 W17],(JIa?go:q)[qi W1t,qr
#define J7T >(b)*W5B)*64],ROPE,ROPE);mm.JdZ(q[qo],go[qo],k[ko],v[ko],W2y],W2y+W45,Jea!=qo,nr,(t.JbP&(W16 16))!=0);Jea=qo
#define J7U mi*W4e,a1[ao+W1x],ap);}J3v{0,8,8,0,0,W4g(Jaj ki=0;ki<8;++ki)Jas(b2[b2o+ki*W4e,b1[bo W5C W1p+W2N],bp);}W2E
#define J7V JHZ,W1j(gs[wo],gs[JcW+wo],f==0?h2PackedK[(JaM(b)*W5B)W1s]:ek[(JaM(b)*W5B)W0y+f],Jcm,col==0,col+W1D t J3u(b)*W5B
#define J7W nr,nr);J0l(q[qo],go[qo],k[ko],v[ko],W2y],W2y+W0K,nr);W2a J1s]J2a(0,W2y],mr,nr,128,nr,ff==0,ff==384,0);}}mm.SqDrain
#define J7X ,Jam key,Jam JIa,Jam Z1m,Jam Z1t,Jam JcP,Jam Z1w,Jam sums,Jam Jfe,Jam JIo,Jam gradQuery,Jam gradKey,Jam gradValue,Jam Z3v,Jam JgV,Jam JcU,W2j
#define J7Y JaR<JJ0 T,JI7 ROPE,JI7 Z01,JI7 JdN,JI7 Jdq W0s,JI7 Jbk W0s,JI7 Z08 W0s,JI7 HF3_GEOM W0s,JI7 COMPACT_VECTOR W0s,Jaj Jgr=0,JI7 HF3_DIRECT_DQ W0s
#define J7Z JI2>Z2I,W3J ringDp,W1d>JbN,W3J JIR,W3J mx,W3J sm,W0n gs,W0n ps,W2j
#define J80 J4R ns*W0K,JJS[Jd6],nap);if(JeR)JaU(a1[ns W1p+W2b,low[Jd6],nap J56>(3+as);Z0v[as J4S=
#define J81 ;W5y B1>bb;W5y A2>a2b;W5y B2>b2b;W5y CO1>cb;Jb0<T>a1,b1,a2,b2;W0B
#define J82 k*16],a1[ao+8192+W1x],p);}}for(Jaj JIr=0;JIr<4 J4U Jaq(Z2H)J4R ns*W0K,JJS[Jd6],nap
#define J83 W03 JgH(W0n JIB,W0n Z1t,W0n Jfe)W44 JK0(Jaj JIg,W0n ag,W5a,Jaj m,Jaj
#define J84 W03 JdZ(W0n qg,W3W,W5I,W0n vg,W3J os,W3J od,JI7 JJQ,Jaj JaB,JI7
#define J85 JaR<JI7 TA,JI7 Z2H W3N,JI7 PREFETCH_LOW W3N,W40>W03 Jdp(W4p,W0a J00>(TA?k:m),J67 Jaq(Z2H||PREFETCH_LOW
#define J86 *k*16],p);Jhf+mi*k*16],a1[ao+8192+W1x],p);}}for(Jaj JIr=0;JIr<4 J4U Jaq(Z2H
#define J87 W2a for(W4H=0;j<JIv;++j)JaU(Jc2[j*nr],JbN[((JIX+j)/t.n1)*W5B],nr
#define J88 T>JJS,W0n low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,W0n Jbm,W3s
#define J7j =Jbl<W0k*>(W5k())W0x(;;){W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>();if(
#define J7k +W1x],p);Jc4(a2[ao+8192+mi*k*16],a1[ao+8192+W1x],p);}}for(Jaj JIr=0;JIr
#define J7l >W11 JIE(Jam JIB,Jam key,Jam JIa,Jam go,Jam JcP,Jam maxG,Jam sumG,Jam idxG,Jam gradQ,Jam gradK,Jam gradV,Jam user,W2j&t);
#define J7m >vg,Jhl,W5I,W0n vg,W3J JeQ,W3J Z0L,Jaj m,Jaj n,Jaj JaB);J4Z=0);JaR<JI7 TA
#define J7n W03 JeG(Z0F&JJ1,W1d>JbB,W1d>JbN,W0n go,W0n o,W3J JIR,W2j&t,JaM Z34,JaM
#define J7o W03 C2PairFullWide128(W4p,W0n low,W5a,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,Jaj
#define J7p JaT JIm Run(W4p,W5a,W0n JcC,W5d>out,W0n Jbm,Jaj Jfm,W3D,JI7 Z20,W3s,Jaj JdP
#define J7q W03 JdZ(W0n qg,W3W,W5I,W0n vg,W3J os,W3J od,JI7 JJQ,Jaj JaB J4m 1
#define J7r W03 Resident448ScoreDB256(W0n qg,W3W,W5I,W0n vg,W3J os,W3J od,JI7 JJQ,Jaj JaB);
#define J7t W2t JgE(W32);W2t JeE(W32);W2t JgR(W32,Jaj JIW);W2t SpatialGroupBalancedTagCount(W32);W2t Jgu(W32,Jaj
#define J7u ;Jf5(ps,JIf,JIX,col,nr,JIv,!Z01);}{}Jbg<2,JeU>(W3B+1);}W2e JeY(Jaj Jdi,Jaj JdV
#define J7v JIb ones=JvL>();J0T,t.JJq,JIQ)W0o()W2v();Jf5(gs,dp,JIX,col,nr,JIv,JHZ);Jf5(ps
#define J7w Z0z Jaq Jaj ARENA_BYTES=185472;Jd0(Z18(T)==2,"spatial AV requires half-size storage");Jd0(ARENA_BYTES<188416,"spatial UB overlaps CANN2201 SDK temporary region");W5y Z17>JIF W1O Jdu W1B JaM JgW=0;Jaj Jgz
#define J7x JaR<JJ0 X>W00 Jb0<X>View(Jaj byte,Jaj size){JIb x=W2o<X>()[byte/Z18(X)];x W1E(size);
#define J7y qr,Z2v,JHZ,ROPE J0h>(gs[wo],gs[JcW+wo]J4y>(b)*W18,col,mr))mm JxN>(gs[wo],gs[JcW
#define J7z JaT JIm C2CachedWeightsPipeline();W11 C2WideGate()W44 C2WideGradTask(Jbe<T>&mm,W33)W44 C2WideGradients(Jbe<T>&mm)W44 Jha(Jbe
#define J7A W0q>();JIf=f;dp=f[W1m;JJP=f[W2b;mult=f[12288];fw=f[W0K;ft=fw[512];JaI
#define J7B if(Jbq&&Jbm W3A()W4u){JaG rp=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax>(Jbm,c,rp);}}
#define J7D ;Jb7 acp{1,W2q(TA?k:m),W2q(TA?m:k),0,W2q(Jcl),W2q(TA?k
#define J7E W03 JeY(Jaj Jdi,Jaj JdV)W44 JdX(Jc9<T>&mm,W33)W44 JeT(Jc9<T>&mm)W44 Jej(Jc9<
#define J7F =r-row<Jbs?r-row:Jbs,nr=t.W4z<JIH?t.W4z:JIH,JIA=(mr+1)/2;J4u);j+=copies;
#define J7G Jan p{0,8,W2q(k/16),0,0,W4g(Jaj ki=0;ki<k/16;++ki)Jas(b2[b2o+ki*
#define J7H )W1q Jfg JAQ r JnD>(b)*W18+JI6 W0O(JIf,W2y+JI6*nr],JIQ)W0O(dp,W2y+JeM+JI6*nr
#define J7I JJ1,W1d>JbB,W1d>JbN,W0n go,W0n o,W3J JIR,W2j&t,JaM Z34,JaM Z22,Jam Jek,Jam pairFlags
#define J8a ,JIs,JaM(64),Jar(32),cp)W1b Add(dst[col],dst[col],JIs,JaM(64),Jar(32),cp);}W0i
#define J8b W05 JcV>(Jao);Jcq()W4t Jgw(Jam Jc6,Jaj Jch,W5c ring,W5I,W0n vg,Jaj Z0T,Jaj Jfw
#define J8c W00 W1d>Z2i()W44 Z33(W1d>JJe,Jaj i)W44 HrScore(Jc9<T,W1j&mm)W44 HrWeights();JaP
#define J8d };W52{0,64,1,0,0,JHZ,W4g(J5L;fp.JJg=1;fp.Jbo=0;fp J4C);JcM()W0x(W21
#define J8e JaR<JI7 TA>W03 CopyK(W4p,W5a,W0n JcC,Jaj Z0f,Jaj JIK,W3D,JI7 Z20,Jaj JdP)W44 Z3f
#define J8i 0,W1U>(k/16),W2q(m/16),0,0,JHZ,W4g(W21=0;mi<m/W4L)Jc4
#define J8K W5a,W0n nextHigh,W0n nextBg,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,Jaj featOff,JgI*Z2L
#define J8M JaM(id)*32];W2Z*p=Jbl<W0k*>(W5k());W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<JcR::
#define J8N ,ek[J0o(0u)W0R W4x(JIk%kt)*64,mr=t.W4z<64?t.W4z:64 J5S);}{}}W2e
#define J8O W0F W0U>=8&&r>=W5s r%W1I&&t.JkA==0&&t.s2 W5z t.s2<=2048&&t.s2%W1I;}W11
#define J8P ;W2Z W48=Jbl<W0k*>(W5k());do{W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>();
#define J8Q 1,0}W0O(a1,qg[Jbb],acp)W0O(b1,kg[Jhn,b1,bp);JDT J1D(a1,gog[Jbb],acp)W0O(b1,vg
#define J8R Je4{W0X Jd9=W1w!W0F r W5z t.s2>=8192;W0X JaJ=JbL();Jc9<T>mm;mm.Init(JJ1,Jd9||JaJ,JaJ
#define J8S JaM JJL,JcW,JJf,kelems;JaM Jgp=0,Z1Z=0,Z2w=0,rowEnd=0;Jaj r,Jau;W3J Jee;W3J Jdw;W1d>Jdd
#define J8U }Jaj JeR=1 W44 Je1(Jaj g)W44 Init(Z0F&p,JI7 Jd9 W0s,JI7 JaJ W0s){W13(Z26){p.JaF(ab,196608
#define J8V JfY,Jaj ownerPlan)W1q JIu=Jh2(JIe)W0x(Jaj JIW=0;JIW<JIu;++JIW)W1q bit=W16 JIW;if(done&bit)JcS W0R
#define J8W )Jc4(a2[off+mi*k*16],a1[(isDp W53:0)+Jbb*m+W1x],J5N>(J5Z,1,0}W0O(b1,JcC
#define J8X W03 Init(Z0F&p,JI7 Jd9 W0s,JI7 JaJ W0s){p.JaF(ab,Jd9?262144:65536);p.JaF(bb,JaJ?245760:(Jd9?131072:
#define J8Y +=20u;}JIb JJ5=W2I Jar*>(Jbl<W0k*>(Je3 W3A()));W09 JaM>x;x W3I(W2I JaM*>(JJ5+
#define J8Z Jhk?(i/qpt)*per+i%qpt:(i/kvpt)*per+qpt+i%kvpt;JdX(mm,JIi);}Jgh+i%qpt;JdX(mm,JIi J5U
#define J90 +wo],(val?go:eq)[qo],Jcm,mr,kr,t.s2,row==0,W0P row+kr==r,r-row-kr,Jbm,Jbq);
#define J91 JIe W3v Jdj=JaJ?JIU*(JIe+1)/Jg2():JIU W0R Jcy=JaJ?1:Jg2()W0x(W33=W5g<Jdj W4d+=Jcy
#define J92 W0T>=8&&r>=256&&r%256==0&&t.s2<=W0H t.s2%W1I;}W11 Jds();W11 Je0();W11 JbL
#define J93 W0x(Jaj Z0G=0;Z0G<JIQ;Z0G+=JeB)W1q used=JIQ-Z0G<JeB?JIQ-Z0G:JeB W3v off=W1n(JJa+Z0G)W0y
#define J94 ,16,1,0,3,JHZ,W4g(Jaj ki=0;ki<4;++ki)Jc4(a2[ao+W2N],a1[JIG W1p+ki*4096
#define J95 (){Jb1 b1;}W00 Jb0<T>BorrowA2(){Jb1 a2;}W00 Jb0<T>BorrowB2(){Jb1 b2;}W00 W0B>BorrowC(){Jb1 c;
#define J96 )W0o();Div(JaI,ones,JaI,W0V>(JIv))W0o();JbG(JaI,JbH,JaI,0.0f,W3y,JIv
#define J97 C9EncodedCounts256<JIj,W1j(Z1f,aligned);countCol=aligned;}W07 Jcw>(Jao)W10 Jcw>(Jao)W3v W4h JaM(b)W14+
#define J98 );}W4Z{Jdx(4);Jam trace=Jbu;if(t.JbP&(W16 17))trace=W2H Jam>(Jbl<Jb3 JaM*>(C9ReadyTrace()W3A(
#define J8k ;W5c encoded;encoded W3I(W2I JIA*>(Jbl<W0k*>(JbN W3A())))W0b Z1f=fb W2n<W1y()[32768
#define J8l Jaj JbF(){if(Jbw())Jb1 256;{}if(JaH())Jb1 256;if(Jds())Jb1 W2O(Jhj JIU=W1n(W0U)
#define J8m (Jaj JIg,W0n ag,W5a,Jaj m,Jaj n,Jaj k,W3D,Jaj JJU,JI7 ovA,JI7 ovB,W0n pa,W0n pb
#define J8n 1u);if(JJO)W2E>(Jbx)W1b W2E>(JbO);}}W0A>(Jcf);W0A>(JbD
#define J8p JbA W1B if(Jbp){Jd3();W0W>(JaY)W2D>(JaY);}W2a if(W4G]){JaS<Jal
#define J8q (W0n qg,W3W,W5I,W0n vg,W3J os,W3J od,JI7 JJQ,Jaj JaB,W0n nextK
#define J8s W13(Z26){p.JaF(ab,196608);W2a p.JaF(ab,Jd9?262144:65536);}p.JaF(bb,JaJ?245760:(Jd9?131072:262144
#define J8t 4u W4K)Jc4(a2[mi W1L],Je5[JIs+mi W12],ap);W3j>(Jbx)W10 Jaw>(Jbx);JI0 Jb7
#define J8u packTask=Jaz()W0R packVC=Jbi()*2u W3v packLimit=(JJL+JaM(W0U)*t.s2)/16u W0x(W33=W5g<Jdj;
#define J8v 0,JHZ,W4g(W21=0;mi<8 W4K){J1g.n=W1M.k=W1M J5K{0,8,8,0,0,JHZ
#define J8w Jbf/512)Jau=t.Jbf/512 W1b if(t.s2>2048&&t.Jbf>2048&&t.Jbf%2048==0&&Jau<t.Jbf/2048)Jau=t.Jbf
#define J8x (W0n gs,W0n ps,W2j&t,JaM JIX,W1e){(JIm)t W0b zeros W2C>(0,8192)W0b raw W1o<
#define J8y ;Jbj(JIJ,JII,JI2(W0V>(t.s2)),W1f LT,A)W0o();JbG(JII,JIJ,JII,W0E,W3y
#define J8z JbU cp{4,128,0,896}W0O(dst[halfBase],raw,cp);W07 Z0y>(Jcf)W10 Z0y>(Jcf);}Z0Y:JaP
#define J8A if(JbA){W4Q>(JbO);JbA W1B W2a if(W4G]){W1X>(as);W4G]W5u
#define J8B 470);LoadQ();Mark(30);ScorePair();ScorePair();Jav<2,Jc8>(10);W07 Z0Q>(JbD)W10 Z0Q>(JbD
#define J8D );Jdx(2);Jaj actor=Jaz();W4Z{actor+=20;}Jdy(actor,0);W4Z{for(Jaj p=Jaz();p<W0D p+=40
#define J8E W0W>(JbO);JbA W3N;}}W3j>(as);W4G]W3N;W1W>(3+as);Z0v[as
#define J8F zeros,0.0f,8192)W0o();Jdg(gs,zeros,raw,JIX,col,JHZ);Jdg(ps,zeros,raw,JIX,col,JIj)W4t
#define J8G Jap,Jap,JaQ(-1),8192)W0o();Adds(Jap,Jap,JaQ(8191),8192)W0o();JdF(Jap,Jap,JaQ
#define J8H )),W1f LT,Q)W0o();JbG(JII,JIJ,JII,W0E,W3y,Q)W0o();Jbj(JIJ,JII,
#define J8I ,8192)W2v J4q(){Jb1 r>=W5s r%W1I&&t.s2%W1I&&JcB()==128&&JbF()==W0D W25;}W11 JeK
#define J8J T>(JJ1,JbB,JbN,go,o,JIR,t,trace,W2H Jam>(Jbl<W0k*>(Jdd W3A())));Jdx(5);JJ1.Reset
#define Jhy 1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JcT=W5h Z1B;W2u,JI2,Jax>(out,c,fp);W07
#define Jhz ;W5y A1>keyBuf;W5y B1>queryBuf;W5y A2>a2Buf;W5y B2>b2Buf;W5y CO1>cBuf;Jb0<T>a1,b1
#define JhA ,W1H,Q)W0o();JbG(JII,JIJ,JII,W0E,W3y,Q)W0o();W0N,W1j(JbX
#define JhB }W01 JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH-1)/JIH W3v JIU=W0d
#define JhD out W3A()W4u)out=out[keyRow W1Z]W0b Jbm=dkr;if(Jbm W3A()W4u)Jbm=Jbm[keyRow W2R];mm.C6B32KV(out,Jbm
#define JhE W02 Jbt>(JaY)W3C>(JaY);W31;mm.m=m;mm.n=W1M.k=k;mm.JbQ=JI6;mm.Jci
#define JhF W08 Jc7=W1n(W0U)*t.s1 W0R JIQ=(t.Jbf+Jau*8-1)/(Jau*8)*8 W0x(W33=vi
#define JhG W0b JbW=W5S>()W0b JcQ=JbW[W2b W0x(JaM JI6=W1n(Jaz())*16;JI6<JJL;JI6+=vc*16
#define JhH ,c,fp);W2a W2u,JI2,Jax>(JIq?od:os,c,fp);}W07 Jce>(JaY)W10 Jce>(JaY
#define Ji8 {Jcj(ft,JIv,nr)W0o();Add(JIf,JIf,JJP,JIQ)W0o();}Exp(JIf,JIf,JIQ);W0i
#define Ji9 Jcc(Jap,JaQ(1),N)W0o();Mins(Jap,Jap,JaQ(N-1),N)W0o();JdF(Jap,Jap,
#define Jib W0b Jap=JJ9 W2V>();Jap W1E(t.s2);Jat(rank,JaQ(-8),t.s2)W0o();Jcc(Jd2
#define Jic a2Buf,65536);JJ1.JaF(b2Buf,65536);JJ1.JaF(cBuf,131072);a1=keyBuf W2n<T>();a1 W1E(131072);b1=queryBuf W2n
#define Jid ==4096u&&t.Jbf W4s&&t.d==512u&&t.dr==64u&&t.JJk==32u&&(t.JJh W0S||t.JJh==1u)&&t.JIM&&t.JIZ
#define Jie =256;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JfN W1B fp.unitFlag=0;fp.JcT=W5h Je8 W0b
#define Jif (Je0())Jb1 W2O(Jdt())Jb1 W0D W08 JIU=W1n(W0U)*((r+63)/64)*((t.s2+127
#define Jih bp);}W2E>(JaY)W3C>(JaY);W5K;mp.m=64;mp.n=W4E.k=W4E.JbQ
#define Jii )W0o()W0b JaE=fb W2V>()[W2b;JaE W1E(t.s2)W0b JbR=W5S>()[12288
#define Jik .JJz=W2r.JbK=W2r.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JcT=W5h Je8
#define Jil .Jhw=fp;rp.JJz=64;rp.JaB=64;JbT<T,JI2,Jax>(Jbm,c,rp);}}W07 Jce>(JaY);Jbp W3N
#define Jim Jaj JIA=0;JIA<2;++JIA)W1q JIg=pass*2+JIA,Z19=JbZ*2+JIA;if(Jdn W3f]>=4)JcS;Jaj JcK=4,Z11
#define Jio W3Z;fp.Z0E=W2r.JJz=W2r.JbK=W2r.JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.unitFlag
#define Jip W3Z;fp.Z0E=W2r.JJz=W2r.JbK=W2r.JaB=W2r.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA
#define Jis W3Z;fp.Z0E=64;fp.JJz=256;fp.JbK=64;fp.JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA
#define Jit J2b));JyO*>(JgV));JaM pos=0;eq=q;ek=k;JbN.J4X*4;wq W3I(W4m*>(JcU+pos));
#define Jiv ,++JJ2)W1q JIg=JJ2%4;Jav<2,Jc8>(W3B)W0R W4x(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs
#define Jiw SqDrain();Jbg<2,JdH>(W3B);}for(W3Y=0;JJ2<2;++JJ2)Jav<2,Jc8>(JJ2*2+1);W5U
#define Jix W13(TA){Jan p{0,W1U>(k/16),1,0,W2q(k/16-1),W4g(W21=0;mi
#define JhI {JdS(mm);JcD W4c>();if(JaJ){JJ1.Reset();mm.Init(JJ1);}JeT(mm);}}}W4Z{if(Jau>1
#define JhJ JIa?(JIq?PSL:PSH):(JIq?CL:CH))+Jeg*32 W0R dst=JIq?ALOW:0 W0x(W21=0;mi<2 W4K)Jas
#define JhK 3+JIK);Z16 W2z]W5u if(bMath W2z]){W0A>(5+JIK);bMath W2z]W5u}}Z0Y:W03
#define JhL Z16[as]W5u W13(!TA){Jan ap{0,8,1,0,7,W4g(W21=0;mi<8 W4K)Jas(a2
#define JhM ,8192)W2v();}J5b f[12288];fw=f[W0K;ft=fw[512];JaI J1K()?ix:ix[t.Jbf]W4t Jei
#define JhO ()W1q vc=Jbi JtJ();{if(dqr W3A()W4u){JbU cp{16,4,28,0}W0O(dqr W5w*64],J0e){
#define JhP ]>=4)Jb1;Jaj JcK=4,Z11=0 W0x(W4H=0;j<4;++j)W1q id=84+Z19*4+j W0b JK5=Jdd[JaM
#define JhQ JcX();JI0 Jbz cp{1,1,1,32,32,W4g(Jaj JIk=0;JIk<4;++JIk)for(W1e=0;col<256;W2U
#define JhS ){JIb JII=W5R>();JII J6T/32));JJZ();JbX W1E(2*JaD+8);Jat(JbX[2*JaD],W0E
#define JhT 8192);JJZ();}J6w);}Add(dp J5W<T>psLow J0Y];ft J0P<Jar>()[W5O;slotMask W1E(256)J3Z W0h>();Jdf
#define JhU J3N t.s2-1,JaQ(-8))W2v();Sub(JaE,rank,Jet,t.s2)W0o();JdO(JIu,JaE,JaQ(3)
#define JhV ){Jat(JIu,0,t.s2)W2v()W0x(JaM JIh=Jaz();JIh<Jc7;JIh+=vc){JaU(JbN[JIh*t.s2],JIu,
#define JhW ;mp.k=W4E.JbQ=JIG W0S;mp.Jci W59(c W39 W20],a2[ao],b2[bo],mp);W1W
#define JhX data;r=t.Jbc):1;{if(t.s2>2048&&t.Jbf>=8192&&t.Jbf%512==0 J50==0)Jau=1;if(JbC())Jau
#define JhY JaU(JJ8,JbB[JJ7*W1m,4096)J75(0)JBR J3i 8),s Jhs[s/8],JbR[s],Jd2,W1H,s)W0o
#define JhZ W08 at=JaM(x W0T)*2097152+JaM(x.key)W0y+x.feat*W0D JbT<T,JI2,Jax>(x.W57 3?dg[at]:
#define Ji0 ++key)W1q JIK=key%2u,ao=JIK W20,W4x key W0w;J3d::Jbt>(JaY);W31;mm.m=W1M.n=W1M
#define Ji1 )Jdl;(JIm)Jct;J1j 1,0}W0O(b1,prefixQ,rp);}}for(Jaj JIq=0;JIq<2;++JIq){if(JIq==0){JaU
#define Ji2 J4g,JIf,JJP,JIQ);Mul(JIf,JIf,mult,JIQ)W2m(fw,fw,W0E,JIv);Jcj(fw,JIv,nr);Add(dp,dp,JJP
#define Ji3 }W07 Jff>(ev);W1g b]W3N;}for(Jaj b=0;b<3u;++b)if(W1g b])W06 Jff>(dEv
#define Ji4 (Jfa?ka:va),(gap+Z3k)/Z18(T))W0R gBase[2]={0u,49152u};JI0 Jb5 gEv[2]={4u,5u}W1O busy
#define Ji5 ;Jdh<T>();if(x.feat==0){fp.JJz=64;fp.JaB=64;JbT<T,JI2,Jax>(dqrg[JaM(x W0T)*8192
#define Ji6 W3f]W5u}H2GDrain(J6Q>(out,c,fp);}}if(J3Y,c,rp);}}}W07 Z0U>(Jbx);JfU W3N;
#define Ji7 )W1q ns=1u-JIK;if(l1[ns]){W1X>(ns);l1[ns]W5u JaU(a1[ns J2z>(col)W1Z
#define Jiz J3m aOff=W1P 8192,bOff=W1P J1Y mi J4O+h*(k*nH)]J1Z[as]W3N;W2a JcM();J5O]J15 SqDrain(){for(Jaj s
#define JiA <W1j(ps[wo],ps[JcW+wo],go[qo],Jcm,JHZ,row+W1D r,r-row-128,Jbm,Jbq);}}W2e
#define JiB );a2=a2Buf W2n<T>();a2 W1E(32768);b2=b2Buf W2n<T>();b2 W1E(32768);c=cBuf W5Q
#define JiD Jeu(ft,JbW,64,128,1,1,8);Jeu(JaI,ft,8,16,1,1,1)W2v()W0O(JIR[JI6],JaI,16
#define JiE i){if(mm.W4F i]){W1X>(i);mm.W4F i]W5u if(mm.Z0v[i]){W0A
#define JiF ;i<2u;++i){if(W4F i]){W1X>(i);W4F i]W5u if(Z0v[i]){W0A
#define JiG Jaj tile){Jaj Jcr=0u W0x(W4H=0;j<N;++j)if(!(st.c3ReadySeen&(1ull<<(tile+j W4A))))Jcr|=W16
#define JiI h2PackedQ[(JaM(b)*W18)W1s]:eq[qo],Jcm,row==0,JHZ,r-row-128,Jbm,Jbq,JHZ,f==0?256u:512u
#define Jj9 >(nk),256,0,512,W2q(nk),1,0};if(W50]){W1X>(ns);W50]W0s
#define Jja W2n<Jar>()[W5O;JIJ W1E((t.s2/8+31)/32*32);JIu W1E(t.s2)W0b rank=JIu;rank W1E
#define Jjb {JJ1.JaF(fb,JbC()?99072:(4*4096+1024+32)*4)W0b f=W5S>();JIf=f;dp=f[4096
#define Jjc &&W0U==1u&&t.s1==32u&&t.n1==8u&&t.s2 W4s&&t.Jbf W4s&&t.d==512u&&t.dr==64u&&t.JJk==32u
#define Jje ,++JJ2){J5J*nr],JIQ)W0O(dp,W2y+JeM+JI6*nr],JIQ)W2v()W0O(fw,JIR[JIX],J4l>()
#define Jjf JId*8+j)J1B b J4H J2M(JIe,JIW),b W2s/32,Z0G W2s%32 J5o]&J28();if(vid J6R);J5C(JIe,pass)W0x
#define Jjg W01 nfirst=sub*64+(JIk+1u)*32u,nb=(JIk+1u)%2u J77>(nev);}for(W4H=0;j<nRows;++j){J3p
#define Jji W0b Jc0=c W39*m W14];Mmad(Jc0,a2[ao],b2[b2o],mm);W1W>(5+bs);JJN[bs]W3N
#define Jjj W0s J4m 1,Jhq.m=m;mp.n=n;mp.k=kp;JoE=JaB;fp.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA
#define Jjk AFree(as);Jay ap{0,8,8,0,0,JHZ,W4g(W21=0;mi<8 W4K)Jc4(aa[as W1p+mi*2048
#define Jjl (W4p,W0a J00>(TA?k:m),1,0}J7P;if(JeR)Mmad(Jc0,a2[ao+W2b,b2[b2o],mm)
#define Jjm )W0x(Jaj JJ7=0;JJ7<8;++JJ7)W1q at=JJR*16+JJ7*2,w=128+12*JJ7;Word(w,at);Mark(w+
#define Jjn 3);KVPlane(at,0,dst);Mark(w+4);if(JJ7==7){L1ReadDone();keyBase=0;DQResident(b,JJR*16+12,131072);Mark
#define Jjp )*r*t.s2;if(at<rt*JrH=W3K;Z2v=W3K[(W1n(b)*W5B)*64];W5U
#define Jjq {for(W4H=0;j<N;++j)if(Jcr&(W16 j))W1q id=(tile+j W4A)*2u W0b a=st.J7b ALL>
#define Jjr W21=0;mi<m/W4L)Jas(a2[ao+mi*k*16],a1[ao+mi*k*16],p);}J4L[bs]W0s
#define Jjt >(ps[wo],ps[JcW+wo],go[qo],Jcm,mr,kr,t.s2,JHZ,row+kr==r,r-row-kr,Jbm,Jbq);
#define Jju ;mm.C9RunKVTagL0B((JIa?ps:gs)[wb],(JIa?go:eq)[qi W1Z+feat],qr[qi W2R],out,rp,feat,!JIa);
#define Jjv J21 at<rt*JrH=W3K;Z2v=W3K[(J6u%kt)*64,mr=t.W4z<64?t.W4z:64;J6p&&!val)
#define Jjx ,J1L*16],src,cp);JaN<J4i{1,1,1,8,8,8};Jhv W47)*16+JaM(col W47))W1p+((JIX
#define Jjy *16],src,cp);JaN<J4i{1,1,1,8,8,8};Jhv W47)*8+JaM(col W47))W1p+((JIX%128
#define JiJ (id)*32];J03(J1u<T>av;J58+40*J2l(gs,ps,t J6e)J7S[qi*64],out,dkr[ki*64],J30(20+
#define JiK ko+ff],64,128,128,512,512,ff==0,ff==0,qr,kr[kp]);}if(c>0)W1q p=c-1,JIq
#define JiL W0R maskBytes=((Q/8+31)/32*32)>((Z1a/8+31)/32*32)?((Q/8+31)/32*32):(
#define JiM ],mr,nr,nr);J0l(q[qo],go[qo],k[ko],v[ko],W2y],W2y+W0K,nr);W0C
#define JiN JaR<JJ0 T,JI7 ROPE,JI7 Z01,JI7 JdN,JI7 Jdq W0s,JI7 Jbk W0s,JI7 Z08 W0s
#define JiO Jbd<W2G>();W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<JcR
#define JiP go,Jam JcP,Jam maxG,Jam sumG,Jam idxG,Jam gradQ,Jam gradK,Jam gradV,Jam user,W2j
#define JiQ JbG(rank W3T>(),JIJ,JaE W3T>(),rank W3O
#define JiR ,W1d>JbB,W1d>JbN,W0n go,W0n o,W0f
#define JiS =1;fp.Jbo=0;fp.JbY=0;fp.W30 W2u,JI2,Jax
#define JiT );W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E
#define JiU .JaB=W4Y.JJg=1;fp.Jbo=0;fp.JbY=0;fp.JcA W0s
#define JiV JaZ(W4m*>(JcU+pos));JqG+pos+(Jcx()-1)*JJf
#define JiX W00 JaQ LocalCompleteRead(Jaj id)W44 LocalCompleteAcquire(Jaj JIi,Jaj&Z1f);W11 C7WideGate()
#define JiY <T>&mm,W32,Jaj&JJK,Jaj*Jcs,Jaj*Jdm,Jaj*Jcg,Jaj&Z2A
#define JiZ )W0o();W07 Jck>(JaK)W10 Jck>(JaK)
#define Jj0 J82);W13(Z2H||PREFETCH_LOW)if(JeR)JaU(a1[ns W1p+W2b,low[Jd6
#define Jj1 JIm Z3h(Jc9<T>&mm)W44 Z3i(Jc9<T>&mm)W44 Z3j
#define Jj2 Jan bp{0,8,8,0,0,W4g(Jaj ki=0;ki<8;++
#define Jj3 ;W5c encoded;encoded W3I(W2I JIA*>(Jbl<W0k*>(JbN W3A
#define Jj4 JaR<JI7 TA,JI7 Z2H W3N,W40>W03 Jdp(W4p,W0a J00
#define Jj5 JJg=1;fp.Jbo=0;fp.JbY=0;fp.W30 fp.JcT=W5h
#define Jj6 ,JIa,Z1m,Z1t,JcP,Z1w,sums,Jfe,JIo,gradQuery,gradKey,gradValue,Z3v,JgV,JcU
#define Jj7 bg,W5d>out,Jaj m,Jaj k,W3D,W1J,W1N,Jaj JbS,Jaj featOff
#define Jj8 JaR<JJ0 X>W00 Jb0<X>JaL(W5y Z17>&a,Jaj byte,Jaj size)
#define JjA JIf,JIQ)W0o()W2m(dp,dp,t.JJq,JIQ);W0i
#define JjB W0l();W07 Jck>(JbO)W10 Jck>(JbO)W0O
#define JjD W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=m
#define JjF W2E>(JaY)W3C>(JaY);W5K;mp.m=64
#define JjG Jcj(JaI,JIv,nr)W0o();Mul(JIf,JIf,JJP,JIQ);JaN
#define JjI ,t.s2,k[ko],v[ko],pk,pv);W2a vec.JaR JfI<
#define JjJ q[qo],go[qo],k[ko],v[ko],W2y],W2y
#define Jka for(JaM JI6=Z2w+W1n(Jaz())*16;JI6<rowEnd;JI6+=vc*16
#define Jkb t.Jbf>=8192&&t.Jbf%512==0 J50==0)Jau=1;if(JbC())Jau
#define Jkc W03 Jcn(W0n a,W0n b,JI7 pa,JI7 pb){Z0X=a;
#define Jkd W11 JeK();W2t JbF();W2t JcB()W44 JdS(Jc9<
#define Jke W01 vc=Jbi()*2 W0b JbW=W5S>()W0b JcQ=JbW[
#define Jkf ,t.s2,k[ko],v[ko],pk,pv);}W2a vec.JaR GatherPackedTile
#define Jkh J87)W2v();JJH(mult,Jc2,W1G,JIQ)W0o();}
#define Jki W00 JaQ JdY(Jaj id)W44 Jdy(Jaj id,JaQ JIa);W11 JdE(){
#define Jkj );}}J52{Jc4(a2[ao+mi*k*16],a1[ao J7k J4V;J80 0;
#define Jkk ,t.s2,k[ko],v[ko],pk,pv);}Jbg<2,JeU>(
#define Jkl .Jbf==8192&&t.s2==W5P W0U*t.s1<=64;}W25;}W00
#define Jkm }W2e C2RawHalfHandoff();W11 C2CachedWeightsGate();JaR<JI7 MULT>W03 JgS J5a,Jar
#define Jkn W2e JeY(Jaj Jdi,Jaj JdV)W44 JdX(Jc9<T>&mm,W33)
#define Jkp r%W1I&&t.s2%W1I&&JcB()==128&&JbF()==W0D W25
#define Jkq .JJq,JIQ)W0o()W2m(ft,ft,W0E,JIv);W0i
#define Jkr ,JaM(64),Jar(64),cp);}W0l();JcF();}JaP
#define Jks ){if(Z1j W2z]){W1X>(JIK);Z1j W2z]W5u
#define Jkt W03 Jf1()W44 Jgg()W44 JeL()J13 J63 t.s2/8+
#define Jku Jbd<W2G>();JaA<JaM,Jda::JgC,JcL::Jca>(x);Jbd<JcR
#define Jkw JaU(JIf,W2y+JI6*nr],JIQ)W0O(dp,W2y+JeM+JI6*
#define Jkz W03 H3GradTask(Jbe<T>&mm,Jaj JIi)W44 H3Gradients(Jbe<T>&mm)
#define JjK JaT JIm PrepareToken(W2j&t,W1d>JbB,W0n go,W0n o,Jaj
#define JjL if(mm.W50]){W1X>(ns);mm.W50]W0s
#define JjM ;W5c halfCountsGm;halfCountsGm W3I(W2I JIA*>(Jbl<W0k*>(JbN W3A
#define JjP for(Jaj i=JJa;i<JJ2;++i)Jav<2,Jc8>((i%4)*2
#define JjQ if(W3V]){W0A>(3u W5D);W3V]W5u
#define JjR );mm.LoadKV(pk,pv);mm.ScoreDP(JIf,dp);Jbg<2,JdH>(1
#define JjS ;W5c Jfj;Jfj W3I(W2I JIA*>(Jbl<W1Y*>(JcP W3A
#define JjT W03 JfC(Jb0<T>JcU,Jb0<T>sharedA2,Jb0<T>sharedB2,W0B>sharedC
#define JjV (Jc6,JIg,kvRing,dk[ko],dv[ko],t.s2,skipKV?vec.KvPackets(JIg
#define JjW ;W00 Jb0<T>BorrowA2();W00 Jb0<T>BorrowB2();W00 W0B>BorrowC
#define JjX (W0n gs,W0n ps,W2j&t,JaM JIX,W1e)W44
#define JjY ==0){W3E,Jda::Jc1,JcL::Jca>(JK5);Jbd<W2G>()
#define Jk1 W0n out,W0n Jbm,W3s,W3D,Jaj newTag,Jaj&Jeq,Jaj&Jd8
#define Jk2 ;Jc2=JJ9 W2V>()W4t PrepareBatch(Jaj JId);Z0Y:W03
#define Jk3 {Jd0(W55,W1z,"fused32 vector helper is FP16 only");W5y Z17>JIF;Jaj sub=0;
#define Jk4 W3J out,Jaj m,Jaj n,Jaj k,Jaj Z1Y,W1J,W1N,Jaj Z2d,Jaj
#define Jk5 W52{0,64,1,0,0,JHZ,0};Jc4(b2[bo],b1 W2z
#define Jk6 JI0 Jb7 qc{1,128,512,0,512,128,1,0}W0O(a1,qg,qc
#define Jk8 W03 JfL(W0n qg,W3W,W5I,W0a J7m,
#define Jk9 W0l();JcF()W4t C2CompactCounts(W0B>dst,W0B>JbE
#define JkB ;Z1w W3I(W4m*>(maxG));sums W3I(W4m*>(sumG
#define JkC ],8192)W2v();Jab],JaI[16],16)W2v();}}W00
#define JkD if(n>=64&&n<=2048){Jaj Q=64;Z0H(Q<n)Q*=2 W0R
#define JkF Jay ap{0,8,4,0,0,JHZ,W4g(W21=0;mi<4
#define JkG Jay ap{0,8,8,0,0,JHZ,W4g(W21=0;mi<8
#define JkJ JIb JJ5=W2I Jar*>(Jbl<W0k*>(Je3 W3A()));W0f
#define Jlb <T>&mm,Jaj Jdi,J54 Jcj(W0B>src,Jaj JIv,Jaj nr);JaR<JI7
#define Jlc ){JIb JK5=Jeh(Jaz());W2Z*p=Jbl<W0k*>(W5k
#define Jld {Jd0(W55,W1z,"fused32 draft is FP16 only");W5y A1>ab J81>c W1O qResident
#define Jlg W5U(qr,kr,JHZ,JHZ)W4t Jdz(Jc9<T>&mm,Jaj Jdi
#define Jlh ,cp);W07 Jff>(JaK)W10 Jff>(JaK)W4t
#define Jli (){Jb1 W0P!J8O Jbw(){W5j;}W11 JbL(){{}W25;
#define Jlj W0o();JbG(JII,JIJ,JII,2048.0f,W3y,2048);W0i
#define Jlk if(JIi>=32u){W1d>Z0K;Z0K W3I(W4n*>(wq W3A(
#define Jll Jav<2,Jc8>(10);W07 Z0Q>(JbD)W10 Z0Q>(JbD
#define Jlm 0,JHZ,W4g(W21=0;mi<8u W4K)Jc4(a2[ao+mi
#define Jlp W4P>(JaY)W3h>(JaY);W3Z;fp.Z0E=128
#define Jlq W5U(qr,kr,JHZ,JHZ)W4t Jdz(Jbe<T JmC;}Jjz-nr
#define Jlr W0l();JbG(JII,JIJ,JII,W0E,W3y,2048);W0i
#define Jls (W0n dst,W0B>f,W27,W1e,Jaj nr,Jaj JIv,JI7 Z2u)
#define Jlt W1F Jdo,JIQ)W0o();W07 Jck>(JaY)W10
#define Jlv W0l();JbG(JII,JIJ,JII,W0E,W3y,A);W0i
#define Jlw W05 Z0q>(JaK)W10 Z0q>(JaK);if(JJ2)Jav<2,Jc8
#define Jly JaR<JI7 TA,W40 J55(TA){if(!Jev||Jeq!=Z2B){Jeq=Z2B;Jfs=
#define JlA ;q W3I(W2I T*>(JIB));k W3I(W2I T*>(key
#define JkL }}}J8c JaT JIm JJx(Jc9<T,W1j&mm,W33)W44 HrWork();
#define JkM <W1j(descNext,nslotBegin,t.s2,k[nko],v[nko],nk,nv);}
#define JkN (){Jb1 W55,W2g W0F!W0P Jdq&&Z08&&!JdN&&!W1v
#define JkP Jbg<2,Jc8>(9);W07 JcV>(Jao);JaS J8b=8){JI0 Jb5
#define JkS W0R JJa=(JIi%Jau)*JIQ W0R n=t.Jbf-JJa<JIQ?t.Jbf
#define JkT W5I,W0n vg,W3J os,W3J od,JI7 JJQ,Jaj
#define JkX mm.Jd3();W0W>(JaY)W2D>(JaY);}if
#define JkY J8d=0;mi<8 W4K)Jc4(a2[mi*k*16],a1[W1x J5c
#define Jl0 W06 Jcw>(JaK)W1b W06 Jcw>(Jao);}if(st){JaU
#define Jl2 W4P>(JIr)W3h>(JIr);W3Z;fp.Z0E=128
#define Jl3 Jaj JJ7=vid+40*Jc3;av.PreparePairCountsHalfGm(t,JbB,JbN,go,o,JIR,JJ7,Jc3
#define Jl4 <T>&mm,W32,Jaj&JJK,Jaj*Jcs,Jaj*JJm,Jaj&Z1S,Jaj&jobs
#define Jl8 (W0B>dst,W0B>stat){JIb JIs=JJ9 W5Q>();JcX
#define Jla W3O<Jaj>(),Jaj(0x3f800000),odd,Jar(128),Jb5(1),Jar(
#define JlC &&(JIq W0S||hasNext)){W3j>(JIK)W10 Jaw>(JIK
#define JlD >0u){if(JJO)W0A>(JbD)W1b W0A>(Jcf)
#define JlE readyReplay&&x.W57 1){Jbg<2,JdH>(liveProduced%11);++liveProduced;}W0C if
#define JlF )W0x(W33=W5g<Jdj W4d+=Jcy,++JJ2)W1q JIg=JJ2%4
#define JlG J1b 64,128,1,0};J62=0;mi<8 W4K J8W,rp);}J1U{
#define JlI W20],bp);W2E>(JaY)W3C>(JaY);if
#define JlJ W02 Jb9>(JaY)W0O(b1[tail],qg[448],tcp);W0W
#define JlK W3v Z1T=C3RFTask(JIi)W0R W4x(Z1T%kt)*JIH,W4h(Z1T/kt
#define JlL JdW(1,1,8,1),extracted)W0o();Jbj(JIJ,JeS,1.0f
#define Jmc (Z01?1:J8Z(){if(JdE())Jb1 W0R vc=Jbi JtJ();{J4c
#define Jme ,Jar(8),Jbv{1,8})W0o();JcF();C7CompactRows64 W4c
#define Jmf 8192);W07 JcI>(JbD);}}if(st){W06 JcI>(JbD
#define Jmg +W2N],bp);W2E>(JaY)W3C>(JaY)
#define Jmh W1q JJR=Z13<32u?Z13/4u:Z13/8u+4u W0R need=Z13<32u?Z13
#define Jmi (W0n qg,W3W,W5I,W0n vg,W0a
#define Jmj ,JaE W3O<Jaj>(),0,N)W0o();J4r W0h>(
#define Jmk ,Jar(4),Jbv{1,8})W0o();JcF();JgS W4c
#define Jml ,Jar(4),Jbv{1,8})W0o();JI0 Jbz cp{1,1
#define Jmm ;Jea=qo;}Jbg<2,JdH>(W3B);}W01 JJa=JJ2>4?JJ2
#define Jmn ;W0n Jc0=val?dv:dk;if(Jc0 W3A()W4u)Jc0=Jc0[JaM
#define Jmq (n W4s){JIb Z1N=tb W5Q>();Z1N W1E(2048u);Jb5 lengths
#define Jmr JHZ;}W0X Z1D=col>=n0 W0R n=Z1D?192u:n0,Z0f=Z1D?n0 W1Z
#define Jms JaF(ab,262144);p.JaF(bb,262144);p J51 W0m>()W4t
#define Jmt W5y A1>l1b;W5y A2>a2b;W5y B2>b2b;W5y
#define Jmw JIj;}}W3j>(as);mm.W4G]W3N;W1W
#define Jmx W01 Z1T=Jaj(st.C3RFTask(JIi)),W4h Z1T/8u W2R,W4x Z1T%8u*
#define Jmy for(W21=0;mi<2u W4K)Jas(a2[j W1k+mi W38],Je5
#define Jmz ::Z17>tb,ib,JJ9;W0B>JIf,dp,JJP,mult,fw,ft,JaI;Jb0
#define JmB JaQ Jgc[16]W0x(Jaj W4h 0;row<16;++row)Jgc W5w]=JII.Jco
#define JlM )JcS;if(t.JbP&(W16 25))Z23(40+2*jobs,(JaM(Jcu(
#define JlN W0b JbW=W4T>(a,0,8192),JcQ=W4T>(a,32768,8192
#define JlO 0;W0X Jdl=((t J2V J07=J5A));}}J1y(Jdn W3f JhP JiJ Jjf(Jim
#define JlP (W4C){W5i{1,W2q(k),64,0,64,W0e
#define JlQ ;if(Z2u){JJH(raw,JJP,W1F Jdo,JIQ)W2v()W0O(dst[JcW
#define JlR );JJ1.JaF(a2b,65536);JJ1.JaF(b2b,65536);JJ1.JaF(cb,131072)
#define JlT ;W3Y=0u W0x(Jaj JIi=C3ScoreOwner(JIe)W4d<32u W4d+=20u,++JJ2){
#define JlU 2048u,JJi W0S,JJi==15u,(15u-JJi)W0w,feat,GUARDED?&Z2L:Jbu,Z35
#define JlX {1,W2q(k),W2q(n),0,512,W2q(
#define JlY W2t Jcx(){if(t.s2>W5P r<=64)Jb1 8 J4t(){{}Jb1
#define JlZ W0l();Adds(JII,JII,1.0f,2048)W0o();W0N
#define Jm0 ;++jobs;ran W3N;if(t.JbP&(W16 17))pollBegin=Jcu();Z0R;
#define Jm1 W1q bs=JIr%2,b2o=bs W1p;if(JJN[bs]){W0A
#define Jm2 W4Z{JJ1.JaF(fb,JbC()?99072:(4*4096+1024+32)*4);
#define Jm3 fp);fp.JaB=W4X JbT<T,JI2,Jax>(dqg[JaM(x W0T)*65536
#define Jm4 JFX vid+40*Jc3<64;Z0R;}}}if(Z1n)JcS;if(Jc3<Jfp){JI0
#define Jm5 );W07 Jce>(JaY);mm.Jbp W3N;mm.Jcz=0;}}};
#define Jm6 if(W34 0&&JbS>0)W1q nk=JbS<128?JbS:128,ns=1-as
#define Jm7 JHZ,JHZ W4f(Jaj JIq=0;JIq<2u;++JIq){for(Jaj JIG=0;JIG
#define Jm8 ){W5i{1,W2q(nk),64,0,64,W2q(nk)
#define Jm9 JbT<T,JI2,Jax>(drg[JaM(x W0T)*262144+JaM(x.key)*64
#define Jma *8+8;++JIh)Z0H(JdY(100+JIh)==0){}JaU(Z1f,encoded[JaM
#define Jmb JIb JbX=W5S>();JbX W1E(2*Q)W0b JJA=fb.JaR
#define JmD JJ2){Jav<2,Jc8>((acked%4u)*2u+1u);++acked;}mm.H2R28Drain
#define JmE (t.s2>2048&&t.Jbf>=8192&&t.Jbf%512==0&&Jau<t.J8w/2048
#define JmF for(Jaj bit=Q;bit>0;bit/=2){Adds(JaE,rank,W0V>(bit
#define JmG JIy{if(out W3A()W4u)JbT<Out,JI2,Jax>(out,c,fp);
#define JmJ JHZ}W1O Z0N[2]={JHZ,JHZ},Z0B[2]={JHZ,JHZ};Jaj sliceStep
#define JmK }for(Jaj JIK=0;JIK<2;++JIK)if(W1g JIK])W0A>
#define JmL );Jat(JK4,JaQ(-1),8)W0o()W0b refs=JK4[8
#define JmM *t.n1;JJL=W1n(W0U)*r;JcW J6g<Jaj>(vc/Jc7
#define Jne JIb JJ8=JIg W0S?c7Packed W2n<T>():JJ9 W2n<T>()[2048u
#define Jnf mult,JIQ)W0o();}Muls(fw,fw,W0E,JIv);W0i
#define Jng qr,Z2v,JHZ,ROPE);if(Jga(W1n(b)*W18,col,mr
#define Jnh W20],JJS[Jd6],nap)W0O(b1[ns W37],bg[W1n(
#define Jni );fixes+=2;W2a JbT<T,JI2,Jax>(dqg[JaM(x W0T)*65536
#define Jnj ++JJ2)W1q W4x(JIi%8u)W12,W4h(JIi/8u)W0w W3v
#define Jnk JIb z=zeroBuf W2n<T>();z W1E(8192);Jat(z,W0c T>
#define Jnl W5u if(Z0B[1]){W0A>(JaK);Z0B[1]W0s
#define Jnm for(Jaj ki=0;ki<4u;++ki)Jas(b2[ki W1L],b1[ki W12
#define Jnn JaM>(Jaz())*16;JI6<JJL;JI6+=vc*16){JaU(raw,go[JI6
#define Jno =JIe%10u W0R nr=JIr<4u?448u:384u,W4x JIr<4u?JIr*448u:1792u
#define Jnp J0u if J5M],mr,J7W()J4n Jc9<T JmC;}Jjz-nr)/J4Y(J7H],JIQ
#define Jnq Jaj JIK){JI0 Jb7 cp{1,256,256,0,512,256,1,0}W0O(b1
#define Jnr ;}for(Jaj JIr=0;JIr<2;W2M)W1q bs=JIr,b2o=bs W1p
#define Jnu ;mp.n=n;mp.k=kp;mp.JbQ=JIG W0S;mp.Jci W59
#define Jnv JaR<JI7 TA J4v(Z0f)*W0D Jb7 bcp{1,128,256,0,512,128,1,
#define Jnw );JJ1.JaF(a2b,65536);JJ1.JaF(b2b,65536);JJ1.J6s<T>go)
#define Jnx W5u if(Z0N W39]){W0A>(3 W5C);Z0N W39]
#define Jny ;mp.n=32;mp.k=W4X mp.JbQ W3N;mp.Jci W59(c
#define Jnz ;J4R ns*W0K,JJS[Jd6],nap);if(JeR)JaU(a1[ns W1p+8192
#define JnA 64){W13(MULT)Mul(dst[col],dst[col],JIs,JaM(64),Jar
#define JnB H4CellMte1();W5K;mp.m=64;mp.n=W4E.k=W4E.JbQ
#define JnC (a1[ao],JJS,acp)W0O(b1[bo],bg,bcp);W0W
#define JmN k;Jb7 nap{1,W2q(TA?nk:m),J71>(ns);W50
#define JmO %10 W0R before=2*(av/10)+(rem<2?rem:2)W0R JI6
#define JmP ;++s)if(Z1y[s]){W1X>(s);Z1y[s]W0s
#define JmR W17+JI6;if(t.JbP&(W16 16)){W09::J40>();W0C
#define JmS for(W1e=0;col<nr;W2U 64u)W1q JIK=(col/64u)%2u,
#define JmT W0l();W2a for(W4H=0;j<n;++j){JI0 JaQ p
#define JmU ;JcX();Jbz cp{1,1,1,16,16,W4g(Jaj g=0;
#define JmX <T>(JJ1,JbB,JbN,go,o,JIR,t,0,W1n(W0U
#define JmY W0z],JaM(64),Jar(8),cp)W0o();JcF();
#define JmZ J74[W1n(key)W1Z],low[vOff W17 W1Z],rcp);}W4M
#define Jn1 Jan ap{0,8,1,0,7,W4g(W21=0;mi<8;++
#define Jn3 for(W4H=0;j<16;++j)W1q tile=JI6+j,W4x tile W14
#define Jn4 JJ1.JaF(fb,(4*8192+1024+64)*4)W0b f=W5S
#define Jn6 );if J6n],View W2T>(169088+cacheItem*4096 W0z*4,512),512);J6o
#define Jn7 ;if(Je6){JI0 W5i{1,128,64,0,64,128,1,0}W0O
#define Jn8 if(aa W2z]){W0A>(4u+JIK);aa W2z]W5u
#define Jna W0O(JIf,wq W5w W1t,8192)W2v()W0x(Jaj JIr=1;JIr<
#define Jnc 8,10},recycle[2]={9,15}W0b JII=W2o W2T>(J0b
#define JnE &&(t.s2==2048||t.s2==4096)&&Jc7>=80&&Jc7%2==0;}W25
#define JnF JaU(dk[off],raw,used W0y);if(hasDv)JaU(dv[off],raw,used
#define JnG JIY()W0x(Jaj r=0;r<used;++r)if(inactiveMask&(Jaj(1)<<
#define JnH JaU(JJP,Jee[(JIr-1)*JJf W17 W1t,8192)W2v();Add(
#define JnI ,JIf,JIX,col,J1W;}Jhu=1;J4j/(rt*kt J5i J7F}J87);J4g J6D
#define JnJ kt)W0R mr=r-row<Jbs?r-row:Jbs,nr=t.W4z<JIH
#define JnK /16),0,0,W4g(Jaj ki=0;ki<k/16;++ki)Jas
#define JnL 8,1,0,7,W4g(W21=0;mi<8u W4K)Jas(a2
#define JnM ],b2[bo],mp);W1W>(3u W5D);W3V]W3N
#define Joe W5u Jay ap{0,8,1,0,7,JHZ,W4g(Jaj ki=0
#define Jof /J5k){(JIm)t W0b zeros W2C>(0,8192)W0b raw W1o<J24
#define Joh .J4s T*>(Jfe));JyO*>(JgV));JaM pos=0;eq=q;ek
#define Joi i);mm.Z0v[i]W5u if(mm.JJN[i]){W0A
#define Joj W0R W4x(JIi%kt)*JIH,W4h((JIi/kt)%rt)*Jbs,b
#define Jol >(3u+i);Z0v[i]W5u if(JJN[i]){W0A
#define Jom overlayK){W5i{1,256,64,0,64,256,1,0}W0O(a1,krg
#define Jon ),3072);if(Jek)tr.J2B,64,64,1,1,8)J73 W05 Jcw
#define Jor Jb7 cp{1,128,128,0,512,128,1,0}W0O(b1[W1P W0K
#define Jos >(3+i);Z0v[i]W5u if(JJN[i]){W0A
#define Jot JIm Ji1(a1,J6l{0 J94],ap);J3O],gog[448],tcp);JaW<J6v
#define Jow 32),cp)W1b Add(dst[col],src[col],JIs,JaM(64),Jar(
#define Jox ){W13(MULT)Mul(dst[col],dst[col]J8a>();JcF();}
#define Joy ,0,64,Jiz=0;s<2;++s)if(Jfr[s]){J5p,Jaj Z2d
#define Joz W5i{1,128,64,0,64,128,1,0}W0O(a1,Z0X,rp)
#define JoA ){JI7 l1[2]={JHZ,JHZ},amath[2]={JHZ,JHZ};Jd3()
#define JoB ;W52{0,32,1,0,0,JHZ,0};Jc4(b2[bo],b1[
#define JoC J0S>();J0W-b2:t.J0K>();}}W2a if J4k();{Jgb
#define JnO +as);Z0v[as]W3N;J23;fp.Z0E=m;fp.JJz=FN;fp.JbK
#define JnP +64],main);Jb7 rope{1,128,64,0,64,128,1,0}W0O
#define JnQ +JIk W4A,b=JIk%3u,JJ5=gBase[b J3R=0;j<nRows;++j)nKeys
#define JnR &&W0U==8u&&t.s1==16u&&t.n1==8u&&r==128u&&t.s2 Jid&&
#define JnS b=0;b<W0U;b+=t.Jed)W1q n=W0U-b<t
#define JnT ev=gEv[b];if(JIk+1u<2u){Jjg[W3F,J2m<T>Jen
#define JnU 2,0,0,JHZ,W4g(W21=0;mi<2u W4K)Jc4(a2
#define JnV ];JIJ W1E((A/8+31)/32*32);Jat(JII,W0E,A)
#define JnW };if(mm J5v[as]W5u JaU(mm.a1[ao],JJS,acp);if
#define JnX +ff],64,128,128,512,512,JHZ,JHZ,qr,kr[kp])W1b mm
#define JnY 1,W2y+W0K,64,128,128,128,ff==0,ff==384,16384,ff
#define JnZ <W1j(gs[wo],gs[JcW+wo],f==0?JiI);}mm.H2GDrain(
#define Jo1 ,0,512,256,1,0};Jb7 tcp{1,128,64,0,512,128,1
#define Jo2 =0 W0x(W4H=0;j<4;++j){J5r){for(W4H=0;
#define Jo4 W02 JcV>(JaK)W10 JcV>(JaK
#define Jo5 W02 Z0q>(Jao)W10 Z0q>(Jao
#define Jo6 W02 Jb9>(JaY)W2D>(JaY
#define Jo7 )W0o();JdF(Jap,Jap,JaQ
#define Jo8 W02 Jbt>(JaY)W3C>(JaY
#define Jo9 ,Jbv{1,8})W0o();
#define Joa W02 Jaw>(JaY)W10 Jaw>(JaY
#define Job ,255u,255u,255u,255u,255u,255u,255u,255u
#define Joc W02 JcV>(Jao)W10 JcV>(Jao
#define Jod J7Y,JI7 SCORE_DB W0s,JI7 UNIQUE_FRAME W0s,JI7 LOCAL_COMPLETE W0s
#define JoF W2Z*p=Jbl<W0k*>(W5k
#define JoG W02 Z0c>(JaY)W3h>(JaY
#define JoH ,W1f GT,Jaj(64))W0o(
#define JoI T>&mm,Jaj Jdi,Jaj JdV)W44 Z25
#define JoJ (W0B>dst,W0B>src,W0B
#define JoK ,b2[bo],mp);W1W>(
#define JoL W02 Jce>(JaY)W10 Jce>(JaY
#define JoM W03 Jd3(){if(Jbp){W06
#define JoN W02 JcI>(Jao)W10 JcI>(Jao
#define JoO W02 Z0y>(Jao)W10 Z0y>(Jao
#define Jpf Jda::Jc1,JcL::Jca>(JK5);Jbd<JcR::
#define Jpg W0g<Jaj>(),0,2048);W0i
#define Jph JaR<JJ0 T>W11 SfagSparseHalfOwnerSupports(W2j&t,Jaj
#define Jpi W0l();Z1i(JIJ,JbR,Jd2,Jbr
#define Jpj ),cp)W0o();JcF();
#define Jpk W0h>();JiQ W0m>(),W3x JdR
#define Jpm )W0o();}Jcc(Jap,JaQ(
#define Jpn W44 JIY(){JaN<JdL>();}
#define Jpo W3I(W2I JIA*>(Jbl<W1Y*>
#define Jpp ,8192u)W0o();Add(sum16,sum16
#define Jpq W52{0,64,1,0,0,JHZ,0
#define Jpr JaR<JI7 TA,JI7 DIRECT_DQ W0s,W40>W03
#define Jps ;mp.JbQ W3N;mp.Jci W59(c
#define Jpt >();Jcv(JbR,JbX,JaE W3O<Jaj
#define Jpu ,0,0,W4g(Jaj ki=0;ki
#define Jpv W01 vc=Jbi()*2,vi=Jaz();
#define Jpw W02 Z0q>(JaK)W10 Z0q>(JaK
#define Jpx W5T(qr,kr,JHZ,JHZ)W4t
#define Jpy W0l();Jbj(JIJ,JII,JI2(JaO
#define Jpz JaZ(W4n*>(ring W3A(C6CountPacketOffset(JIh
#define JpA JIb JII=tb W5Q>();JII W1E
#define JpB Jd0(W3p,JI2>W0p||W3p,T
#define JpC ,1,0,0,JHZ,0};Jc4(b2[
#define JpD W01 JIe=Jaz()/2u,sub=Je9();
#define JoP =tb W2n<T>();JJ1.JaF(JJ9
#define JoQ (Jc9<T>&mm)W1q JIe=Jaz()
#define JoR W0l();JJH(JII,ix,W1G
#define JoS W08 Jc7=W1n(W0U)*t.
#define JoT W3I(W2I Z36*>(Jbl<W1Y*>
#define JoU ,Jap W3O<Jaj>(),0,8192);
#define JoV W4p,W0n low,W0n
#define JoW ;JaM Jf6=0;Jf3(refs,JcY,JIJ W3O
#define JoY W0l()W0b Jet=JbR W3O W2T
#define JoZ W4Q>(JaY)W1b W4Q
#define Jp0 *>(Jbl<W0k*>(JbN W3A())
#define Jp1 W02 Jb9>(Jbx)W2D>(Jbx
#define Jp2 Jf5(gs,dp,JIX,col,nr,JIv,JHZ)
#define Jp4 Jce>(JaY);Jbp W3N;Jcz=0;}}
#define Jp5 {1,128,256,0,512,128,1,0
#define Jp6 =Jbl<W0k*>(W5k());*
#define Jp7 W1t,8192);W07 JcI>(Jcf)
#define Jp8 ,JdW(1,1,8,1),Jf6);JaN
#define Jp9 JI0 Jan ap={0,4,1,0,3,0
#define Jpa {if(JbC())Jf1()W1b Jgg();
#define Jpd W02 Jck>(JaK)W10 Jck>(JaK
#define Jpe ,JaM JIX,W1e,JI7 Jdl W0s,JI7 Jct W0s
#define JpG W02 Jck>(JaY)W10 Jck>(JaY
#define JpH W02 Jck>(Jao)W10 Jck>(Jao
#define JpI >();Jbj(JIJ,JII,0.0f,W1H,
#define JpJ );Jcv(Jet,rank,Jap W3O<Jaj>(
#define JpK W0l();Jbj(JbH,JaI,0.0f,Jbr
#define JpL Exp(JIf,JIf,JIQ)W0o();
#define JpM ,1,1,8)W0o();Jeu
#define JpN W5i{1,128,64,0,64,128,1
#define JpO Z0R;}Jbd<W2G>()W4t
#define JpP W02 V_S>(JaK)W10 V_S>(JaK
#define Jqg W02 S_V>(JaK)W10 S_V>(JaK
#define Jqh PrepareCountMergeC7()W44 PrepareCountMergeC2Parts()W44 PrepareCountMergeHF3(
#define Jqi W08 ro=W1n(JIe)*Jd7*4+
#define Jqj !=JdK);W2a vec.WeightsTile(Jc6,t.Jbf
#define Jqk );JaM extracted=0;Jf3(JeS,JJ6,Jar(2
#define Jql ,272)W0o();JcX();Jdf
#define Jqn W1O W1g 2]={JHZ,JHZ W4f(Jaj
#define Jqo ,JaQ(2),2048)W0o();
#define Jqp W4g(W21=0;mi<4u W4K
#define Jqr JIm Jf1()W44 Jgg()W44 JeL
#define Jqs W0n JIB,W0n grad,W0n
#define Jqt {1,64,512,0,512,64,1,0
#define Jqu ),0,W2q(Jcl),W2q
#define Jqv (out W3A()W4u)JbT<T,JI2,Jax
#define Jqw ){W08 keyRow=W1n(b)W15+
#define Jqx ,0,4096,64,1,0}W0O(Je5[
#define Jqy for(W1e=0;col<W0D W5H)Mul
#define Jqz W1n(b)*W5B)W0y
#define JqA JaN<JdL>()W4t JgQ(){JeN
#define JqB for(W21=0;mi<2 W4K)Jc4(
#define JqC {1,128,512,0,512,128,1,0
#define JqD W0n qg,W3W,W0n
#define JqE )W0o();Mins(Jap,Jap,JaQ
#define JqF ,8192)W0o();H4Iota(Jap,JaQ
#define JpQ [448],tcp);W0W>(JbO)
#define JpR c7L1Ready&(W16 ns))&&(Z2L==Jbu||Z2L->
#define JpS row<W46 W4M 128u){W08 wo=JaM(row
#define JpT (W0n q,W0n go,W0a
#define JpU );JcX();JI0 Jbz cp{1,1,1,
#define JpV (Jaj JIW=0;JIW<JIu;++JIW)W1q
#define JpW ],fp);if(Jbq&&Jbm W3A()W4u
#define JpX ,64,0,64,256,1,0}W0O(
#define JpY W3j>(as);W4G]W3N
#define Jq0 W3j>(as);Z1j[as]W3N
#define Jq1 (W2j&t JiR>JIR,JaM JJ7,JI7 initializeMarkers);
#define Jq2 Jbd<W2G>();W3E,Jda::Jc1
#define Jq4 W0O(ix,JbB[JIh*t.Jbf+JJa],
#define Jq5 (Jbe<T>&mm){mm.Je1(0u);
#define Jq6 JfO){W1X>(Jao);JfO W0s
#define Jq7 W0o()W2m(Jdc,Jdc,W0E
#define Jq8 (Jaj JIr=0;JIr<2u;W2M)W1q
#define Jq9 JJ2>=4)Jav<2,Jc8>(W3B+1
#define Jqb JaZ(W2I JIA*>(ring W3A(C6CountPacketOffset(JIh
#define Jqc )W0o()W2v()W0O(JbN
#define Jqd (Jaj JIq=0;JIq<2;++JIq)for(Jaj
#define Jqe JI7 Jdt();W11 JaH(){Jb1 W0P!
#define Jqf )W2v();JJH(JbW,raw,W1G,
#define JqH ){Jaj nb=ns;W13(TA){nb=c7BCursor
#define JqI JcD W4c>();W4Z{Z25();}}
#define JqJ ,32u)W0o();JcX()W5r
#define JqL W03 Z2X(W27,W1e,Jaj nr,Jaj JIv
#define JqM W03 HrWeights()W44 JJx(Jc9<T,
#define JqN ,bMath[2]={JHZ,JHZ}W1O Jbp W0s
#define JqO );Jdb();if(Jek)tr.Jd4(tw+
#define JqQ (t.JbP&(W16 25)){JIb x=C5NzTrace
#define Jrh W3f])W0A>(5u W5D);
#define Jri 2),8192)W0o();Jcv(Jc0
#define Jrj ,JaM(128),W1U>(32),nzc)
#define Jrk Jgi){W1X>(Jbx);Jgi W0s
#define Jrl W03 Jh4(W0B>dst,W0B>src
#define Jrm W32,Jaj Jel,Jaj*JeX,Jaj*Jdn);JaP
#define Jrn JI7 ROPE,JI7 Z01,JI7 JdN,JI7 TINY W0s>JJ0 JJI
#define Jro ;fp.JJz=W2r.JbK=W2r.JaB
#define Jrs ]={JHZ,JHZ W4f(Jaj JIG=0;JIG
#define Jrt W03 Compute64(W09 J7Z&t,JaM JIX,W1e
#define Jru W03 Init(Z0F&p,JI7 Jd9 W0s,JI7 JaJ
#define Jrv ,W2H Jam>(Jbl<W0k*>(Jdd W3A
#define Jrw ,W1j(JbX,JII,JJl,JJA,JaQ(64)
#define Jrx *Q)W0b JIJ=fb W2n<Jar>()
#define Jry ,128u,1024u,row W0S,row+128u W4s,2048u
#define JrA W0l()W0b JcY=JJ6 W3O W2T
#define JrB JaM>tr;if(Jek)tr W3I(W2I JaM
#define JrC W3I(W4n*>(Jbl<W1Y*>
#define JrD W3N,W40>W03 Jdp(W09 J88);
#define JrE J1E JaM JJL,JcW,JJf,kelems;Jaj r,Jau;Jak
#define JrF W03 H3ResidentPair(W4p,W5a
#define JrG [as]){W0A>(3u+as)
#define JqR )W44 JdX(Jbe<T>&mm,W33
#define JqS J85)if J53*k*16],p);Jhf+mi
#define JqU using GPParams=typename SfagC3GpParams<NZ>::Type;GPParams acp;W13
#define JqV Jaj JeR=1 W44 Je1(Jaj g){JeR=
#define JqW [bs]){W0A>(5+bs)
#define JqX ;fp.JJz=n;fp.JbK=m;fp.JaB
#define JqY W02 JcG>(JaY)W10 JcG>(JaY
#define JqZ p.JaF(bb,262144);p J51 W0m>()
#define Jr0 W02 Jb9>(Jcf)W2D>(Jcf
#define Jr1 [as]){W0A>(3+as)
#define Jr4 2u)W0w,feat=(at%2u)W12;Jak
#define Jr5 mm.Jd3()W4t JeT(Jbe<T>
#define Jr7 W0x(Jaj f=0;f<4;++f){
#define Jr8 JaU(JbN[JIh*t.s2],JIu,t.s2
#define Jr9 W0l();Jcc(Jd2,JI2(W0V
#define Jra ],bg[W1n(k)W1t,nbp
#define Jrc JJH(Jh6,Jc0,W1G,8192);W0i
#define Jrd W0B>dst,W0B>JbE)W44
#define Jre W0R JIQ=JIv*nr W3v JIX=W0d
#define Jrg dqTmp W3I(W4m*>(user+priv+344064
#define JrK dqFullRead){W1X>(Jbx);dqFullRead W0s
#define JrL (JaH())Jb1 W2O(Jds())Jb1 128
#define JrM W2v()W4t JfA()W1q vc
#define JrN ;if(nb>=6u)nb-=6u;}C7StageWide128<TA>
#define JrO ;JI0 Jh5 mergeInfo(lengths,JHZ,3,1);JI0 Jhd
#define JrP (W4H=0;j<4;++j)W1q
#define JrQ ,JaQ(1),N)W0o();
#define JrR JaQ&J8V job=SpatialHybridJob(ownerPlan,JIe,JIW)W0R
#define Jsi if(JdE())Jb1 W0R vc=Jbi JtJ()
#define Jsj .H3ResidentPairFinish(Jd8);mm.Jd3()W4t
#define Jsn W3I(W4m*>(JJ5+48ull*1048576)
#define Jso W1W>(JbD)W1b W1W
#define Jsp JaY)W10 Z0U>(JaY);}W00
#define Jsq W02 Jbh>(JbM);W0A>(JbM
#define Jss W02 Z0Q>(JaK)W10 Z0Q>(JaK
#define Jst W02 JcI>(JaK)W10 JcI>(JaK
#define Jsu W1O Z0N[2]={JHZ,JHZ},Z0B[2
#define Jsv W02 Jcw>(Jbx)W10 Jcw>(Jbx
#define Jsw W02 JcI>(JaY)W10 JcI>(JaY
#define Jsx W02 Jcw>(JaK)W10 Jcw>(Jao
#define Jsy >();Jcv(JbR,JbX,Jap W3O<Jaj
#define Jsz Jbe<T>mm;mm.InitH3Score512(JJ1);JdS(mm
#define JsD W02 Jck>(Jcf)W10 Jck>(Jcf
#define JsE W02 JcV>(Jcf)W10 JcV>(Jcf
#define JsF W02 Z0y>(JaK)W10 Z0y>(JaK
#define JsG .JcT=W5h Z1B;W2u,JI2,Jax>(
#define JsH ;}if Je4{Jbe<T>mm;mm.JeF(JJ1
#define JrS ;JbB W3I(W4n*>(idxG));
#define JrT dp W3I(W4m*>(user+priv+32768
#define JrU {1,64,64,0,64,64,1,0
#define JrV W0n out,W0n Jbm,W3s,Jaj
#define JrW JaR<JI7 TA,Jaj FN=512,W40>W03
#define JrY ();Z0Y:W03 JIw(Jam JIB J7X&data)
#define JrZ ;}Z0Y:W03 JIw(Jam JIB J7X&data){
#define Js0 W0n Z0X,JcC W1O JK6 W0s,Je6 W0s
#define Js1 W02 Z0c>(JbO)W3h>(JbO
#define Js2 JIy Jfu();}}JcD W4c>();W4Z
#define Js3 W0l();JJH(Jap,JIS,W1F Jdo
#define Js4 ;JIf W3I(W4m*>(user+priv+
#define Js5 c6B32B1){W1X>(Jbx);c6B32B1 W0s
#define Js6 ,W3y,JIv)W0o();}
#define Js8 rp=fp;rp.JJz=64;rp.JaB=64;
#define Js9 Jj4>(TA?k:m),J67 J53 J86)J80 0
#define Jsa W0W>(JbO);JbA W3N;}}
#define Jsb ,JIf,JIX,col,nr,JIv,!Z01);}
#define Jsc (){Jb1 Jh7.JaR GetWithOffset<U>(16384u/Z18(
#define Jsd Jj0],nap J56>(3+as);Z0v[as J4S
#define Jse (Jbe<T>&mm)W1q JIe=Jaz()
#define Jsf ;if(JbA){W4Q>(JbO);
#define Jsg JJ2)W1q JIg=JJ2%4;Jav<2,Jc8
#define Jsh W0l();Mins(Jap,Jap,W0V
#define JsJ {1,128,448,0,512,128,1,0
#define JsK Jb0<W1y JII,W1e){Jaq Jaj N=2176;
#define JsL ;Jav<2,Jc8>(3);W13(!Jer
#define JsM W3I(W2I T*>(user+priv+212992)
#define JsP );if(balanceC4){vec.WeightsTileCached(Jc6,t.Jbf
#define JsQ ,JcL::Jca>(JJm);Jbd<W2G>(
#define JsR Jaq(W0P!W0F JdN&&!Jdq){Jb1 t Jkl
#define JsS JHZ;}if(Jbp){W06 Jce>(JaY
#define Jtj W3I(W2I T*>(JJ5+16777216));
#define Jtk 31)/32*32);JIu W1E(t J57()
#define Jtl W01 rt=(r+63)/64,kt=(t
#define Jtn ,0,JHZ,W4g(W21=0u;mi
#define Jto ;}W5K;mp.m=W4E.n=128
#define Jtp W02 V_S>(Jao)W10 V_S>(Jao
#define Jtq W02 S_V>(Jao)W10 S_V>(Jao
#define Jtr (gs[wo],gs[JcW+wo],k[JaM
#define Jts W3I(W2I T*>(user+priv+81920)
#define Jtt W3I(W2I T*>(user+priv+57344)
#define Jty (){{}W25;}W11 JeK(){
#define JtA W0l()W0b JaE=fb W2V
#define JtB W0W>(JaY)W0O(b1[65536u]
#define JtC ;W2E>(JIg)W3C
#define JtD ;W2E>(JIr)W3C
#define JtE ,W1G,8192);W2a W06 JcI
#define JtG =Jb7{1,W2q(m),W0e
#define JtH W5y Z17>JIF;JJ1.JaF(JIF,185472)
#define JtI W2z])W0A>(JIK?Jcf:JbM
#define JsT );W0W>(JbO)W0x(Jaj key
#define JsU KVPlane(at,1,dst+32768);Mark(w+6
#define JsV );}W5U(qr,kr,JHZ,JHZ);
#define JsW Jav<0,PIPE_S>(8);Jbg<2,JdH>
#define JsX {W13(W1w!W0F W55,JIA
#define JsY ){Jaq JaM advance=TA?JaM(32)W20:16384u
#define JsZ )W0o();Adds(JdC,JdC,JaQ
#define Jt0 )W0o();JdO(JIu,JaE,JaQ
#define Jt1 (out W3A()W4u)JbT<Out,JI2,Jax
#define Jt2 JaM(x.type)*4194304;W2u,JI2,Jax
#define Jt3 Jam JIB,Jam key,Jam JIa,Jam JiP&t){if
#define Jt5 W01 JIe=Jaz()/2u,sub=Je9(),
#define Jt6 W4t Init(Z0F&JJ1){JJ1.JaF(
#define Jt7 ()W1q JIe=Jaz()/2,sub=Je9
#define Jt8 JIb Je7=fb W2V>();Je7 W1E
#define Jt9 sub W2R W3v ro=JaM(JIe)*262144u+JaM
#define Jta )JcS W0R JId=tag/4 W0x(Jaj JJF=
#define Jtb JaI,W0V>(JIv))W0o(
#define Jtc JIb zero=f W3O<T>();Jat(zero
#define Jtf W08 at=JaM(JId)*2097152+JaM(key)*
#define Jti 1.0f,JIv)W0o();}C3DirectRows128 W4c
#define JtK JI0 Jb7 bcp{1,128,256,0,W2q
#define JtL ,1,0,16,16,1 W4f(W1e
#define JtM 127)W47)*(Z01?1:Jhk?(i J4F
#define JtN Jar(8),Jbv{1,8})W5r(JfV
#define JtO ;W0n Jbm=dkr;W0X Jbq=!val&&
#define JtQ W0x(Jaj d=0;d<W4X d+=64)
#define JtS (W0n dst,W0B>src,Jb0<T
#define JtT ,W0V>(JJ2+1u));W0C Jbg<
#define Juk W08 qi=JaM(JId)W0y,ki=JaM(JId
#define Jul )W47);Jb1 JbF()==256||JIU>=Jbi(
#define Jum )W0o();JcX()W5r(JfG
#define Juq Jbp W5u}Jkc JcC=b;JK6=pa;Je6
#define Jur ,32)W0o();Jbj(JIJ,JeS
#define Jus ]W0b JfD=JJ9 W2V>()[4096
#define Jut 0u);W2E>(JbO)W0x(Jaj
#define Juu for(Jaj i=0;i<2u;++i){if
#define Juv ,512,0,512,256,1,0}W0O(
#define Juw Jc4(a2[ao+mi*k*16],a1[ao
#define Jux ,W1F Jdo,JaM(64),Jar(64),cpCast
#define Juy (W0n qg,W3W,JkT JaB)
#define Juz ;JdX(mm,(i/8u)*136u+i%8u
#define JuA )JcS W0R w=128u+jobs*8u;Jf2(w
#define JuB W32){Jb1 W35 8?7u:6u;}W00
#define JuC JId=JbZ/4 W3v qi=JaM(JbZ)W14
#define JuD ();JJ1.Reset();Jei();Jgb()
#define JuG W06 Z0U>(JIr);Jd8 W39]W0s
#define JuH W4p,W5a,W0n
#define JuI ),W2q(m),W2q(256u
#define JuJ ,JI7&p){W13(Jeb==4)JfK<T
#define JtU if(h2G4L1 W3f]){W1X>(JIg
#define JtV fix){W06 Z0U>(JbO);fix W0s
#define JtW Jbb W1L+ki W12],bp);Jas(b2[
#define JtX )W0o();Adds(Jap,Jap,JaQ
#define JtY for(Jaj JIh=Jaz();JIh<W0D JIh+=40
#define JtZ W05 Jbh>(as?Jao:JbD);abusy[as
#define Ju1 JIb Jfy=qr;{}W5U(qr,Jfy,JHZ
#define Ju2 W3v qi=JaM(JJO%4u)W0w W0b JJ8
#define Ju3 *JIW,JbZ=JJ7/8 W0R JIk=JcK W5x
#define Ju4 ,0u};Jhd W0m>lists(Z1m,Z1m[1024u]
#define Ju5 )W0o();W13(Jeb==4)
#define Ju6 mm){W13(Jeb==4){JfK<T>::
#define Ju7 W0x(Jaj h=0;h<2u;++h){
#define Ju8 .k=k;mm.JbQ=JI6;mm.Jci W0s
#define Ju9 128,k=128,JIq=16384,JJv=32768 W0R
#define Jua (W0n gs,W0n ps,W0a
#define Jub W1G,8192)W2v();Mul(JbW,JbW
#define Juc (actual,Jdc,actual,Jdc);MrgSort W0m>(JJ6
#define Jud JIb JbX=W5S>();JbX W1E
#define Jue ],J4T::Z0U>(JaY);Jbp W3N;Jcz=
#define Jug ],cc[cs*W0K,fp);++fixes;}
#define Juh );Jdx(tw+1);Jav<2,Jc8>(
#define Jui W2t Jcx(){if(t.s2>W5P r
#define Juj ,0.0f,64-JIv)W0o();Jbj
#define JuL W08 wo=JaM(row)W15 W0z;W5U
#define JuO W1U>(nr/8u),W1U>(nr
#define JuP W15 W0z],raw,store);Jbg<2,JeU
#define JuQ )W0o();
#define JuR W5Q>()[
#define JuS W3I(W4n*>
#define JuT );W0W>(
#define JuU ){W0A>(
#define Jvl W2V>()[
#define Jvm W3I(W2I T*>
#define Jvn =W5R>(
#define Jvo W5S>()
#define Jvp W3O<Jaj>(),
#define Jvq W3O W2T>();
#define Jvr JJ9 W5Q>()
#define Jvs ){W1X>(
#define Jvt =Jh7.JaR GetWithOffset W0m>(
#define Jvu [2]={JHZ,JHZ}
#define Jvv W5Q>();
#define Jvw for(W4H=0;j<
#define Jvx W3I(W4m*>
#define Jvy JaR<JJ0 S>Z0z W03
#define JvA W05 Jb9>(JbO);
#define JvB W2n<Jar>()[
#define JvC W2n<T>();
#define JvD );Jbg<2,JeU>(
#define JvE ;mp.Jci W59(
#define JvF if(t.JbP&(W16
#define JvG ){W06 Z0U>(
#define JvH );Jav<2,Jc8>(
#define JvI JJ9 W2V>()
#define JvJ W3O<Jb5>(),
#define JvK ])W0A>(
#define JuV ){W13(Jeb==4)
#define JuW Jbd<W2G>();
#define JuX W2V>();
#define JuY );W07 JcI>(
#define JuZ );Jbg<2,JdH>(
#define Jv0 ){Jav<2,Jc8>(
#define Jv1 for(W1e=0;col<
#define Jv2 JbT<T,JI2,Jax>(
#define Jv3 ],JaM(64),Jar(
#define Jv4 W1q vi=Jaz();
#define Jv5 );JaN<JdL>();
#define Jv6 W32,Jaj JIW);W00
#define Jv7 )W1q JIe=Jaz()
#define Jv8 W05 Jb9>(JaY);
#define Jv9 W05 Jb9>(Jbx);
#define Jva (Jbe<T>&mm,Jaj
#define Jvb ){W3j>(
#define Jvc =W1n(b)*
#define Jvd W3O<T>();
#define Jve ;Jat(JJl,Jaj(0)
#define Jvf =W5b>(ix.Jco
#define Jvg W3O<W1y();
#define Jvh (JcZ<T>&mm,Jaj
#define Jvi )W10 Jaw>(
#define Jvj ,fp);W07 Z0U
#define Jvk W05 Jaw>(JIg);
#define JvM );W1W>(
#define JvN JaR<JI7 TA,W40>JaP
#define JvP );W3j>(
#define JvQ }JaR<JI7 MULT>W03
#define JvR W05 Jbh>(JbD);
#define JvS ;W2u,JI2,Jax>
#define JvT W3T>(),
#define JvU JJ0 S,JJ0 MM>Z0z W00
#define JvV W05 Z0U>(JaY);
#define Jwm )W3C>(
#define Jwn )W10 JcI>(
#define Jwo (Jbe<T>&mm);
#define Jwp W05 Jb9>(JbM);
#define Jwq ){W4Q>(
#define Jwr ){W1W>(
#define Jws ;}Jav<2,Jc8>(
#define Jwt (W1n(b)*
#define Jwu JaR<JJ0 U>W00 Jb0<
#define Jwv );JcD W4c>();
#define Jww >(JbD)W10
#define Jwx ;}W0W>(
#define Jwy ,W1F Jdo,8192)W2v
#define Jwz for(Jaj JIr=0;JIr<
#define JwA Jaj Jdi,Jaj JdV);W00
#define JwB .Z2M(Z2j::Z3l);Jf2
#define JwC W2B>();}W00
#define JwD W2v();}}W00
#define JwE )W2v();}W00
#define JwF ;}W0l();
#define JwG .Z2M(Z1o::Z2j::Z4n)
#define JwH )W10 Jck>(
#define JwI (W4p,W09
#define JwJ W1W>(3+
#define JwK ])W06 Z0U>(
#define JwL );W07 Z0y>(
#define JvW <typename T,JI7 Z04,JI7 Z07
#define JvX for(Jaj JIg=0;JIg<
#define JvY W2n<T>()[
#define JvZ JIG)W1q JIg=JIG%
#define Jw0 W3A())-W2H JaM
#define Jw1 W0B>dst,W0B
#define Jw2 );W3Z;fp.Z0E=
#define Jw3 W0x(Jaj JIk=0;JIk
#define Jw4 for(Jaj Jbb=0;Jbb<
#define Jw6 W05 Z0y>(JaY);
#define Jw7 for(W21=0;mi<
#define Jw8 W3O<JdD>();
#define Jw9 JI7 JfP W0s,JI7 Z26 W0s
#define Jwa ()W44 JdB()
#define Jwb W2t SfagC4ActiveKvPackets(Jaj u,Jaj
#define Jwc if Je4{Jav<2,Jc8>
#define Jwd W05 Z0U>(JbO);
#define Jwe W1n(W0U)
#define Jwf Jav<2,Jc8>(W1P
#define Jwg =Jh7.JaR GetWithOffset W2T>(
#define Jwh W05 Jb9>(JIg);
#define Jwi ,1,0}W0O(a1
#define Jwj for(Jaj JIA=0;JIA<
#define Jwk (Jc9<T>&mm,Jaj
#define Jwl )W0R JIe=Jaz()
#define JwN for(Jaj JIW=0;JIW<
#define JwO ,JHZ,W4g(Jaj
#define JwP ,128,1,0}W0O
#define JwQ W4Q>(as)
#define JwR ;fp.JcT=W5h Je8;
#define JwS ;JbT<JIA,JI2,Jax>
#define JwT );Jb1 x W4t
#define JwU );W3E,Jda::Jc1
#define JwV );Jh5 info(lengths,JHZ,
#define JwW ])W1X>(
#define Jxn ;}W5K;mp.m=
#define Jxo for(Jaj g=0;g<
#define Jxp );}JcD W4c>()
#define Jxq JcD W4c>();W4Z
#define Jxr >(Jao)W10
#define Jxs for(Jaj i=0;i<
#define Jxt [JIX],32u)W0O(
#define Jxu Jd0(Z18(T)==2,
#define Jxv +mi W12],ap);
#define Jxw ,W1G,8192);JJH
#define Jxx Jcu())W4t
#define Jxy );Jbg<0,JeU>(
#define JxA W2n<Jaj>()[
#define JxB }W2e Jf4(){
#define JxC ]W5u Jan bp{0
#define JxD for(Jaj JIK=0;JIK<
#define JxE [W1n(JIh)*
#define JxF W3O<Jaj>();
#define JxG ();JJ1.Reset();
#define JxI for(Jaj f=0;f<
#define JxJ JaU(JJ8,JbB[JaM(JIh
#define JxK )W2v()W0x(Jaj
#define JxL for(Jaj ki=0;ki<
#define JxM ,Jda::JgC,JcL::Jca>
#define JwX );W07 Jck>(
#define JwY W1t,8192)W0O(
#define JwZ ,JaM(128),Jar(8
#define Jx0 W03 Z30();W00
#define Jx2 W0S)W4Q>
#define Jx3 W4K)Jc4(a2[ao
#define Jx4 ),Jb5(1),Jb5(
#define Jx5 ){JaU(raw,go[JI6
#define Jx6 .Jd3()W4t
#define Jx7 W3O<::W1y()
#define Jx8 for(Jaj sub=0;sub<
#define Jx9 JJ1.Reset();mm.Init
#define Jxa for(Jaj JIq=0;JIq<
#define Jxb W00 Jb0<T>JeI()
#define Jxc ;Jcc(Jap,JaQ(0)
#define Jxd W3O<W1y(),
#define Jxe ,W2q(n),
#define Jxf ,W0c T>(0),
#define Jxg Jav<2,Jc8>(0)
#define Jxh +1));hist.Jd4(
#define Jxi W08 ro=JaM(JIe)*
#define Jxj ,JIX,col,nr,JIv,
#define Jxk );}mm.Jd3();
#define Jxm +W45,Jea!=qo,nr
#define JxP ,W5I,W09
#define JxQ ++JJ2)W1q JIg=JJ2
#define JxS (W32,Jaj JIW){if
#define JxT W0x(Jaj JJi=0;JJi
#define JxU W0n q,W0a
#define JxV Reset();W5y Z17
#define JxW W3E,Jpf ALL>()
#define JxX );JcD<W1j();
#define Jyo W4t JeF(Z0F&
#define Jyp JaR<JI7 TA,W40,JI7
#define Jyq W05 Z0y>(Jbx);
#define Jyr W3J JIf,W0f
#define Jys )W0O(JIR[JI6],
#define Jyt W4t Jgb(){
#define Jyu [JaM(row)W1Z+feat
#define Jyv )W10 JcV>(
#define Jyw )W10 Jcw>(
#define Jyx );W13(Jeb!=4)
#define Jyy ;JaR<JI7 MULT>W03
#define Jyz T,JI7 Z26 W0s>JJ0 Jc9
#define JyA ,0,0,JHZ,0}
#define JyB ,W2q(JIA),
#define JyC ;}}W1W>
#define JyD );W2E>(
#define JyE )W3h>(
#define JyF Z0Y:W03 Init(Z0F&
#define JyG );W07 Jcw>(
#define JyH W0x(Jaj W4h 0;row
#define JyI );W07 JcV>(
#define JyJ Jaj c=Jaz();W4Z
#define JyK Jbe<T>&mm,W33
#define JyM W0x(Jaj q=0;q
#define JyN for(Jaj h=0;h<
#define JxY Jo9 JcF()W4t
#define JxZ W3A()),W2H __ubuf__
#define Jy0 fp.JbK=64;fp.JaB
#define Jy1 );W0i Jpt>()
#define Jy2 ){W07 Z0q>(
#define Jy3 Adds(JaE,rank,W0V
#define Jy4 W13(W4w
#define Jy5 W5y A1>ab J81>
#define Jy6 );Mul(JbW,JbW,JcQ
#define Jy7 );W5K;mp.m=
#define Jy8 +W1x],ap);
#define Jy9 W0B>src,Jaj JIv,
#define Jyb JbN.J4X*4;wq.JiV
#define Jyc W3O W2T>()[
#define Jyd {0,8,W2q
#define Jye ;JcX()W5r(JIs,
#define Jyf =fb W2n<JdD>(
#define Jyg );Jat(rank,JaQ(-
#define Jyh Jb7 acp{1,128,128,
#define Jyi for(Jaj JIg=0u;JIg<
#define Jyj [W1n(k)*
#define Jyl W55,W1z)
#define Jym JJH(JII,JJ8,W1G
#define Jyn 128u];W2Z*p J7j
#define JyQ user+metaBase+W1n(
#define JyR JaR<JJ0 T,JI7 RESIDENT>JJ0
#define JyS ={JHZ,JHZ,JHZ,JHZ
#define JyT JJN[i]W5u}
#define JyU ,a2,b2;W0B>
#define JyV mp.k=64;mp.JbQ
#define JyW JbU cp{16,4,0,
#define JyX W3T>();
#define JyY for(Jaj i=0u;i<
#define Jzp fp.JJz=256;fp.JbK
#define Jzq <TA>(JJS,bg,JcC
#define Jzs )ScoreStep(JHZ);Flush()
#define Jzt Jei();Jdx(7);
#define Jzv ,1,0}W0O(mm
#define Jzw ;fp.JcT=W5h Z1B;
#define Jzx W1n(JId)W15
#define Jzz W03 H3ResidentPairFinish(Jaj&Jd8)
#define JzA W2j&t JiR>JIR,JaM
#define JzB W0n key,W0a
#define JzC W05 Jce>(JaY);
#define JzD m),1,0};Jb7
#define JzE JaM Jd6=(JaM(JId)*
#define JzF ;JaW<Jo4);}W00
#define JzG ,W1G,8192u)W2v
#define JzH JIw(Jam JIB J7X&data){
#define JzI p.JcT=W5h Je8;p
#define JzJ for(Jaj s=0;s<
#define JzK )Jav<2,Jc8>((
#define JzM for(Jaj b=0;b<
#define JzN W1E(t.s2)W0b
#define JzO W03 JfA();W00
#define JyZ W01 JJa=JJ2>4?JJ2
#define Jz0 0u&&JbS>0u)W1q
#define Jz1 ]W5u W2e
#define Jz2 ],JaM(16),Jar(
#define Jz3 );if(C3CompactWb())C3CompactRows
#define Jz4 ,Jaj Z0T,Jaj Jfw,JaM Z3z
#define Jz5 JaR<JJ0 T,JI7 ROPE,JI7
#define Jz6 ],ap);Jas(a2[
#define Jz7 W4d+=vc){W08 JIh
#define Jz8 LATE3 W0s,JI7 CACHE_A W0s>
#define Jz9 (!CACHE_A||W2f)JaU
#define Jza ;}Jbg<2,JdH>(
#define Jzc Jat(JIu,0,t.s2
#define Jzd JcC=b;JK6=pa;Je6
#define Jze ,8192)W2v();}
#define Jzf ){W06 Jcw>(
#define Jzg W3A()==Jbu)Jb1;
#define Jzh W2z]W3N;}W00
#define Jzi (Jbe<T>&mm){
#define Jzj ){Jav<0,PIPE_S>(
#define Jzk +=Jbi())H3GradTask(mm,
#define Jzl JiN,JI7 HF3_GEOM W0s>JJ0 Jhc
#define Jzm W01 vc=Jg2()*2
#define Jzn 1,0}W0O(b1,
#define Jzo ;Jh5 info(n,JHZ,15
#define JzQ (dst[col],dst[col
#define JzR );Jat(JJW,Jaj(0
#define JzS ,W2q(nr/16u
#define JzT +JI6*nr],JIQ);
#define JzU mm.Je1(0u);mm.
#define JzV =JJ9 W2n<JdD>(
#define JzW W00 W3J C5NzRaw(Jaj
#define JzY ,JI2(W0V>(u
#define JzZ JeH<T>JdT;JdT.JeF
#define JAq <JJ0 T,JI7 WRITER_GATE>JJ0 Jfc
#define JAr if(out W3A()W4u
#define JAs ){W4P>(
#define JAt (W0V>(JJi))
#define JAu >(JIK?JbO:JaY);
#define JAv [countOffset],ccp)W0O(
#define JAw if(dqr W3A()W4u
#define JAx [W1n(col)*
#define JAy );SpatialAfterDQQueryPrefix(JIi,JfY);
#define JAz >(JIf,JIf,JfG);
#define JAA ;W00 W09 JaM>C5NzTrace(
#define JAB W3I(W2I::JIA Jp0
#define JAF )W4A:W1n(
#define JAG W4t
#define JAH )W44
#define JAI );}W00
#define JAJ }W2e
#define JAK ;}W2t
#define JAL ();W00
#define JAM ,W1G,
#define JAN W0V>(
#define JAO W2q(
#define JAP W1n(
#define JA0 ;}}W00
#define JA1 ,W0n
#define JA2 JaR<JJ0 T>
#define JA3 W5b>(
#define JA4 ;JJ1.JaF(
#define JA5 ,W1F Jdo,
#define JA6 )W2v()
#define JA7 ,W3y,
#define JA8 );W11
#define JA9 );W2t
#define JAa Jav<2,Jc8
#define JAb (W0n
#define JAc )W0x(Jaj
#define JAd ;}W11
#define JAe =W4T>
#define JAf ;W0n
#define JAg ;W07
#define JAh ,W3J
#define JAi ()W0O(
#define JAj ;W1d>
#define JAk ]W5u
#define JAl W5y Z17
#define JAm Z0Y:W03
#define JAn }W07
#define JAo }}}W00
#define JAp )W06
#define JAR JaM(JIh)*
#define JAS W10
#define JAT Jb8()W4u
#define JAU W2C>
#define JAV ){W13(
#define JAW =Jaz();
#define JAX W4Q
#define JAY ;W5y
#define JAZ W3A())
#define JB0 W0l(
#define JBr W3I((Jb3
#define JBs =JaL W2T>
#define JBt ));W09
#define JBu ;Jdx(tw+
#define JBw JfK<T>::
#define JBx ()W1q
#define JBy JaM(JId)*
#define JBz )JcS W0R
#define JBA JaN<JdL>(
#define JBB JaZ(W2I
#define JBC ;}W4Z{
#define JBD );W0i
#define JBE JaM(row)*
#define JBF }W00 W09
#define JBG ))W0R
#define JBH Z2M(Z2j::Z4n
#define JBI ,W0B>
#define JBJ ;}for(Jaj
#define JBK JcD W4c>(
#define JBL W13(JfP)
#define JBM ,0}W0O
#define JBN W0O(a1[
#define JBO .JcT=W5h
#define JBP {W06
#define JBQ for(Jaj JIh=
#define JB1 (W0B>
#define JB2 );W4Z{
#define JB3 W07 Jcw
#define JB4 Jbg<2,JeU
#define JB5 Jay ap{0,
#define JB6 ,W1e,Jaj
#define JB7 W1E(1024)
#define JB8 ;W3J
#define JBa JaM(b)*
#define JBb T>W03
#define JBc );W2a
#define JBd W4K)Jas
#define JBe Jbg<2,JdH
#define JBf ,JaQ(0)
#define JBg W2H JaM>(
#define JBh );JJZ()
#define JBi );W13(
#define JBj ,W1U>
#define JBk W32=Jaz(
#define JBl W13(Jeb==
#define JBm >Z0z W03
#define JBn Jh7.JaR GetWithOffset<
#define JBo W4Z{JdJ(
#define JBp ;}W13(
#define JBq W0O(b1[
#define JBS W0C W1q
#define JBT ));Jf2(
#define JBU (W1h)
#define JBV .Jco(0)
#define JBW ;Jhd W0m>
#define JBX )Jb1 W0R
#define JBY );Jb1;}
#define JBZ ,cp)W0O
#define JC0 W3O<Jar
#define JC1 (JIf,JIf,
#define JCs W32){Jb1
#define JCt );Jdb()
#define JCu ;++ki)Jas
#define JCv )W0b Jap=
#define JCw Jb7 cp{1,
#define JCx ()W0R
#define JCy JaM(col)*
#define JCz ,JaM JIX,Jaj
#define JCA JaW<Jo5);
#define JCB W4Z{if(
#define JCC W1d>JJe
#define JCD Jav<0,PIPE_S
#define JCE Jan bp{0,
#define JCF Jan ap{0,
#define JCG W0O(l1[
#define JCH :0u W0R
#define JCI }W00 Jb0<
#define JCJ JHZ>(Jc6,
#define JCK W1E(JaD)
#define JCL {W1d>
#define JCM W0R JI6=
#define JCN (W1d>
#define JCO .JaF(ab,
#define JCP JaM(JIe)*
#define JCQ }W25;}
#define JCR W1E(4096)
#define JC2 W1E(2048)
#define JC3 W1o W2T>
#define JC4 JaR JvW>JJ0 JIV
#define JC5 JaU(ix,JbB
#define JC6 (t.JbP&
#define JC7 8192u)W0R
#define JC8 ;Jap W1E(
#define JC9 W0s>JJ0 Jhc
#define JCa W0s>W00
#define JCb JHZ W44
#define JCc ,Jcu())
#define JCd );}JaW<
#define JCe W4f(Jaj
#define JCf ;JIJ W1E(
#define JCg ,JaQ(1)
#define JCh W1E(8192)
#define JCi (t.Jge&
#define JCj JaW<Jo4);
#define JCk {W07
#define JCl (Jaz())
#define JCm =Jaz(),
#define JCn ;JcX();
#define JCo W0O(raw,
#define JCp W1Z+feat]
#define JCq )W1q bit
#define JCr (t.JbP>>
#define JCT (W32,Jaj
#define JCU JaU(Je5[JIs
#define JCV >(JaY);
#define JCW ,JHZ,JHZ)
#define JCX ,Jap Jvp 0,
#define JCY >(Jcf);
#define JCZ Jcl,Z20,JdP
#define JD0 ]W3N;}
#define JD1 ,W3x JdR,
#define JD2 ;mp.JbQ=
#define JDt )W1q ns
#define JDu Jat(ones,1.0f
#define JDv for(Jaj bit=
#define JDw j)W1q
#define JDx Jcq();}
#define JDy W03 JcM(
#define JDz JaW<Jo8);
#define JDA JaW<Jo6);
#define JDB ){for(Jaj
#define JDC ;fp.JaB=
#define JDD ;W5j;}
#define JDE )W1q JIg
#define JDF )W1q JJO
#define JDG W1E(16384)
#define JDH ];W2Z
#define JDI W1E(128)
#define JDJ ();Jdx(
#define JDK ;Jdx(w+
#define JDM W20;if(
#define JDO if(Jaz()
#define JDP ){JIb JK5=
#define JDQ JaM(JI6)*
#define JDR ]W3N;if
#define JDS ,Jaj JIL,Jaj
#define JD3 ,W2j&t
#define JD4 Jan p{0,
#define JD5 W01 W4x(
#define JD6 JIW)W1q
#define JD7 JaR<JJ0 T,
#define JD8 (JII,JII,
#define JD9 JI0 Jbz cp{1
#define JDa )W0b JIJ=
#define JDc .JaB=W4X
#define JDd (Jaz()-
#define JDe );if Je4{
#define JDf =hist.Jco(
#define JDg JcW+wo],
#define JDh Jb5 lengths[4]
#define JDi JuQ JcX();
#define JDj }}if Je4{
#define JDk if(Jau>1
#define JDl ;}if Je4{
#define JDm mi W1k],
#define JDn ;W5c
#define JDq H4CellL1Retire();Jb7
#define JDr Jbg<0,JeU
#define JDs .JfN W1B
#define JDU W2T>()
#define JDV Jdq&&!W1v
#define JDW );trace.Jd4
#define JDX W1E(A)
#define JDY W13(TA)
#define JDZ JaM(Z0f)*
#define JE0 Jbg<0,JdH
#define JE1 2u)W1q
#define JE3 ,JI2,Jax>
#define JEu ,W1f LT,
#define JEv >(Jao);
#define JEw .Jci W1B
#define JEx ,Jaj at,Jaj
#define JEy ;p.JaF(
#define JEz Jc4(b2[bo
#define JEA 0 W0x(Jaj
#define JEB .Jd3();
#define JEC Jvm(user+priv
#define JED ;Jb0<T>
#define JEE JJi)W1q
#define JEF ,128,128,
#define JEG W09::W1y
#define JEH ;}JaR<JI7
#define JEI JuT Jbx);}
#define JEJ ,0.0f,W1f
#define JEK .Jbo=0;
#define JEL ,t.JJq,
#define JEM =JII.Jco(
#define JEN ;JI0 JbU cp{
#define JEO {JJ1.JaF(
#define JEQ ;if(JIG==
#define JER ,8192)W0b
#define JES ,JdD(16384)
#define JET 0 W44
#define JE4 Jbj(JIJ,JII
#define JE5 JuQ Jbj(mask,
#define JE6 W2n<Z36
#define JE7 W0O(b1,
#define JE8 )Jc4(a2[
#define JE9 W0R as=
#define JEa >(JbO);
#define JEb for(JaM JI6=
#define JEc W3v wo=
#define JEd ,Jaj Jbb){
#define JEe );JaN Jpk,
#define JEf )W1q as
#define JEg )W25;JI0
#define JEh }JaR<JvU JIm
#define JEi ;fp.JJz=
#define JEj W1B if(
#define JEk ]W0b JbH=
#define JEl (JaE,rank,
#define JEm )W0b Jd2=
#define JEn ;Jf2(tw,
#define JEo JaR JvW>JJ0 JIO
#define JEp ;fp.JbK=
#define JEq JaW<Joa);
#define JEr ;W09 JaM>
#define JEs Jam JIB J7X&data
#define JEt W3v Jd6=
#define JEV ,Jaj JIg,Jaj
#define JEW col,JI7 Z3O)
#define JEY JaU(mm.b1
#define JEZ ;Jat(JbX[
#define JF0 Jbp W5u
#define JF1 W2n<JIA
#define JF2 JaL<JdD>(
#define JF3 );MrgSort W0m
#define JF4 W08 Je2=JIX
#define JFv W1E(32768)
#define JFw Jj2 ki)Jas(
#define JFx ;mm.JaR C2PairFullWide128
#define JFy JJ1.Reset()
#define JFz ))W0O(
#define JFA mm.Init(JJ1
#define JFB if(JIq W0S
#define JFE W5i{1,
#define JFF mi*W4e,
#define JFG ?Jcf:JbM)
#define JFH (Jc9<T>
#define JFI A W0m>(
#define JFL JuQ Jbj(JbH,
#define JFM typename T>JJ0 SfagR1Owner
#define JFN Jaj word,JaM JIa
#define JFO JfX(JfY);
#define JFP W03
#define JFQ W0R
#define JFR W1q
#define JFT W3v
#define JFU W0O(
#define JFV for(Jaj
#define JFW ;W00
#define JF5 W1E(
#define JF6 ;Jat(
#define JF7 [JaM(
#define JF8 .Jd4(
#define JF9 ;JaW<
#define JFa W13(
#define JFb ,JaQ(
#define JFc ){JI0
#define JFd ){Jb1
#define JFe Jaz()
#define JFf .Jco(
#define JFg ;Jcc(
#define JFh }W01
#define JFi W11
#define JFj ;W0X
#define JFk {JaU(
#define JFl W1B
#define JFn )W0b
#define JFo t.Jbf
#define JFp Z0z Jaq Jaj
#define JFq Jbi()
#define JFr );JI0
#define JFs W2t
#define JFt W4Z{
#define JFu ;Jf2(
#define JFY ,Jaj(
#define JFZ )JaU(
#define JG0 JuQ Muls(
#define JG1 +JaM(
#define JG2 JuQ JbG(
#define JG3 =JaM(
#define JG4 ;Jd0(
#define JG5 W20]
#define JG6 ){Jb7
#define JG7 ,Jar(
#define JGy (W09
#define JGz );Jaj
#define JGA ){Jaj
#define JGB JaU(a1
#define JGC =JJ9 JuR
#define JGD JaR<JI7
#define JGE (Jaj id
#define JGF ;Jdx(
#define JGG mm.JaR
#define JGH Jbe<T
#define JGI ,JaD JuQ
#define JGJ Jeb==4
#define JGK JuQ Adds(
#define JGL W1d
#define JGM W0f
#define JGN (Jaj JIh
#define JGO JuQ JdF(
#define JGP JaR<JJ0
#define JGQ Jf4()
#define JGR ;JI0 JbU
#define JGS W25;
#define JGT ,1024 JuQ
#define JGU JbU cp{
#define JGV JuQ Jat(
#define JGW W0a
#define JGX JHZ,JHZ
#define JG8 )JcS;
#define JG9 ){JaW
#define JGa JaY);
#define JGb JuQ Jcv(
#define JGc W2T>
#define JGd t.Jge
#define JGe (Jaj JIg
#define JGf (Jaj JIi
#define JGg (W32
#define JGh >JJ0 JIT
#define JGi JuQ JJH(
#define JGj Jb0<T
#define JGk JbO);
#define JGl *W0K
#define JGm ,Jaj*
#define JGn ,Jaj Jbb
#define JGo ,Jaj n
#define JGp JIy if(
#define JGq if Je4{
#define JGr (JcU+
#define JGs W2j&
#define JGt Jay p{
#define JGu W0s,
#define JGv ,Jb5(
#define JGw ;Jaq Jaj
#define JGx JA2 JJ0 JcZ
#define JH0 ]W0b
#define JH1 ,JdD(
#define JH2 JaK);
#define JH3 JbC()
#define JH4 =JIF Jvl
#define JH5 ;JA2 JJ0
#define JH6 ;Z27(
#define JH7 Jbb],
#define JH8 512u],
#define JHz W3N;
#define JHA ){JIb
#define JHB );Jb1
#define JHC for(JaM
#define JHD ,JaQ JIa
#define JHE ;W09
#define JHF =fb JuR
#define JHG JIy JaU(
#define JHH W0m>
#define JHI 32768u],
#define JHJ Jcf);
#define JHK ,4096 JuQ
#define JHL ;JJH(
#define JHM JcF()
#define JHN }JaU(
#define JHO ,Jaj&
#define JHP JuQ Jcc(
#define JHQ .unitFlag=
#define JHR Jas(b2
#define JHS };Jb7
#define JHT W2n
#define JHU )Z37=
#define JHV {Jaq Jaj
#define JHW }Jdx(
#define JHX (Jam Jc6
#define JHY W3N JAG
#define JH9 ){JaS
#define JHb using Z1u::
#define JHc ],Je5
#define JHd W2y
#define JHe {Jat(
#define JHf C9ReadyTrace()
#define JHg Jec<scoreEnd
#define JHh >JJ0 Jhc
#define JHi Jdw[4194304ull
#define JHj 2048u],
#define JHk <Jaj>
#define JHl );Jbj
#define JHm (JaM(
#define JHn ){Jd0
#define JHo W05
#define JHp Jbb W0S
#define JHq W0g JDU
#define JHr Z0z Jaq JI7
#define JHs W1p+
#define JHt ;Z0H(
#define JHu )Jat(
#define JHv Jbm,Jbq
#define JHw (JIG==
#define JHx .JaB=
#define JHy Jbl<Jb3
#define J9b JaR<typename
#define J9c Z0H(JdY
#define J9d Jcg W3f
#define J9e >JJ0 SfagSmallrowWholeCount
#define J9f CopyCarryQuery(nextQg
#define J9g ,Jaj
#define J9h (Jaj
#define J9i ;Jaj
#define J9j W01
#define J9k JA2 JJ0
#define J9L W08
#define J9M ,JaM
#define J9N Jaj(
#define J9O JFc JaM
#define J9P Jeu(
#define J9Q JI0 Jb7
#define J9R ;JaM
#define J9S Jgb(
#define J9T ;Jb7
#define J9U JFl Jaj
#define J9V Jdy(
#define J9W Jas(
#define J9X JaM(
#define J9Y ,Jam
#define J9Z Jcc(
#define Ja0 JuQ JIb
#define Ja1 )Jb1
#define Ja2 JbU{
#define Ja3 Jc4(
#define Ja4 W00
#define Ja5 )JaS
#define Ja6 JgP(
#define Ja7 JgQ(
#define Ja8 W20
#define Ja9 JaI,
#define Jaa W4Z
#define J9l Jaq JaM
#define J9m JI0 JaQ
#define J9n Jdx(
#define J9o queryReleased[
#define J9p JaA<
#define J9q ,Jbb
#define J9r Jay ap
#define J9s JA6 JFU
#define J9t ,JaE
#define J9u JI0 Jb5
#define J9v Jc9<
#define J9x Jd3(
#define J9y JaU(
#define J9z ;JaG
#define J9A ,JaD
#define J9B ;JbU
#define J9C (JcZ
#define J9D JFr JdD
#define J9E Jea=
#define J9F .JaF
#define J9G JGu JI7
#define J9H JI7 Jdt
#define J9I W2H
#define J9K JAI Jaj
#define Jac SpatialGroupBalancedTagCount(
#define Jad Jan bp
#define Jae ;Jb1
#define Jaf C9InitEncodedCountConstants(
#define Jag JeD(
#define Jaj JK7
#define Jak JJM
#define Jal Z02
#define JaM JJT
#define JaN Z0w
#define JaO Z0x
#define JaP Z0g
#define JaQ Z09
#define JaR Z0e
#define JaS Z0h
#define JaT Z0l
#define JaU Z0J
#define JaV Z0A
#define JaW Z0r
#define JaX JIt
#define JaY Z1b
#define JaZ JJb
#define Jb0 Z3A
#define Jb1 JJd
#define Jb2 Z3B
#define Jb3 Z0d
#define Jb4 Z14
#define Jb5 JJ3
#define Jb6 SparseFlashAttentionGradTilingData
#define Jb7 Z2k
#define Jb8 Z1U
#define Jb9 Z0V
#define Jba VSEL_TENSOR_SCALAR_MODE
#define Jbb feature
#define Jam Z0m
#define Jan Z4b
#define Jao Z1g
#define Jap offsets
#define Jaq JIn
#define Jar Z1l
#define Jas Z4c
#define Jat Z2l
#define Jau Z3m
#define Jav CrossCoreWaitFlag
#define Jaw Z1z
#define Jax CFG_ROW_MAJOR
#define Jay LoadData2DParams
#define Jaz Z2m
#define JaA Z3C
#define JaB Z38
#define JaC RoundMode
#define JaD capacity
#define JaE candidate
#define JaF Z1r
#define JaG FixpipeParamsV220
#define JaH Z4o
#define JaI reduced
#define JaJ Z3D
#define JaK EVENT_ID7
#define JaL SfagPairView
#define Jbd Z28
#define Jbe SfagStageGemmScore448
#define Jbf JJG
#define Jbg CrossCoreSetFlag
#define Jbh Z1O
#define Jbi Z2n
#define Jbj CompareScalar
#define Jbk Z0p
#define Jbl Z2N
#define Jbm ropeOut
#define JbN counts
#define JbO EVENT_ID1
#define JbP JJn
#define JbQ cmatrixInitVal
#define JbR fetched
#define JbS remaining
#define JbT Fixpipe
#define JbU Z2O
#define JbV CAST_NONE
#define JbW goRows
#define JbX sorted
#define JbY dstNdStride
#define JbZ row128
#define Jc0 result
#define Jc1 Z3E
#define Jc2 gridcounts
#define Jc3 prepared
#define Jc4 LoadData
#define Jc5 JIC
#define Jc6 descriptor
#define Jc7 tokens
#define Jc8 PIPE_MTE2
#define Jc9 SfagStageGemm
#define Jca Z3F
#define Jcb Z3G
#define Jcc CreateVecIndex
#define Jbn Z3T
#define Jbo srcNdStride
#define Jbp fixPending
#define Jbq writeRope
#define Jbr CMPMODE
#define Jbs height
#define Jbt MTE1_M
#define Jbu nullptr
#define Jbv Z3H
#define Jbw Z3U
#define Jbx EVENT_ID2
#define Jby SELMODE
#define Jbz BinaryRepeatParams
#define JbA Z3n
#define JbB indices
#define JbC Z3I
#define JbD EVENT_ID5
#define JbE packed
#define JbF PipelineWidth
#define JbG Select
#define JbH positive
#define JbI MmadParams
#define JbJ Z2P
#define JbK srcStride
#define JbL TailResidentScore
#define JbM EVENT_ID3
#define Jce FIX_MTE2
#define Jcf EVENT_ID4
#define Jcg keyGroupMask
#define Jch slotStart
#define Jci cmatrixSource
#define Jcj BroadcastWeightRows
#define Jck V_MTE3
#define Jcl astride
#define Jcm directOut
#define Jcn Prefix
#define JcO packet
#define JcP output
#define JcQ outRows
#define JcR Z29
#define JcS continue
#define JcT quantPre
#define JcU JIc
#define JcV MTE3_S
#define JcW weights
#define JcX SetMaskNorm
#define JcY positions
#define JcZ SfagSpatialGrad128WholeQueryB
#define Jd0 static_assert
#define Jd1 target
#define Jd2 boundary
#define Jd3 DrainFix
#define Jd4 SetValue
#define Jd5 fullL1
#define Jd6 offset
#define Jd7 slotFloats
#define Jd8 cPending
#define Jd9 residentScore
#define Jda CacheLine
#define Jdb SfagPairSync
#define Jdc marker
#define Jdd spatialFlags
#define Jco GetValue
#define Jcp beginTask
#define Jcq SetAtomicNone
#define Jcr pending
#define Jcs kvDone
#define Jct rawPacked
#define Jcu GetSystemCycle
#define Jcv Gather
#define Jcw V_MTE2
#define Jcx QParts
#define Jcy taskStride
#define Jcz fullStep
#define JcA reluEn
#define JcB PipelineRows
#define JcC prefixB
#define JcD SyncAll
#define JcE validUnique
#define JcF ResetMask
#define JcG MTE2_MTE3
#define JcH finish
#define JcI MTE2_V
#define JcJ packedCounts
#define JcK prefix
#define JcL DcciDst
#define JcM WaitMath
#define JcN SpatialPublishWord
#define Jdf CopyRepeatParams
#define Jdg PackHigh16x512NZ
#define Jdh SetAtomicAdd
#define Jdi batchStart
#define Jdj endTask
#define Jdk partial
#define Jdl rawHalf
#define Jdm batchMask
#define Jdn rowPrefix
#define Jdo CAST_RINT
#define JdP featureHalf
#define JdQ volatile
#define JdR VSEL_TENSOR_TENSOR_MODE
#define JdS ScorePipeline
#define JdT scoreMM
#define JdU SpatialReadWithUpper
#define JdV batchCount
#define JdW GatherMaskParams
#define JdX GradTask
#define JdY SpatialRead
#define JdZ Resident448Score256
#define Je0 Z4p
#define Je1 SetLowGate
#define Je2 firstToken
#define Je3 dsLowFlags
#define Je4 ASCEND_IS_AIC
#define Je5 c6RetainedL1
#define Je6 overlayB
#define Je7 previous
#define Je8 F322F16
#define Je9 GetSubBlockIdx
#define Jea residentRow
#define Jeb C3_COMPLETE
#define Jec scoreNext
#define Jed d2chunk
#define Jee qpartial
#define Jdp PairFullExactLow
#define Jdq FULL_COUNT
#define Jdr SpatialUniqueRaw
#define Jds TallBatchTile128
#define Jdt WideBatchTile128
#define Jdu rowCacheValid
#define Jdv current
#define Jdw spatialRing
#define Jdx C9ReadyMark
#define Jdy SpatialPublish
#define Jdz ScorePipelineWindow
#define JdA SfagPairPrepareGate
#define JdB PrepareFullSpanByte
#define JdC descending
#define JdD int16_t
#define JdE DirectDQMigrationGate
#define JdF ShiftLeft
#define JdG SfagSpatialGrad128
#define JdH PIPE_FIX
#define JdI SpatialFinishQueryPrefix
#define JdJ InitPrepareBuffers
#define JdK slotBegin
#define JdL PIPE_ALL
#define JdM mathPending
#define JdN Z0s
#define JdO ShiftRight
#define Jeg keyGroup
#define Jeh LocalCompleteLine
#define Jei InitWeightBuffers
#define Jej GradientsWindow
#define Jek traceBase
#define Jel published
#define Jem Prepare
#define Jen packedV
#define Jeo DataCopyPad
#define Jep globalFirst
#define JeQ outScore
#define JeR lowGate
#define JeS payload
#define JeT Gradients
#define JeU PIPE_MTE3
#define JeV cachedRaw
#define JeW RunSpatialUniquePairICWholeQueryBAivLeader
#define JeX pairCache
#define JeY WeightsPipelineWindow
#define JeZ groupKeys
#define Jf0 packedK
#define Jf1 PrepareLargeCountRank
#define Jf2 C9ReadyWord
#define Jf3 GatherMask
#define Jf4 PrepareDelta
#define Jf5 PackTile
#define Jf6 selected
#define Jf7 BorrowQueryWorkspace
#define Jf8 SpatialGroupRawReadyRead
#define Jf9 SpatialAivLeaderAggregateRowNz
#define Jfa keyFirst
#define Jfb c6WbScores
#define Jfc SfagC5GradWriterGate
#define Jfd DataCopyExtParams
#define Jfe JIp
#define Jff MTE3_MTE2
#define Jeq cachedTag
#define Jer EARLY_RECYCLE
#define Jes OperandsReady
#define Jet nextRank
#define Jeu WholeReduceSum
#define Jev cacheValid
#define Jew SpatialOneFullQueryBDQ
#define Jex SpatialAcquireKeyGroup
#define Jey SpatialActualAggregate
#define Jez wholeTokens
#define JeA SharedKvScoreValueWide
#define JeB SFAG_R1_TILE
#define JeD SpatialGroupRawReadyPublish
#define JeE SpatialGroupBalancedDQCount
#define JeF InitScore448
#define JeG SfagPreparePair
#define JeH SfagTransposedNzScore
#define JeI BorrowWorkspace
#define JeJ transferToScore
#define JeK TallPairedScore
#define JeL PrepareFullSpan
#define JeM dpOffset
#define JeN SfagSpatialAV
#define JeP retireEnabled
#define Jfi acquiredKeyGroups
#define Jfj halfOutput
#define Jfk SpatialFullQueryBTagCount
#define Jfl valueBatch
#define Jfm reduction
#define Jfn deltaRows
#define Jfo halfScore
#define Jfp pairCount
#define Jfq RunSpatialUniquePairAVNzAivLeader
#define JfR threshold
#define JfS c6WbInverse
#define JfT scoreHalf
#define JfU h2GradFixPending
#define JfV H2OwnerStatStage
#define JfW LowFlagScope
#define JfX SpatialServicePendingQueryPrefix
#define JfY pendingPrefixRow
#define JfZ BeginFix
#define Jg1 c3ReadyFlags
#define Jg2 C3Blocks
#define Jg3 SFAG_R1_OWNER_CORES
#define Jg4 SpatialTryOneGroupScoreBalanced
#define Jg5 SupportGroups
#define Jg6 actualSupport
#define Jg7 ScoreDrain448
#define Jg8 residentValid
#define Jg9 BroadcastRows
#define Jga DsLowZero
#define Jgb WeightsPipeline
#define Jgc packetKeys
#define Jgd C6CountPacketReady
#define Jge Z2C
#define Jgf initializeStats
#define Jgg PrepareOriginal
#define Jfr blockL1Busy
#define Jfs loadedBMask
#define Jft SpatialOneFullQueryBKV
#define Jfu RunSpatialUniquePairAV
#define Jfv RunSpatialUniquePairIC
#define Jfw packetCount
#define Jfx c6HalfCodes
#define Jfy queryPrefix
#define Jfz SpatialGroupScoreBalancedGate
#define JfA PrepareDeltaMoved
#define JfB SpatialActualSkip
#define JfC BindBorrowed
#define JfD fixedOffsets
#define JfE protectEarly
#define JfF RunSpatialGroupScoreBalanced
#define JfG compactStage
#define JfH H3ZeroDQPrefix
#define JfI GatherPackedTileAsync
#define JfJ SfagC3GeometryStorage
#define JfK SfagC3ReadyOps
#define JfL ResidentTailPaired128
#define JfM reuseConstants
#define JfN isChannelSplit
#define JfO h2R6ResidentReleased
#define JfP ROOT_C8_DQ_EARLY
#define JfQ Key256StationaryFourQueryNzK128SkewDualC
#define Jgi fullBReadPending
#define Jgj zeroCounts
#define Jgk ZeroPack16x512NZ
#define Jgl SfagTemporalScore448
#define Jgm RetireWord
#define Jgn DataCopyPadExtParams
#define Jgo preference
#define Jgp tokenBegin
#define Jgq SfagSparseHalfVector
#define Jgr Z39
#define JgP RunSpatialUniquePairICGroupScoreBalanced
#define JgQ RunSpatialUniquePairAVGroupScoreBalanced
#define JgR SpatialGroupBalancedDQTask
#define JgS C2CompactRows
#define JgT SfagC3Submission20261026R1
#define JgV gradKeyRope
#define JgW rowCacheTag
#define JgY h2R28L1Released
#define JgZ H3ScoreCopyBank
#define Jh0 Resident448GroupTailReuse
#define Jh1 Compute16x512NZ
#define Jh2 SpatialFourOutputJobCount
#define Jh3 SpatialAivLeaderServiceNz
#define Jh4 CopyNZRowStat16
#define Jh5 MrgSort4Info
#define Jh6 floatOffsets
#define Jh7 h2OwnerArena
#define Jh8 InitH3ResidentPair
#define Jh9 LocalCompleteWrite
#define Jha H3ResidentPairTask
#define Jhb nGlobalFirst
#define Jhc JI8
#define Jhd Z4d
#define Jhe ApplyQuery16NZ
#define Jgs grp3SkipActive
#define Jgt SpatialTryOneFullQueryB
#define Jgu SpatialGroupBalancedTag
#define Jgv H3ResidentPairGradients
#define Jgw DrainKvHalfPacketsAhead
#define Jgx SfagC3CompactTilingData
#define Jgy H2OwnerCachedRows
#define Jgz pairSupportGroups
#define JgA startupDiagnostic
#define JgB C7LoadNativeBTile
#define JgC ENTIRE_DATA_CACHE
#define JgD SpatialGroupScoreStart
#define JgE SpatialGroupScoreCount
#define JgF RunH3ResidentPairFrame
#define JgG c6ResidentOffset
#define JgH HrResidentInputs
#define JgI SfagC7GPProgress
#define JgJ C7ScoreInputBank
#define JgK SetMMColumnMajor
#define JgL opportunityStart
#define JgM SpatialResetPair
#define JgN SingleValuePlane
#endif

#define JHZ Z06
#define JI0 Z0b
#define JI1 GetUserWorkspace
#define JI2 Z0n
#define JI3 sparseIndices
#define JI4 softmaxMax
#define JI5 softmaxSum
#define JI6 Z0Z
#define JI7 bool
#define JI8 SfagDenseStageWindowPrep
#define JI9 dQueryRope
#define JIa value
#define JIb auto
#define JIc Z2a
#define JId batch
#define JIe core
#define JIf score
#define JIg slot
#define JIh token
#define JIi task
#define JIj true
#define JIk group
#define JIl dKeyRope
#define JIm void
#define JIn Z1P
#define JIo keyRope
#define JIp queryRope
#define JIq plane
#define JIr part
#define JIs stage
#define JIt Z0O
#define JIu count
#define JIv nrows
#define JIw Process
#define JIx dQuery
#define JIy else
#define JIz dValue
#define JIA half
#define JIB query
#define JIC IsSameType
#define JIE SfagSparseHalfOwnerProcess
#define JIF arena
#define JIG panel
#define JIH width
#define JII keys
#define JIJ valid
#define JIK bank
#define JIL begin
#define JIM hasQueryLengths
#define JIN sortedGroups
#define JIO KernelSparseFlashAttentionGradDuplicate
#define JIP SetSysWorkspace
#define JIQ span
#define JIR delta
#define JIS bytefloat
#define JIT SfagDenseStage
#define JIU tasks
#define JIV KernelSparseFlashAttentionGradBase
#define JIW item
#define JIX row0
#define JIY Sync
#define JIZ hasKvLengths
#define JJ0 class
#define JJ1 pipe
#define JJ2 iter
#define JJ3 Z0D
#define JJ4 actualSeqQuery
#define JJ5 base
#define JJ6 merged
#define JJ7 pair
#define JJ8 input
#define JJ9 gridb
#define JJa start
#define JJb Z0W
#define JJc consecutive
#define JJd Z05
#define JJe flags
#define JJf qelems
#define JJg ndNum
#define JJh sparseMode
#define JJi step
#define JJj dKey
#define JJk blockSize
#define JJl dummy
#define JJm ready
#define JJn spatialPipeMode
#define JJp dOut
#define JJq scale
#define JJr Z2Q
#define JJs countTask
#define JJt groups
#define JJu SharedKvScore
#define JJv slotHalf
#define JJw denseHistogram
#define JJx HrGradTask
#define JJy markerBits
#define JJz nSize
#define JJA temp
#define JJB actualSeqKv
#define JJC bitkeys
#define JJD d2slots
#define JJE header
#define JJF key128
#define JJG Z3o
#define JJH Cast
#define JJI SfagDenseStageBaseline
#define JJJ cores
#define JJK dqDone
#define JJL rows
#define JJM Z0i
#define JJN fullB
#define JJO next
#define JJP rest
#define JJQ refill
#define JJR owner
#define JJS high
#define JJT Z03
#define JJU bstride
#define JJV headGroup
#define JJW labels
#define JJX qlong
#define JJY rowAbs
#define JJZ Fence
#define JK0 HrSqPrefetch
#define JK1 sharedGroups
#define JK2 C8Store
#define JK3 ResidentRowScore256
#define JK4 allRefs
#define JK5 line
#define JK6 overlayA
#define JK7 Z00

#define Y00 JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.s2+JIH
#define Y0N JFQ Jd7=(Jbw()||(W4w)W49())W53:
#define Y0h W49())W53:32768 JFQ JeM=(Jbw()||(W4w)W49())?32768
#define Y0j JFQ Jd7=(Jbw()||(W4w)W49())W53:32768 JFQ JeM=(Jbw()
#define Y0q JEL JIf,dp,Z1w JxE 32],sums JxE 32],JIR,gsH,gsL,psH,psL
#define Y11 JEk JJ9 JvB W0K JF6 ones,1.0f,JIv)JF6 JaI[JIv]Juj(JbH,W2p JoH);Div(Ja9 ones,Jtb
#define Y1A ){if(
#define Y1J *rt*kt JFQ Jd7=(Jbw()||(W4w)W49())W53:32768 JFQ JeM=
#define Y1Q Z3w(Z1k,JIe,Z21,JJK,Jcs,Jdm,Jcg,
#define Y1e );if(
#define Y1g ))W53:32768 JFQ JeM=(Jbw()||(W4w)W49())?32768:16384 JFQ
#define Y1h ,JIv)JF6 JaI[JIv]Juj(JbH,W2p JoH);Div(Ja9 ones,Jtb);JbG(Ja9 JbH,W2p
#define Y1l W4w
#define Y27 W0Y
#define Y2P W0P W4v
#define Y2c ;if(
#define Y2g JJK!=dqAll||Jcs[0]!=kvAll||Jcs[1]!=kvAll)Z3w(Z1k,JIe,Z21,JJK,Jcs,Jdm,
#define Y2t JFr W1e=(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,
#define Y2u W1t,
#define Y2v JFr W1e=(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b W2s/(rt*JnJ
#define Y34 ones JGC 2304 JEk JJ9 JvB W0K JF6 ones,1.0f,JIv)JF6 JaI[JIv]Juj(JbH,W2p JoH);Div(Ja9
#define Y3C ),cp)W1b Add(dst[Z0S],src[Z0S],JIs J9M(64)JG7
#define Y3j (Jbw()||(W4w)W49())?32768:16384 W5o
#define Y44 )Z3w(Z1k,JIe,Z21,JJK,Jcs,Jdm,Jcg
#define Y4A )JEy a2b,65536)JEy b2b,65536)JEy cb,131072);a1=ab JvC b1=bb JvC a2=a2b JvC b2=b2b
#define Y4J ,JI7 Z3a,JI7 Z3p,JI7 Z3J>__global__ JaP JIm sparse_flash_attention_grad(Jam JIB J9Y key J9Y JIa J9Y JI3 J9Y JJp J9Y out J9Y JI4 J9Y JI5 J9Y
#define Y4P JyI W41 JcV>(JH2
#define Y4W JHC JIi=Jgp*Jau+vi W4d<Z1Z*Jau Jz7 W2s/Jau JkS-JJa:JIQ Jq4 n)JF6 JIu,0,
#define Y4m W1w
#define Y57 &&W0P
#define Y5L JFr Jaj dqAll=(W16(2*Z21))-1 JFT kvAll=JHm 1)<<kvCount)-1 J9i Jdm[8]=
#define Y5W ){if(JCi 8u)!=0u&&skipKV){if(JCi 16u)!=0u){vec.
#define Y5q ;++i)x JF8 i,0 JGz c JAW JFt c+=20u;}x JF8 0,0x5346414754494d45ull);x JF8 1,c);
#define Y62 ||(W4w)W49())?32768
#define Y64 W41 JcV>(
#define Y65 W1a JcV>(JH2
#define Y6E JCi 8u)!=0u&&skipKV){if(JCi 16u)!=0u
#define Y70 -1)/JIH JFT JIU=Jwe*rt*kt JFQ Jd7=(Jbw()||(W4w)W49(
#define Y77 JEL JIf,dp,Z1w JxE 32],sums JxE 32],JIR,gsH,gsL,psH,psL);}JB4>(2)
#define Y79 Jho W0E,A)Jve,A JGi JII,ix JAM n W2Y,JI2(JAN t.s2))JEu A JG2 JII,JIJ
#define Y7C ;Jgl<T>JdT;JdT.JeF(JJ1);JdG<T>Z1k;Z1k.JfC(JdT.JeI(),JdT.
#define Y7D JJ4 J9Y JJB J9Y Jfe J9Y JIo J9Y JIx J9Y JJj J9Y JIz J9Y JI9 J9Y JIl J9Y JcU J9Y tiling)W1r(Jb6);GET_TILING_DATA_WITH_STRUCT(Jb6,
#define Y7b s2+JIH-1)/JIH JFT JIU=Jwe*rt*kt JFQ Jd7=(Jbw()||(W4w)
#define Y7n &&W0P!
#define Y7z JG2 JII,JIJ,JII,-
#define Y8K W0P
#define Y8p 8192
#define Y8v W1a
#define Y9X JIe JCm passes=Z2z(JIe JFr Jaj Z21=Z47(JIe),kvCount=Z3Q(JIe JFr Jaj dqAll=(W16(2*Z21
#define Y9Y Jdi J9g JdV W0Y JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(
#define Y9g 65536
#define Y9j J9b T,JI7 Z0a,JI7 Z04,JI7 Z07,JI7 Z0j,JI7 Z0k,JI7 JdN,JI7 Jbk,JI7 Z0t,JI7 Z0o,JI7 Z08
#define YAI 17)JHA x=JHf;Jxs 512u;++i)x JF8 i,0 JGz c JAW JFt c+=20u;}x JF8 0,0x5346414754494d45ull
#define YAm )*r JmR JFk JIf,mode?Jdw[JIi*32768ull+JI6*nr]:JHd JzT J9y dp,mode?JHi+JIi*32768ull
#define YBU ,kt=(t.s2+
#define YBZ JbH,W2p
#define YCE CopyK Jzq,0,0,JCZ);J9x W2F JAX JCV JFV Z0f=0;Z0f<Jfm;Z0f+=128 JEf=(Z0f/
#define YCk -128)/16)}JFU dst[JIX*W5B],raw,cp JwL JHJ W1a Z0y JCY JvQ ApplyRow16 JoJ>
#define YD0 Jyo p){p JCO 245760)JEy bb,262144)JEy a2b,65536)JEy b2b,65536)JEy cb,131072);a1=ab
#define YDI (rt*JnJ?t.W4z:JIH
#define YE0 JbG(Ja9 JbH,W2p JA7
#define YE3 cp{1,1,
#define YE5 =Z2Z(JIe
#define YE9 &&W4v
#define YEa JaR Z48
#define YF2 JJc=W1w!W0F r W5z t.s2>=8192 JFT Jcp=JJc?JIU*JIe/JFq:JIe JFT Jdj
#define YFR );++pass){if(Jel<=pass*4 JBz JbZ=Z2Z(JIe,pass JAc JIA=0;JIA<2;++JIA
#define YFc [JIk*8 W0y+Z0S],src[JIk*8 W0y+Z0S],J5H JAG Z3d(JzA JJ7){JIY()
#define YFy {Jka Jx5 W1t,8192 Jqf 8192 J9s raw,o[W3F,8192 W2k JcQ,W23 8192 W0I Mul(
#define YGl J7w=4 Jpn J7x Jb1 x JEH MULT>JFP ApplyRow JoJ>JIs JFc Jbz cp{1,1,0,16,16,1 JCe Z0S
#define YGp W0Y JJ7=vid+40*Jc3;av.Z3d(t,JbB,JbN,go,o,JIR,JJ7 JyI W41 JcV
#define YHH 32 JDH W48=JHy JaQ*>(JK5 JAZ
#define YHI ;bit/=2){Adds(JaE,Z0C JFb
#define YHX <2;++
#define YHY *65536
#define YHr W1w!W0F r W5z t.s2>=8192
#define YI3 (JFn JJ5=J9I Jb3 Jar*>(JHy JaQ*>(Je3 JAZ);Jdw Jvx(JJ5));Jdd JuS(JJ5+41943040ull
#define YI6 ){(JIm)Jdl;(JIm)Jct;J0X JFj Z1J=!Jdu||JgW!=JIX W0b Jfo W1o<W1y(65536
#define YIA JIs JFc Jbz cp{1,1,0,64,64,1 JCe Z0S=0;Z0S<W4X Z0S+=64 JAV MULT)Mul
#define YIH [Z0S],JIs J9M(64)JG7 16),cp)JwF W54 ApplyCounts16 JoJ>JbE JFc Jbz cp{1,1,1,
#define YIJ ]&255u)!=255u JG8 Jft(mm,tag,JJF,!(J9d]&256u));Jcs W5G|=Z1R;Z2A
#define YIP z JHA JII JAe(a,69760+z W1p,2048 JFn JJl=JaL JHk(a,77952+z W1p,2048 JFn dst=
#define YIa ,8})W5r(Z2g,Z0P JG7 8),Jbv{1,8})W5r(Z2h,Jfn JG7 8)Jo9 JHM
#define YJf W0F W0P
#define YJg W0y
#define YJk (dst[Z0S],src[Z0S],JIs J9M(64)JG7 16),cp)W1b Add(dst[Z0S],src
#define YJs W14+Z0S],src[JIk*8 W14+Z0S],JbE[JIk W14+Z0S Jv3 8),cp JuQ W54
#define YK7 JFn dst=JbX[z*W2b;dst JCR Jve JBR,2048 JuQ W0N,W1j(dst,JII,JJl,W2K
#define YKO Jcg[32]={0},JJK=0 J9R Jcs[2]={0,0}J9i
#define YKX JDv 2048;bit;bit/=2){Adds(JaE,Z0C JFb
#define YKq dp JEL 8192 JuQ PackHigh(gs,dp,raw,t,JIX,col,JHZ);PackHigh(ps,JIf,raw,t,JIX
#define YL9 rp{1,128,64,0,64 JwP
#define YLB ,64,W5t JIk=0;JIk<2;++JIk)JFV Z0S=0;Z0S<W4X Z0S+=64)Mul(dst
#define YLD ,JbN[(Je2+JIk)*W5B],128 JuY Jao Jwn Jao JFn JbE=raw JyX JbE JB7 JHL JbE,grid
#define YLF ,Jgj[s],s)JFU JIR[JJ7*16],zeroDelta,16 JyI W41 JcV>(JH2 JIY(JBY J5g J5T
#define YLJ 0;z<2;++z JHA JII JAU(69760+z W1p,2048 JFn JJl W1o JHk(77952+z W1p,2048
#define YLT ;++h JHu raw[h W1t Jxf 64 W0I JHN dq W5w W1t,raw,8192 JA6 JBu 4);++traceChunk
#define YLj j=0;j<
#define YLk )*65536
#define YLl Z01
#define YLp )*JJf W17 W1t,8192 W0I Add W4W Jze J9n tw+2)JHL raw,JIf Jwy JDJ tw+3 JBi ROPE
#define YLy ,16,W5t JIk=0;JIk<8;++JIk)JFV Z0S=0;Z0S<W0D Z0S+=64)Mul(dst
#define YM6 J9j s=t.s2,n=2*s W0b JJ8 JC3(135296,4096 JFn JbX JAU(0,16384 JFn JJA JAU(151680
#define YMh JFq*2,vi JAW Jaj traceChunk=0;JHC W4h JAP vi)*16;row<JJL;W4M vc*16 W0Y tw
#define Yac ones,Jtb);JbG(Ja9 JbH,W2p JA7 JIv JuQ J0T JEL W1C JIY();Jp2
#define Yaf t.s2
#define Yah )JBu 2 JFr W1e=(JIi%kt)*JIH,W4h(JIi/kt%rt)*Jbs,b W2s/(
#define Yam JIi/(rt*JnJ?t.W4z:JIH,JIA=(mr+1)/2 JFQ JIL=sub*JIA,end
#define Yap rt*JnJ?t.W4z:JIH,JIA=(mr+1)/2 JFQ JIL=sub*JIA,end=JIL+
#define Yb5 ))-1 JFT kvAll=JHm 1)<<kvCount)-1 J9i Jdm[8]={0},Jcg[32]={0
#define YbY (135424 JER grid JC3(151808,1024 JFn Z2f JAU(155904,128 JFn Z2g JAU(157952,128 JFn Z2h JAU(160000,128 JFn Jfn
#define Yc4 JBx vc=JFq*2,vi JAW JoS s1 JFQ JIQ=(JFo+Jau*8-1)/(Jau*8)*8;
#define Ycc mode W0Y JIH=JbF(),Jbs=JcB(),rt=(r+Jbs-1)/Jbs,kt=(t.
#define Ye9 ],JIs J9M(64)JG7
#define YeM JAU(155904,512 JFn Z2g JAU(157952,512 JFn Z2h JAU(160000,512 JFn Jfn JAU(162048,64 JFn Z15 JAU(162304,
#define Yei (q[qi W1t,go[qi W1t,k[ki W1t,v[
#define Yem (),JdT.BorrowA2(),JdT.BorrowB2(),JdT
#define YfC ,JII,W0E JA7 A W2Y JEJ GE,A JG2 JII,JIJ,JII,W0E JA7 A
#define YfW JJK!=dqAll||Jcs[0]!=kvAll||Jcs[1]!=
#define Yg6 ){vec.JaR Z48
#define Yg9 W41
#define YgT [1]!=kvAll)Z3w(Z1k,JIe,Z21,JJK,
#define Yga JAI JIm
#define Ygd )JDl JAa>(2);mm.GradQ(gsH,gsL,JIg==JdK JFr Jaj Z2R=skipKV?SfagC4ActiveKvPackets(Z4e,JIg):
#define Ygx JaQ highest=JAN highest0>highest1?highest0:highest1);Jgz=JA3(highest+512)/512)JAg S_V Jxr S_V JEv if(highest
#define Yh1 JHA rawBase=J9I Jb3 Jar*>(JHy JI2*>(wq JAZ)JAj groupReady;groupReady JuS(rawBase+65536)JFn JK5=groupReady JF7
#define Yh8 <T>Z1k;Z1k.JfC(JdT.JeI(),JdT
#define Yhj t.s2+JIH-1)/JIH JFT JIU
#define Yhl W0P!
#define Yj2 JFr Jaj
#define YjT JIi/(rt*JnJ?t.W4z
#define Ykm JAU(162048,16 JFn Z15 JAU(162304,16 JFn Z0P JAU(162560,64 JFr JI7 Z1J=!Jdu||JgW!=JIX W0b Jfo
#define Yl8 ,8192
#define Yl9 2048
#define YlF BorrowA2(),JdT.BorrowB2(),JdT.BorrowC()JAc pass=0;pass<passes;++pass W0Y JbZ=Z2Z(
#define YlQ Z15,Z15,W0E,64 JGV ones,1.0f,64 JFL Z0P JEJ GT JFY 64)JuQ Div(Z0P,ones,Z0P JFb 64)
#define Ylg 64 JFn Z0P JAU(162560,64 JFr JI7 Z1J=!Jdu||JgW!=JIX J9B inputCp{64,16,240 JBM(JIf,Z2I,
#define Ylo PackHigh JtS>raw JD3 JCz JEW{JJH(raw,src JA5 8192);if(Z3O JG9 JHo Jcw>(JbM Jyw JbM W4r Jck
#define Ym4 ?1:JFq;JHC JIi=W5g<Jdj W4d+=
#define YmB JER Z1x W1o<W1y(81920,8192)JDn JfT,Z1q;JfT Jpo(Z2I JAZ));Z1q Jpo(ringDp JAZ)JFr
#define YmM W41 JcV>(JH2 J9V
#define YmW 64)JA6 JEZ z*8192+W1m,W0E JHK JAn V_S Jxr V_S JEv JI0 JI2 highest0=JbX JBV,highest1=JbX JFf 8192 JFr
#define Ymk JcV>(JH2
#define YnW =J9I Jb3 Jar*>(JHy JaQ*>(Je3 JAZ);
#define Yns JFn ones JGC 2304 JEk JJ9 JvB W0K JF6 ones,1.0f,JIv)
#define Yo3 ,t.s2
#define Yoa ;JgW=JIX;Jdu JHz}W5f t.JJq,8192 JuQ ApplyRow W4c>JC1 Z2f);Exp JC1 8192 JuQ ApplyRow<W1j
#define Yog JC1 Z2g);ApplyCounts JC1 JbE);ApplyRow W4c>(dp,dp,Z2h);Mul(dp,dp,JIf,8192 JG0 dp
#define Yq4 Jrt JHA JIf JAU(0,8192),dp JAU(32768 JER raw W1o<T>(135424 JER grid JC3(151808,1024 JFn Z2f
#define YqP ,gsH,gsL,psH,psL
#define YqW JHo JcV>(
#define Yr2 +128<slotEnd){if(JCi 8u)!=0u&&skipKV){if(JCi 16u)!=0u){vec.JaR Z48<JCJ
#define Yrf <slotEnd){if(JCi 8u)!=0u&&skipKV){if(JCi 16u)!=0u){vec.JaR Z48<JCJ JIg+
#define Yrl JG2 Z0P,JbH,Z0P,0.0f JA7 64 JG0 Jfn,Jfn,W0E,64 JDi Brcb(Z2f,Z15 JG7 8),Jbv{1
#define Ys9 passes*2;++JIg)Z2J=Z2J&&Jdn W3f]>=4;if(JJK!=dqAll||Jcs[0]!=kvAll||Jcs
#define YsE JAe(a,
#define YsF 32768
#define Ysw Jdm[8]={0},Jcg[32]={0
#define YtN ,col,JIj JAI JIm Z3R JGy J7Z&t Jpe JHA JIf JAU(0,8192),dp JAU(32768 JER raw W1o<T>
#define Ytp );JbG(Ja9 JbH,W2p JA7 JIv JuQ
#define Yu0 <0 JHA Jgj JC3(69760,n JFn zeroDelta JAU(0,16)JF6 Jgj JBf,n)JF6 zeroDelta,0.0f,16 JiZ W5m
#define YvP JJK!=dqAll||Jcs[0]
#define YvS );Word(
#define YwF inputCp)JFU dp,ringDp,inputCp);if(Z1J)JFk Jfn,JIR[JIX],64)JFU Z15,mx[JIX],
#define YwX JCe c=0;c<256;c+=64)Copy(JbE[c],JbR[c Jv3
#define Ywi =0;Z0S<W0D Z0S+=64 JAV MULT)Mul(dst[Z0S],src[Z0S],JIs J9M(64)JG7 64
#define Ywn JAM 1024 JuQ if(Z1J JHA JbH=grid JC0>()[W4e;JbH JF5 8 JFn ones JAU(131584,64)W2m(
#define YxK JFc Jbz cp{1,1,
#define YxR =Z2Z(JIe,pass
#define YxT [qi W1t,
#define YxX {if(
#define YxY 16384
#define YxZ (tag==Z2A)!=(Jgo==0)JG8 if(Jcs W5G==65535u Jta 0;JJF<16;++JJF W0Y Z1R
#define YzG child=0;child<2;++child W0Y W4x JIk W0y+child W1s JFT ki W3G)*2048 W0z JFT JJ5
#define YzY ;JdT W5T(qr[qi*64],kr[ki*64],JIj,JIj);
#define Yzs JHd+JeM JzT JIY();JHN fw,JIR[JIX],J4l>(JFn ones JGC 2304 JEk JJ9 JvB W0K JF6 ones,1.0f

#define Z00 uint32_t
#define Z01 VALUE
#define Z02 HardEvent
#define Z03 uint64_t
#define Z04 HAS_ROPE
#define Z05 return
#define Z06 false
#define Z07 HAS_VALUE
#define Z08 BYTE_COUNT
#define Z09 int32_t
#define Z0a DUP_PATH
#define Z0b const
#define Z0c M_FIX
#define Z0d __gm__
#define Z0e template
#define Z0f kbase
#define Z0g __aicore__
#define Z0h WaitFlag
#define Z0i GlobalTensor
#define Z0j CUBE_PATH
#define Z0k FW_PATH
#define Z0l inline
#define Z0m GM_ADDR
#define Z0n float
#define Z0o OPT_WINDOW
#define Z0p BATCH_WINDOW
#define Z0q FIX_S
#define Z0r SetFlag
#define Z0s COUNT_RANK
#define Z0t OPT_SCORE
#define Z0u words
#define Z0v fullA
#define Z0w PipeBarrier
#define Z0x static_cast
#define Z0y MTE3_V
#define Z0z static
#define Z0A PIPE_V
#define Z0B bBusy
#define Z0C ranks
#define Z0D uint16_t
#define Z0E mSize
#define Z0F TPipe
#define Z0G local
#define Z0H while
#define Z0I chunk
#define Z0J DataCopy
#define Z0K extra
#define Z0L outDp
#define Z0M FORMAT_ND
#define Z0N aBusy
#define Z0O reinterpret_cast
#define Z0P invRows
#define Z0Q MTE2_S
#define Z0R break
#define Z0S colHalf
#define Z0T kvRows
#define Z0U FIX_M
#define Z0V MTE2_MTE1
#define Z0W SetGlobalBuffer
#define Z0X prefixA
#define Z0Y public
#define Z0Z first
#define Z10 queryN
#define Z11 upper
#define Z12 block
#define Z13 macro
#define Z14 SetSize
#define Z15 mxRows
#define Z16 aMath
#define Z17 VECCALC
#define Z18 sizeof
#define Z19 row64
#define Z1a pitch
#define Z1b EVENT_ID0
#define Z1c packedCp
#define Z1d platform
#define Z1e support
#define Z1f cache
#define Z1g EVENT_ID6
#define Z1h hasValue
#define Z1i Compare
#define Z1j l1Read
#define Z1k gradMM
#define Z1l uint8_t
#define Z1m sparse
#define Z1n worked
#define Z1o AscendC
#define Z1p struct
#define Z1q dpHalf
#define Z1r InitBuffer
#define Z1s hasRope
#define Z1t gradOut
#define Z1u Geometry
#define Z1v DataType
#define Z1w maxima
#define Z1x halfDp
#define Z1y keyRead
#define Z1z MTE1_MTE2
#define Z1A scoreN
#define Z1B NoQuant
#define Z1C ReduceSum
#define Z1D second
#define Z1E outputScoreP
#define Z1F wanted
#define Z1G ParamType
#define Z1H Format
#define Z1I PairFull
#define Z1J loadRows
#define Z1K upper0
#define Z1L upper1
#define Z1M keyBatch
#define Z1N joined
#define Z1O M_MTE1
#define Z1P constexpr
#define Z1Q finished
#define Z1R doneBit
#define Z1S lastTag
#define Z1T logical
#define Z1U GetPhyAddr
#define Z1V namespace
#define Z1W DT_FLOAT16
#define Z1X queryRows
#define Z1Y ostride
#define Z1Z tokenEnd
#define Z20 overlay
#define Z21 dqCount
#define Z22 pairEnd
#define Z23 C5NzWord
#define Z24 markers
#define Z25 Convert
#define Z26 HR_GRAPH
#define Z27 C5NzMark
#define Z28 DataSyncBarrier
#define Z29 MemDsbT
#define Z2a workspace
#define Z2b DT_BF16
#define Z2c OPTIONAL
#define Z2d cOffset
#define Z2e lastToken
#define Z2f mxStage
#define Z2g invStage
#define Z2h deltaStage
#define Z2i HrFlags
#define Z2j CacheMode
#define Z2k Nd2NzParams
#define Z2l Duplicate
#define Z2m GetBlockIdx
#define Z2n GetBlockNum
#define Z2o pairFree
#define Z2p outputDP
#define Z2q KeyCopy
#define Z2r GRAPH_FAILED
#define Z2s GetDataType
#define Z2t AutoContiguous
#define Z2u writeLow
#define Z2v keyPrefix
#define Z2w rowBegin
#define Z2x tagCount
#define Z2y mathBusy
#define Z2z SpatialPasses
#define Z2A lastKVTag
#define Z2B cacheTag
#define Z2C c5RectMode
#define Z2D compensate
#define Z2E hasDValue
#define Z2F H2WbGrid
#define Z2G F322BF16
#define Z2H WITH_LOW
#define Z2I ringScore
#define Z2J aggregated
#define Z2K freeFlag
#define Z2L progress
#define Z2M SetL2CacheHint
#define Z2N const_cast
#define Z2O DataCopyParams
#define Z2P TPosition
#define Z2Q baseIndex
#define Z2R kvPackets
#define Z2S valuePlane
#define Z2T produced
#define Z2U batchTile
#define Z2V splitCount
#define Z2W countPad
#define Z2X MarkDsLow
#define Z2Y HrSqCompute
#define Z2Z SpatialRow128
#define Z30 WaitSlots
#define Z31 readySlot
#define Z32 zeroGroup
#define Z33 HrWaitFlag
#define Z34 pairBegin
#define Z35 nextMacro
#define Z36 bfloat16_t
#define Z37 countWorker
#define Z38 dstStride
#define Z39 COUNT_ALGO
#define Z3a HR_GRAPH_ENTRY
#define Z3c c6WbArena
#define Z3d PreparePair
#define Z3e D2ChunkGate
#define Z3f DrainLocal
#define Z3g C8PublishQ
#define Z3h C8AsyncScore
#define Z3i C8AsyncGrad
#define Z3j C8AsyncConvert
#define Z3k batchBytes
#define Z3l CACHE_MODE_NORMAL
#define Z3m countParts
#define Z3n fullPrefetched
#define Z3o sparseCount
#define Z3p C10_OPT_ENTRY
#define Z3q earlyCount
#define Z3r C5NzInputs
#define Z3s C2NZ128Load
#define Z3t C2StreamFlags
#define Z3u coeffAdvance
#define Z3v gradQueryRope
#define Z3w SpatialTryOneGrad
#define Z3x SpatialGate
#define Z3y rowElements
#define Z3z tensorBytes
#define Z3A LocalTensor
#define Z3B ReinterpretCast
#define Z3C DataCacheCleanAndInvalid
#define Z3D tailResident
#define Z3E SINGLE_CACHE_LINE
#define Z3F CACHELINE_OUT
#define Z3G QuantMode_t
#define Z3H BrcbRepeatParams
#define Z3I LargeCountRank
#define Z3J C10_WB_ENTRY
#define Z3K FORCE_SEPARATE
#define Z3M H4CountType
#define Z3N GetRequiredInputTensor
#define Z3O releaseInputs
#define Z3P SpatialReadWord
#define Z3Q SpatialKVCount
#define Z3R Compute16x512
#define Z3T ASCEND_IS_AIV
#define Z3U Score448Family
#define Z3V INDEPENDENT_GEOMETRY
#define Z3W GetValidUnique
#define Z3X childReadyLine
#define Z3Y pendingQuery
#define Z3Z SpatialPairAggregate
#define Z40 h2R6BReleased
#define Z43 GetOriginShape
#define Z44 ResidentPaired128
#define Z45 WeightsChunk448
#define Z46 SpatialOneGrad
#define Z47 SpatialDQCount
#define Z48 GatherPackedTilePair
#define Z4b LoadData2dTransposeParams
#define Z4c LoadDataWithTranspose
#define Z4d MrgSortSrcList
#define Z4e validUniqueAic
#define Z4f LoadAccumulator
#define Z4g StoreAccumulator
#define Z4h GradKVMixedDispatch191
#define Z4i ringPairPrimed
#define Z4j set_vector_mask
#define Z4k SpatialPrepareOne
#define Z4l C2StreamGPReadyAcquire
#define Z4m SpatialFullQueryBTag
#define Z4n CACHE_MODE_DISABLE
#define Z4o WideBatchSingle128x256
#define Z4p TallBalancedTile128
#define Z4q SFAG_CUBE_RING_BYTES
#define Z4r C2StreamReadyKVHalf
#define Z4u DrainKvHalfPacketsAheadOutputPair
#define Z4v GatherPackedTileIndependent32

#define W00 JaP JaT
#define W01 JI0 Jaj
#define W02 Jal::
#define W03 W00 JIm
#define W04 JaR Get
#define W05 <W02
#define W06 JaS W05
#define W07 JaW W05
#define W08 JI0 JaM
#define W09 Jak<
#define W0a W09 T
#define W0b ;JIb
#define W0c JaO<
#define W0d W0c JaM
#define W0e W0c Jb5
#define W0f W09 JI2
#define W0g JaR Jb2
#define W0h <JaV
#define W0i JaN W0h
#define W0j ;REGISTER_TILING_FOR_TILINGKEY
#define W0k Jb3 JaQ
#define W0l W0i>
#define W0m <JI2
#define W0n W0a>
#define W0o ;W0l
#define W0p ::JIa
#define W0q W04 W0m
#define W0r SparseFlashAttentionGradTilingData t
#define W0s =JHZ
#define W0t &&COMPACT_VECTOR
#define W0u Z0g Z0l
#define W0v Z02::
#define W0w *128u
#define W0x ;for
#define W0y *512
#define W0z +col
#define W0A W06 Jbh
#define W0B Jb0 W0m
#define W0C }JIy
#define W0D 128;
#define W0E -1.0f
#define W0F ROPE&&
#define W0G JA0 JIm
#define W0H 1024&&
#define W0I JA6;
#define W0J s2 W0z
#define W0K 16384]
#define W0L Z0e Get
#define W0M Yaf W0z
#define W0N Sort W0m
#define W0O ;JaU
#define W0P Z01&&
#define W0Q W0u void
#define W0R ;W01
#define W0S ==0u
#define W0T .JId
#define W0U t W0T
#define W0V W0c JaQ
#define W0W W07 Jb9
#define W0X JI0 JI7
#define W0Y JFc Jaj
#define W0Z this->
#define W10 ;W06
#define W11 W00 JI7
#define W12 *256u
#define W13 if Jaq
#define W14 *128
#define W15 *4096u
#define W16 1u<<
#define W17 +row
#define W18 r W17
#define W19 Jwt W18
#define W1a JaS JHo
#define W1b ;JIy
#define W1c Z0b Z00
#define W1d W09 JaQ
#define W1e Jaj col
#define W1f Jbr::
#define W1g busy[
#define W1h C10_OPT W0t
#define W1i if JIn
#define W1j JIj>
#define W1k *2048u
#define W1l {0xAAAAAAAAAAAAAAAAULL
#define W1m 4096]
#define W1n W0d>
#define W1o =View
#define W1p *16384
#define W1q {W01
#define W1r {REGISTER_TILING_DEFAULT
#define W1s *256
#define W1t W0y]
#define W1u Jbt JCV
#define W1v Jbk&&
#define W1w W1v W0P
#define W1x mi W1s
#define W1y JIA>
#define W1z W1y W0p
#define W1A }else
#define W1B W0s;
#define W1C JIQ JuQ
#define W1D 128==
#define W1E .Jb4
#define W1F JaC::
#define W1G W1F JbV
#define W1H W1f GE
#define W1I W1D 0
#define W1J JI7 JI6
#define W1K JGa Y8v
#define W1L *8192u
#define W1M W0D mm
#define W1N JI7 JcH
#define W1O ;JI7
#define W1P JIg*
#define W1Q fb Jvl
#define W1R <W0v
#define W1S Z0h W1R
#define W1T Z0r W1R
#define W1U W0c Jar
#define W1V W0C W1i
#define W1W W07 Jbh
#define W1X W06 Jaw
#define W1Y Jb3 JI2
#define W1Z *512u
#define W20 *16384u
#define W21 Jaj mi
#define W22 2048u JuQ
#define W23 raw JAM
#define W24 Jce JCV
#define W25 Jb1 JHZ
#define W26 JaW JHo
#define W27 JaM row
#define W28 Z07&&
#define W29 W28 Z0j
#define W2a W0C{
#define W2b 8192]
#define W2c ge::
#define W2d W2c Z0M
#define W2e }W03
#define W2f featOff W0S
#define W2g W1z&&
#define W2h W1j JIs
#define W2i Jxu"Transposed NZ Score requires two-byte operands"
#define W2j JI0 Jb6
#define W2k JA6 JHL
#define W2l ;auto
#define W2m ;Muls
#define W2n .W04
#define W2o JIF W2n
#define W2p Ja9 0.0f
#define W2q W0e>
#define W2r W0D fp
#define W2s =JIi
#define W2t W00 Jaj
#define W2u JbT W0m
#define W2v ;JIY
#define W2w &&Yaf
#define W2x ring[
#define W2y W2x ro
#define W2z [JIK
#define W2A Z04&&
#define W2B ::ALL
#define W2C W1o W0m
#define W2D W10 Jb9
#define W2E W07 Jbt
#define W2F JuT JGa
#define W2G JcR W2B
#define W2H JaX<
#define W2I W2H Jb3
#define W2J Ja1 JHZ
#define W2K JJA JFb
#define W2L JFr JaM
#define W2M ++JIr
#define W2N ki W1s
#define W2O W0D if
#define W2P Z0k&&
#define W2Q W2P Z0s
#define W2R *64u
#define W2S JuQ Mins
#define W2T <JaQ
#define W2U col+=
#define W2V W2n W2T
#define W2W W2A W29
#define W2X W1P Jd7
#define W2Y JuQ JE4
#define W2Z JdQ W0k
#define W30 JcA W1B
#define W31 JbI mm
#define W32 Jaj JIe
#define W33 JaM JIi
#define W34 JIr==
#define W35 JIe<
#define W36 JA6 W0b
#define W37 *32768u
#define W38 *1024u
#define W39 [JIr
#define W3a Z0o&&
#define W3b W3a Z08
#define W3c Z0a&&
#define W3d JAk if
#define W3e info JuQ
#define W3f [JIg
#define W3g W2g W3c
#define W3h W10 Z0c
#define W3i W0H Yaf
#define W3j W07 Jaw
#define W3k TBuf<
#define W3l W3k JbJ
#define W3m Jxu"Grad128 requires a two-byte input type"
#define W3n Jwt W0M
#define W3o Jc5<
#define W3p W3o Out
#define W3q Jap JFb
#define W3r JIQ W0I
#define W3s JI7 Jbq
#define W3t JF5 320
#define W3u JA6 Y2c
#define W3v ;W08
#define W3w *65536u
#define W3x Jby::
#define W3y W3x Jba
#define W3z Jbb==
#define W3A .Jb8
#define W3B W1P 2
#define W3C W10 Jbt
#define W3D Jaj Jcl
#define W3E JaA W2T
#define W3F JI6 W1t
#define W3G JG3 JId
#define W3H JF7 JId
#define W3I .JaZ
#define W3J W0f>
#define W3K W0Z kr
#define W3L .JIw
#define W3M JIs W3L
#define W3N =JIj
#define W3O .W0g
#define W3P +sub
#define W3Q Z0p&&
#define W3R W3Q Z0t
#define W3S Z0b Z03
#define W3T W3O W0m
#define W3U W2c DT_INT32
#define W3V W1g JIg
#define W3W W0n gog
#define W3X JuQ Sub
#define W3Y Jaj JJ2
#define W3Z JaG fp
#define W40 JJ0 Out
#define W41 JH2 W1a
#define W42 JFc JaQ
#define W43 JF5 1024u
#define W44 ;W03
#define W45 32768]
#define W46 2048u;
#define W47 /128
#define W48 *addr
#define W49 ||JaH
#define W4a Jae Emit
#define W4b JIe>=
#define W4c <JHZ
#define W4d ;JIi
#define W4e 2048]
#define W4f }W0x
#define W4g 0 W4f
#define W4h row=
#define W4i {Z05
#define W4j W1A W4i
#define W4k Y8p]
#define W4l Y2u Y8p
#define W4m W2H W1Y
#define W4n W2H W0k
#define W4o W1T Z0V
#define W4p W0n JJS
#define W4q JGr pos
#define W4r JCd W02
#define W4s ==2048u
#define W4t ;W2e
#define W4u !=Jbu
#define W4v !ROPE
#define W4w W1w W4v
#define W4x col=
#define W4y -col
#define W4z s2 W4y
#define W4A *16u
#define W4B using Z1V
#define W4C W3z 0
#define W4D W0Z Input
#define W4E W0D mp
#define W4F Jd5[
#define W4G W4F as
#define W4H Jaj j
#define W4I 2048u JFn
#define W4J ++mi
#define W4K ;W4J
#define W4L 16 W4K
#define W4M row+=
#define W4N W1I JAd
#define W4O W0z W47
#define W4P W07 Z0c
#define W4Q W06 Jb9
#define W4R Jc2 JAM
#define W4S JbR Jvq
#define W4T JaL W0m
#define W4U JaE J9t
#define W4V JJg JiS
#define W4W JC1 JJP
#define W4X 512;
#define W4Y W4X fp
#define W4Z if Jbn
#define W50 W4F ns
#define W51 W35 8u
#define W52 Jay bp
#define W53 ?65536
#define W54 }JFP
#define W55 W3o T
#define W56 ;JFV
#define W57 type==
#define W58 W4X Jbb
#define W59 W1B Mmad
#define W5a W0n bg
#define W5b W0c Jaj
#define W5c W09 W1y
#define W5d W09 Out
#define W5e SfagSmallrowWholeCount<
#define W5f Muls JC1
#define W5g Jcp W4d
#define W5h Jcb::
#define W5i Jb7 rp
#define W5j Jb1 JIj
#define W5k JK5 W3A
#define W5l JIY JAi
#define W5m JFU JbN
#define W5n JFQ Jcy
#define W5o JFQ JIe
#define W5p JF5 4096u
#define W5q JAP key
#define W5r ;Brcb
#define W5s 512&&
#define W5t 0 JCe
#define W5u W1B}
#define W5v ]JuU
#define W5w [row
#define W5x [JIW
#define W5y W3l::
#define W5z >=W0H
#define W5A .W0J
#define W5B t W5A
#define W5C +JIr
#define W5D +JIg
#define W5E *1024
#define W5F {0xAAAAAAAAAAAAAAAAull
#define W5G W5x]
#define W5H W2U 64
#define W5I W0n kg
#define W5J SfagStageGemmScore448<
#define W5K JbI mp
#define W5L ;W3M
#define W5M JIs W5L
#define W5N W2h W5L
#define W5O 65536]
#define W5P 4096&&
#define W5Q .W0q
#define W5R JIF W5Q
#define W5S fb W5Q
#define W5T .Jcn
#define W5U mm W5T
