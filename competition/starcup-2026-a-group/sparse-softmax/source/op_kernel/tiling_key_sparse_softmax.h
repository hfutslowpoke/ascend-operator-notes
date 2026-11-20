// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SparseSoftmax,
    // PATH6/PATH7: raw-UB index routes (index 模式、tasks=1、axis=8、inner=1、广播、fp32/fp16；以及 index 模式、tasks=2、axis=4、inner=1、广播、fp32)。
    // PATH8/PATH9: raw-UB ptr routes (ptr 模式、tasks=1、axis=8、inner=1、ptrLength=5；以及 ptr 模式、tasks=2、axis=4、inner=1、ptrLength=3)，
    // both fp32-only because the RawPtr helper takes the fp32 LocalTensor views directly.
    // PATH10: streaming route for axisLength > 4096 (group tables in the user
    // workspace), fp32 and fp16.
    // PATH0..5 unchanged.
    // PATH11/PATH12 are the run-time-length members of the small-shape family: one
    // instantiation serves every 1<=axis<=8 index (11) / CSR (12) case with inner==1.
    // PATH13/PATH14 are the 64-bit id routes (13 = 64-bit index, 14 = 64-bit ptr).
    ASCENDC_TPL_UINT_DECL(PATH, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    // float16/float32 only: bfloat16 cannot be compiled on this toolchain
    // (device-side bfloat16<->float32 Cast is rejected by the backend).
    ASCENDC_TPL_DATATYPE_DECL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 4),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 5),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 6),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 7),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 8),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 9),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 10),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 11),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 12),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 13),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 14),
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16),
    ),
);
