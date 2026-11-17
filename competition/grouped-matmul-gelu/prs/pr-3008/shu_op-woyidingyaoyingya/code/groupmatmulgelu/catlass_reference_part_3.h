    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        

        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.isChannelSplit = true;
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <
    class ArchTag, class TensorSrc, class TensorDst, ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    class Enable = void>
struct CopyL0CToGmSparseTla {
    
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_>
struct CopyL0CToGmSparseTla<
    Catlass::Arch::AtlasA2, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;

    struct Params {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        auto dstShape = dstTensor.shape();
        auto srcShape = srcTensor.shape();
        auto dstStride = dstTensor.stride();
        auto coord = dstTensor.coord();

        AscendC::FixpipeParamsV220 params;
        params.nSize = tla::min<int, int>(
            tla::get<1, 0>(srcShape) * tla::get<1, 1>(srcShape), tla::get<1>(dstShape) - tla::get<1>(coord));
        params.mSize = tla::min<int, int>(
            tla::get<0, 0>(srcShape) * tla::get<0, 1>(srcShape), tla::get<0>(dstShape) - tla::get<0>(coord));
        params.srcStride = tla::get<0, 0>(srcShape) * tla::get<0, 1>(srcShape);
        params.dstStride = tla::get<0>(dstStride);
        params.quantPre = quantPre;

        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto dstOffset = dstTensor.layout()(coord);
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], params);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp



// END vendored catlass/gemm/tile/copy_l0c_to_gm.hpp


// BEGIN vendored catlass/gemm/tile/copy_l0c_to_ub.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */




// END vendored catlass/gemm/tile/copy_l0c_to_ub.hpp


// BEGIN vendored catlass/gemm/tile/copy_l1_to_bt.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/arch/arch.hpp
// already embedded: catlass/layout/layout.hpp
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile { using namespace tla; }

namespace Catlass::Gemm::Tile {

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToBT {
    
};

template <class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToBT<
    ArchTag, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C2 = BYTE_PER_C2 / sizeof(ElementSrc);

    CATLASS_DEVICE
    CopyL1ToBT()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> dstTensor, AscendC::LocalTensor<ElementSrc> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = (layoutDst.shape(0) + ELE_NUM_PER_C2 - 1) / ELE_NUM_PER_C2;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp



// END vendored catlass/gemm/tile/copy_l1_to_bt.hpp


// BEGIN vendored catlass/gemm/tile/copy_l1_to_fp.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/arch/arch.hpp
// already embedded: catlass/layout/layout.hpp
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile { using namespace tla; }

namespace Catlass::Gemm::Tile {

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToFP {
    
};

template <class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToFP<
    ArchTag, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_FP = BYTE_PER_BLK_FP / sizeof(ElementSrc);

    CATLASS_DEVICE
    CopyL1ToFP()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> dstTensor, AscendC::LocalTensor<ElementSrc> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = (layoutDst.shape(0) + ELE_NUM_PER_FP - 1) / ELE_NUM_PER_FP;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/copy_l1_to_fp.hpp


// BEGIN vendored catlass/gemm/tile/copy_l1_to_l0a.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/numeric_size.hpp
// already embedded: catlass/arch/arch.hpp
// already embedded: catlass/layout/layout.hpp
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: catlass/gemm/tile/tile_copy_tla.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile {

// Keep constexpr objects used as reference non-type template arguments at namespace
// scope. GCC 13 ICEs when such an argument is defined in class-template scope.
#ifndef CATLASS_ATLASA2_LOAD3D_CONFIG_DEFINED
#define CATLASS_ATLASA2_LOAD3D_CONFIG_DEFINED
inline constexpr AscendC::IsResetLoad3dConfig CONV3D_LOAD3DV2_DEFAULT_CONFIG = {true, true};
inline constexpr AscendC::IsResetLoad3dConfig LOAD3DV2_DEFAULT_CONFIG = {false, false};
#endif

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
    
};

////////////////////////////////
/// new add gemm
template <class ArchTag, class Element>
struct CopyL1ToL0A<
    ArchTag, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<
    ArchTag, Catlass::Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)));
        ;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0A<
    ArchTag, Catlass::Gemm::GemmType<float, layout::nN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<float, layout::zZ, AscendC::TPosition::A2>> {
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0)));
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;
        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1) * 2], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0A<
    ArchTag, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>> {
    using Element = int8_t;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};



//////////////////////////////////////////

/// Partial specialization for zN in and zZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for float, zN in and zZ out.
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<float, layout::zN, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);
        AscendC::SetFmatrix(1, l1M, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0K;
        loadDataParams.mExtension = l0M;
        loadDataParams.channelSize = l1K;

        AscendC::LoadData<Element, LOAD3DV2_DEFAULT_CONFIG>(dstTensor, srcTensor, loadDataParams);
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for int8_t, nZ in and zZ out. (Transpose A)
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>> {
    using Element = int8_t;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for float, nZ in and zZ out. (Transpose A)
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<float, layout::nZ, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        constexpr uint8_t PAD_LIST[] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);
        // K, M need to be 16 aligned for f32
        uint16_t l1MAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1M);
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l0MAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0M);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0MAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.enTranspose = true;
        loadDataParams.channelSize = l1MAlign;

        AscendC::LoadData<Element, LOAD3DV2_DEFAULT_CONFIG>(dstTensor, srcTensor, loadDataParams);
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////

/// Partial specialization for CopyL1ToL0A, AtlasA2, zN in and zZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow =
            CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol =
            CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(
                dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, float, zN in and zZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<float>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<float>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszN<float, LayoutSrc>::value && tla::detail::iszZ<float, LayoutDst>::value>> {
    using Element = float;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        uint16_t l1K = RoundUp<ELE_NUM_PER_C0>(tla::get<1>(srcTensor.originShape()));
        uint16_t l0M = RoundUp<ELE_NUM_PER_C0>(tla::get<0>(dstTensor.originShape()));
        uint16_t l0K = RoundUp<ELE_NUM_PER_C0>(tla::get<1>(dstTensor.originShape()));
        AscendC::SetFmatrix(1, l1M, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0K;
        loadDataParams.mExtension = l0M;
        loadDataParams.channelSize = l1K;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData<Element, LOAD3DV2_DEFAULT_CONFIG>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, nZ in and zZ out. (Transpose A)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::iszZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow =
            CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol =
            CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(
                dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, int8_t, nZ in and zZ out. (Transpose A)
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::isnZ<int8_t, LayoutSrc>::value && tla::detail::iszZ<int8_t, LayoutDst>::value>> {
    using Element = int8_t;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterShapeRow =
            CeilDiv(tla::get<0>(srcTensor.originShape()), tla::get<0, 0>(srcTensor.shape()));
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeCol =
            CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = dstOuterShapeCol - 1;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < srcOuterShapeRow; i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor.data()[dstOffset + i * dstOuterStrideRow * 2],
                srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, float, nZ in and zZ out. (Transpose A)
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<float>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<float>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::isnZ<float, LayoutSrc>::value && tla::detail::iszZ<float, LayoutDst>::value>> {
    using Element = float;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = tla::get<0>(srcTensor.originShape());
        uint16_t l1K = tla::get<0, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.shape());
        uint16_t l0M = tla::get<0>(dstTensor.originShape());
        uint16_t l0K = tla::get<1>(dstTensor.originShape());
        // K, M need to be 16 aligned for f32
        uint16_t l1MAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1M);
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l0MAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0M);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0MAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.enTranspose = true;
        loadDataParams.channelSize = l1MAlign;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData<Element, LOAD3DV2_DEFAULT_CONFIG>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, zN in and zZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopySparseTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);
    static constexpr uint16_t HW_N0 =
        16; ///< Hardware configuration parameter for N0, reserved for future hardware configuration
    static constexpr uint16_t HW_M0 = 16;                ///< Hardware configuration parameter for M0
    static constexpr uint64_t M_POS_BIT = 48;            ///< Bit position for M index
    static constexpr uint64_t K_POS_BIT = 32;            ///< Bit position for K index
    static constexpr uint64_t M_STEP_BIT = 16;           ///< Bit step for M index
    static constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0}; ///< Padding list for data alignment

    // Methods

    CATLASS_DEVICE
    TileCopySparseTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        auto srcShape = srcTensor.shape();
        auto dstShape = dstTensor.shape();
        auto coord = srcTensor.coord();
        uint16_t aL1K = tla::get<1, 0>(srcShape) * tla::get<1, 1>(srcShape);
        uint16_t aL1M = tla::get<0, 0>(srcShape) * tla::get<0, 1>(srcShape);
        uint16_t wAlign = RoundUp(aL1M, HW_M0);
        AscendC::Load3DSetFMatrixCal(1, wAlign, PAD_LIST);
        // format(M, K), K_axis is k direction, and M_axis is m direction in load3d intrin
        uint16_t madMAlign = tla::get<0, 0>(dstShape) * tla::get<0, 1>(dstShape);
        uint16_t kStep = tla::get<1, 0>(dstShape) * tla::get<1, 1>(dstShape);
        AscendC::LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = aL1K;
        loadData3DV2.extConfig = ((uint64_t)tla::get<0>(coord) << M_POS_BIT) |
                                 ((uint64_t)tla::get<1>(coord) << K_POS_BIT) | ((uint64_t)madMAlign << M_STEP_BIT) |
                                 (uint64_t)kStep;

        AscendC::LoadData(dstTensor.data(), srcTensor.data(), loadData3DV2);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp



// END vendored catlass/gemm/tile/copy_l1_to_l0a.hpp


// BEGIN vendored catlass/gemm/tile/copy_l1_to_l0b.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/numeric_size.hpp
// already embedded: catlass/arch/arch.hpp
// already embedded: catlass/layout/layout.hpp
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: catlass/gemm/tile/tile_copy_tla.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile {

// Keep constexpr objects used as reference non-type template arguments at namespace
// scope. GCC 13 ICEs when such an argument is defined in class-template scope.
#ifndef CATLASS_ATLASA2_LOAD3D_CONFIG_DEFINED
#define CATLASS_ATLASA2_LOAD3D_CONFIG_DEFINED
inline constexpr AscendC::IsResetLoad3dConfig CONV3D_LOAD3DV2_DEFAULT_CONFIG = {true, true};
inline constexpr AscendC::IsResetLoad3dConfig LOAD3DV2_DEFAULT_CONFIG = {false, false};
#endif

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B {
    
};

////////////////////////////////////////
/// new add gemm
template <class ArchTag, class Element>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0B()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) { // K N
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<float, layout::zZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<float, layout::nZ, AscendC::TPosition::B2>> {
    using Element = float;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0B()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1))) - 1;
        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) { // K N
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0B()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor, LayoutDst layoutDst,
        LayoutSrc layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1) * 2], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};
/////////////////////////////////////////////

////////////////////////////////////////////
/// new add gemv
template <class ArchTag, class Element>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(3); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(3)], srcTensor[i * layoutSrc.stride(3)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = layoutDst.shape(1) * layoutDst.shape(3);
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    };
};

template <class ArchTag>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<float, layout::nN, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<float, layout::zN, AscendC::TPosition::B2>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nN;
    using Element = float;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)) - 1;

        for (uint32_t i = 0; i < CeilDiv<2 * ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3) * 2], srcTensor[i * layoutSrc.stride(3)], loadDataParams);
        }
    };
};

template <class ArchTag>
struct CopyL1ToL0B<
    ArchTag, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>,
    Catlass::Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B2>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nZ;
    using Element = int8_t;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3)], srcTensor[i * layoutSrc.stride(3) * 2], loadDataParams);
        }
    }
};
////////////////////////////////////////////

/// Partial specialization for int8_t, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1) * 2], loadDataParams);
        }
    }
};

/// Partial specialization for float, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<float, layout::zN, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1K = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1N = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0K = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0N = layoutDst.shape(2) * layoutDst.shape(3);
        // K, N need to be 16 aligned for f32
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l1NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1N);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        uint16_t l0NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0N);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_RIGHT);
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0NAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.channelSize = l1NAlign;
        loadDataParams.fMatrixCtrl = true;

        AscendC::LoadData<Element, LOAD3DV2_DEFAULT_CONFIG>(dstTensor, srcTensor, loadDataParams);
    }
};

/// Partial specialization for zN in and nZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<AscendC::int4b_t, layout::zN, AscendC::TPosition::A1>> {
    using Element = AscendC::int4b_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 4;
        loadDataParams.dstGap = 3;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1) * 4], loadDataParams);
        }
    }
};

/// Partial specialization for nZ in and nZ out. (Transpose B)
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        if (layoutSrc.shape(3) == layoutDst.shape(3)) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
                AscendC::LoadData(
                    dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyL1ToL0B, AtlasA2, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow =
            CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol =
            CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(
                dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow =
            CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol =
            CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData2DParams loadDataParams;
        if (srcOuterStrideRow == dstOuterStrideRow) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = dstOuterShapeRow * dstOuterShapeCol;
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = dstOuterShapeCol;
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
                AscendC::LoadData(
                    dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                    srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
            }
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, int8_t, zN in and nZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<int8_t>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::iszN<int8_t, LayoutSrc>::value && tla::detail::isnZ<int8_t, LayoutDst>::value>> {
    using Element = int8_t;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterShapeCol =
            CeilDiv(tla::get<1>(srcTensor.originShape()), tla::get<1, 0>(srcTensor.shape()));
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow =
            CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = srcOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadDataWithTranspose(
                dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                srcTensor.data()[srcOffset + i * srcOuterStrideRow * 2], loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, float, zN in and nZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<float>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<float>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::iszN<float, LayoutSrc>::value && tla::detail::isnZ<float, LayoutDst>::value>> {
    using Element = float;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1K = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        uint16_t l1N = tla::get<1>(srcTensor.originShape());
        uint16_t l0K = tla::get<0>(dstTensor.originShape());
        uint16_t l0N = tla::get<1>(dstTensor.originShape());
        // K, N need to be 16 aligned for f32
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l1NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1N);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        uint16_t l0NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0N);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_RIGHT);
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0NAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.channelSize = l1NAlign;
        loadDataParams.fMatrixCtrl = true;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData<Element, LOAD3DV2_DEFAULT_CONFIG>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, zN in and nZ out.
template <
    class ElementA, class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc,
    class CoordDst, class ElementIdx, class LayoutIdx, class CoordIdx>
struct CopyL1ToL0BSparseTla<
    Arch::AtlasA2, ElementA, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    tla::Tensor<AscendC::LocalTensor<ElementIdx>, LayoutIdx, CoordIdx, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value &&
        tla::detail::iszN<ElementIdx, LayoutIdx>::value>> {
    static constexpr uint8_t INDEX_SHIFT = 2; ///< Shift value for index manipulation

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0BSparseTla() {};

    template <class TensorDst, class TensorSrc, class TensorIdx>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, TensorIdx const& idxTensor)
    {
        

        auto dstShape = dstTensor.shape();
        auto dstStride = dstTensor.stride();
        auto srcShape = srcTensor.shape();
        auto srcStride = srcTensor.stride();
        auto coord = srcTensor.coord();

        // SET LOAD2D parameters , loop axis: K or M, or 1
        uint16_t madK = tla::get<1, 0>(dstShape) * tla::get<1, 1>(dstShape);

        // k is c0Size_ aligned for f32
        uint16_t kC0 = CeilDiv(madK, tla::get<1, 0>(dstShape));
        uint16_t nFraC0 = tla::get<0, 1>(dstShape);
        uint16_t l0bLoop = 1;
        uint64_t l0bDstAddrStride = 0;
        uint64_t l0bSrcAddrStride = 0;
        uint8_t l0bRepeat = kC0 * nFraC0;
        uint16_t l0bDststride = 0;
        uint16_t l0bSrcstride = 1;

        uint16_t bL1N = tla::get<0, 0>(srcShape) * tla::get<0, 1>(srcShape);
        if (nFraC0 * tla::get<0, 0>(dstShape) == bL1N) { // loop=1
            l0bLoop = 1;
        } else if (nFraC0 >= kC0) { // LOOP is K and repeat is n axis
            l0bLoop = kC0;
            l0bDstAddrStride = nFraC0 * tla::get<1, 1>(dstStride);
            l0bSrcAddrStride = bL1N * tla::get<1, 0>(srcShape);
            l0bRepeat = nFraC0;

            l0bDststride = 0;
            l0bSrcstride = 1;
        } else { // LOOP is N  and repeat is K axis
            l0bLoop = nFraC0;
            l0bDstAddrStride = tla::get<1, 1>(dstStride);
            l0bSrcAddrStride = tla::get<0, 1>(srcStride);
            l0bRepeat = kC0;

            l0bDststride = nFraC0 - 1;
            l0bSrcstride = bL1N;
        }
        // use load2d for L1_2_L0B
        // startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrmode
        AscendC::LoadData2dParams loadDataParams{0, l0bRepeat, l0bSrcstride, 0, l0bDststride, 0, 0};
        uint16_t bL1KOffset = tla::get<1>(coord);
        uint64_t l1bOffset = tla::get<0>(coord) * tla::get<1, 0>(srcShape) + bL1KOffset * bL1N;
        uint64_t l0bOffset = 0;
        AscendC::LocalTensor<ElementA> dstLocal;
        AscendC::LocalTensor<ElementA> srcLocal;
        AscendC::LocalTensor<uint8_t> l1BIndex;
        for (uint64_t i = 0; i < l0bLoop; i++) {
            dstLocal.SetAddr(dstTensor.data()[l0bOffset].address_);
            srcLocal.SetAddr(srcTensor.data()[l1bOffset].address_);
            l1BIndex.SetAddr(idxTensor.data()[l1bOffset >> INDEX_SHIFT].address_);
            AscendC::LoadDataWithSparse(dstLocal, srcLocal, l1BIndex, loadDataParams);
            l1bOffset += l0bSrcAddrStride;
            l0bOffset += l0bDstAddrStride;
        }
    }
};

// Partial specialization for CopyL1ToL0B, AtlasA2, nZ in and nZ out. (Transpose B)
template <
    class ElementA, class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc,
    class CoordDst, class ElementIdx, class LayoutIdx, class CoordIdx>
struct CopyL1ToL0BSparseTla<
    Arch::AtlasA2, ElementA, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    tla::Tensor<AscendC::LocalTensor<ElementIdx>, LayoutIdx, CoordIdx, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value &&
        tla::detail::isnZ<ElementIdx, LayoutIdx>::value>> {
    static constexpr uint8_t INDEX_SHIFT = 2; ///< Shift value for index manipulation

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0BSparseTla() {};

    template <class TensorDst, class TensorSrc, class TensorIdx>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, TensorIdx const& idxTensor)
    {
        

        auto dstShape = dstTensor.shape();
        auto dstStride = dstTensor.stride();
        auto srcShape = srcTensor.shape();
        auto srcStride = srcTensor.stride();
        auto coord = srcTensor.coord();

        // SET LOAD2D parameters , loop axis: K or M, or 1
        uint16_t madK = tla::get<0, 0>(dstShape) * tla::get<0, 1>(dstShape);

        // k is c0Size_ aligned for f32
        uint16_t kC0 = CeilDiv(madK, tla::get<0, 0>(dstShape));
        uint16_t nFraC0 = tla::get<1, 1>(dstShape);
        uint16_t l0bLoop = 1;
        uint64_t l0bDstAddrStride = 0;
        uint64_t l0bSrcAddrStride = 0;
        uint8_t l0bRepeat = kC0 * nFraC0;

        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(l0bRepeat);
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;
        uint16_t bL1N = tla::get<1, 0>(srcShape) * tla::get<1, 1>(srcShape);
        uint16_t bL1KOffset = tla::get<0>(coord);
        uint64_t l1bOffset = tla::get<1>(coord) * tla::get<0, 0>(srcShape) + bL1KOffset * bL1N;

        AscendC::LocalTensor<ElementA> dstLocal;
        AscendC::LocalTensor<ElementA> srcLocal;
        AscendC::LocalTensor<uint8_t> l1BIndex;
        dstLocal.SetAddr(dstTensor.data().address_);
        srcLocal.SetAddr(srcTensor.data()[l1bOffset].address_);
        l1BIndex.SetAddr(idxTensor.data()[l1bOffset >> INDEX_SHIFT].address_);

        AscendC::LoadDataWithSparse(dstLocal, srcLocal, l1BIndex, loadDataParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp



// END vendored catlass/gemm/tile/copy_l1_to_l0b.hpp


// BEGIN vendored catlass/gemm/tile/copy_ub_to_gm.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/arch/arch.hpp
// already embedded: catlass/gemm/tile/tile_copy_tla.hpp
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile {

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint16_t row = tla::get<0>(dstTensor.originShape());
        const uint16_t col = tla::get<1>(dstTensor.originShape());

        AscendC::DataCopyExtParams dataCopyParams(
            row, col * sizeof(ElementSrc), (tla::get<0>(srcTensor.stride()) - col) / ELE_NUM_PER_C0,
            (tla::get<0>(dstTensor.stride()) - col) * sizeof(ElementSrc), 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    };
};

/// Partial specialization for AtlasA2, RowMajor in and PaddingRowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>, layout::RowMajor,
    layout::PaddingRowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<1, 1>(dstTensor.shape()), tla::get<1, 0>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<1, 1>(dstTensor.stride()) - tla::get<1, 0>(dstTensor.shape())) * sizeof(ElementSrc), 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    };
};

template <class ArchTag, class GmType>
struct CopyUb2Gm {
    
};

template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0), layoutDst.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
            (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element), 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp



// END vendored catlass/gemm/tile/copy_ub_to_gm.hpp

// already embedded: catlass/gemm/tile/tile_copy_tla.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

template <
    class ArchTag, class AType, class BType, class CType, class PrologueA_, class PrologueB_, class BiasType = void>
struct TileCopyWithPrologueDeqPerTensor {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType, Tile::ScaleGranularity::PER_TENSOR>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

template <
    class ArchTag, class AType, class BType, class CType, class PrologueA_, class PrologueB_, class BiasType = void>
struct TileCopyWithPrologue {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

///////////////////////////////////
/// new add
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmTpe type for Bias operand
    class BiasType = void>
struct TileCopyGemm {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    // change structual
    using L1AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1AType;
    using L1BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1BType;
    using L0AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0AType;
    using L0BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0BType;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType, L1BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1AType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1BType, L0BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};
//////////////////////////////


// fixpipe开启relu开关
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct ReluTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType> {
    // 重写 CopyL0CToGm
    using ElementAccumulator = typename TileCopy<ArchTag, AType, BType, CType, BiasType>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<
        ArchTag, ElementAccumulator, CType, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, true>;
};

// fixpipe开启随路量化
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void,
    /// GemmType type for Bias operand
    ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR>
struct QuantTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType> {
    // 重写 CopyL0CToGm
    using ElementAccumulator = typename TileCopy<ArchTag, AType, BType, CType, BiasType>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;

    using CopyGmToL1Scale = Gemm::Tile::CopyGmToL1<
        ArchTag, Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::GM>,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>>;

    using CopyL1ToFP = Gemm::Tile::CopyL1ToFP<
        ArchTag, Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>>;
};

// sparse Tile
template <
    /// Tag indicating architecture
    class ArchTag, class ElementA_, class LayoutTagA, class ElementB_, class LayoutTagB, class ElementC_,
    class LayoutTagC>
struct SparseTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = layout::zZ;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL1BIdx = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL1BIdx = detail::TagToLayout_t<int32_t, LayoutTagL1BIdx>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1BIdx =
        tla::Tensor<AscendC::LocalTensor<int32_t>, LayoutL1BIdx, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopySparseTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopySparseTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorIdx>
    using CopyGmToL1BIdx = Gemm::Tile::TileCopySparseTla<ArchTag, TensorIdx, TensorL1BIdx>;

    using CopyL1ToL0A = Gemm::Tile::TileCopySparseTla<ArchTag, TensorL1A, TensorL0A>;

    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0BSparseTla<ArchTag, ElementA, TensorL1B, TensorL0B, TensorL1BIdx>;

    template <class TensorC>
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmSparseTla<ArchTag, TensorL0C, TensorC>;
};


template <
    /// Tag indicating architecture
    class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_, class ElementC_,
    class LayoutTagC_, class ElementBias = void, bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY_ = ScaleGranularity::NO_QUANT, class L0CCopyMode = CopyToGM>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    static constexpr bool ReluEnable = ReluEnable_;
    static constexpr ScaleGranularity DEQUANT_GRANULARITY = DEQUANT_GRANULARITY_;

    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;
    static constexpr bool HAS_QUANT_TENSOR = (DEQUANT_GRANULARITY == ScaleGranularity::PER_CHANNEL);

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = typename helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL0C = layout::L0C;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1AVectorLayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0>, AscendC::TPosition::A1>;
    using TensorL1ALayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    using TensorL1A =
        std::conditional_t<tla::detail::isVector<LayoutTagA>::value, TensorL1AVectorLayout, TensorL1ALayout>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;
    using TensorL1Bias = std::conditional_t<
        HAS_BIAS,
        tla::Tensor<
            AscendC::LocalTensor<ElementBias>, detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
            tla::Coord<tla::_0>, AscendC::TPosition::A1>,
        EmptyClass>;
    using TensorL0Bias = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>, detail::TagToLayout_t<ElementAccumulator, layout::VectorLayout>,
        tla::Coord<tla::_0>, AscendC::TPosition::C2>;
    using TensorL1Quant = std::conditional_t<
        HAS_QUANT_TENSOR,
        tla::Tensor<
            AscendC::LocalTensor<uint64_t>, detail::TagToLayout_t<uint64_t, layout::VectorLayout>, tla::Coord<tla::_0>,
            AscendC::TPosition::A1>,
        EmptyClass>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorBias>
    using CopyGmToL1Bias =
        std::conditional_t<HAS_BIAS, Gemm::Tile::TileCopyTla<ArchTag, TensorBias, TensorL1Bias>, EmptyClass>;

    template <class TensorQuant>
    using CopyGmToL1Scale =
        std::conditional_t<HAS_QUANT_TENSOR, Gemm::Tile::TileCopyTla<ArchTag, TensorQuant, TensorL1Quant>, EmptyClass>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL1ToBT =
        std::conditional_t<HAS_BIAS, Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>, EmptyClass>;

    template <class TensorC>
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
};

template <
    /// Tag indicating architecture
    class ArchTag, class TensorA, class LayoutTagA, class TensorB, class LayoutTagB, class TensorC, class LayoutTagC,
    class TensorBias = void, class LayoutTagBias = void, bool IS_PADDING_A = false, bool IS_PADDING_B = false>
struct PaddingPackedTileCopyTla {
    
    
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = layout::zZ;
    using LayoutTagL0B = layout::nZ;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using LayoutPaddingTagA = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::RowMajor>, layout::PaddingRowMajor, layout::PaddingColumnMajor>;
    using LayoutPaddingTagB = std::conditional_t<
        std::is_same_v<LayoutTagB, layout::RowMajor>, layout::PaddingRowMajor, layout::PaddingColumnMajor>;

    using CopyGmToL1A = std::conditional_t<
        IS_PADDING_A, Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutPaddingTagA, LayoutTagL1A>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutTagA, LayoutTagL1A>>;
    using CopyGmToL1B = std::conditional_t<
        IS_PADDING_B, Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutPaddingTagB, LayoutTagL1B>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutTagB, LayoutTagL1B>>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
};


} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/tile_copy.hpp


// BEGIN vendored catlass/gemm/tile/tile_mmad.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/gemm/helper.hpp
// already embedded: tla/tensor.hpp
namespace Catlass::Gemm::Tile {

///////////////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // Methods

    CATLASS_DEVICE
    TileMmad()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementAccumulator> const& l0CTensor, AscendC::LocalTensor<ElementA> const& l0ATensor,
        AscendC::LocalTensor<ElementB> const& l0BTensor, uint32_t m, uint32_t n, uint32_t k, bool initC = true,
        uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (
            std::is_same_v<ElementA, float> && (std::is_same_v<typename AType_::Layout, layout::ColumnMajor> ||
                                                std::is_same_v<typename AType_::Layout, layout::nZ>)) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr (std::is_same_v<typename AType_::Layout, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif

        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementAccumulator> const& l0CTensor, AscendC::LocalTensor<ElementA> const& l0ATensor,
        AscendC::LocalTensor<ElementB> const& l0BTensor, AscendC::LocalTensor<ElementAccumulator> const& l0BiasTensor,
        uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        using LayoutL1A = typename helper::L1ATypeSelector<AType_>::L1AType::Layout;
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif

        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

///////////////////////////////////////////TileMmadTla/////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag,
    /// Element for A matrix operand
    class ElementA,
    /// LayoutTag for A matrix operand in L1
    class LayoutTagL1A>
struct TileMmadTla {
    // Methods

    CATLASS_DEVICE
    TileMmadTla()
    {}

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(
        TensorC const& l0CTensor, TensorA const& l0ATensor, TensorB const& l0BTensor, uint32_t m, uint32_t n,
        uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr (std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB, class TensorBias>
    CATLASS_DEVICE void operator()(
        TensorC const& l0CTensor, TensorA const& l0ATensor, TensorB const& l0BTensor, TensorBias const& l0BiasTensor,
        uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), l0BiasTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(
        TensorC const& l0CTensor, TensorA const& l0ATensor, TensorB const& l0BTensor, uint32_t m, uint32_t n,
        uint32_t k, uint32_t l0Batch)
    {
        const uint32_t L0AM = tla::get<0, 0>(l0ATensor.shape()) * tla::get<0, 1>(l0ATensor.shape());
        const uint32_t L0AK = tla::get<1, 0>(l0ATensor.shape()) * tla::get<1, 1>(l0ATensor.shape());
        const uint32_t L0BK = tla::get<0, 0>(l0BTensor.shape()) * tla::get<0, 1>(l0BTensor.shape());
        const uint32_t L0BN = tla::get<1, 0>(l0BTensor.shape()) * tla::get<1, 1>(l0BTensor.shape());
        const uint32_t L0CM = tla::get<0, 0>(l0CTensor.shape()) * tla::get<0, 1>(l0CTensor.shape());
        const uint32_t L0CN = tla::get<1, 0>(l0CTensor.shape()) * tla::get<1, 1>(l0CTensor.shape());

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif
        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendC::Mmad(
                l0CTensor.data()[l0BatchIdx * L0CM * L0CN], l0ATensor.data()[l0BatchIdx * L0AM * L0AK],
                l0BTensor.data()[l0BatchIdx * L0BK * L0BN], mmadParams);
        }
    }

    // automatically extracts actual sizes from tensor originShape
    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(
        TensorC const& l0CTensor, TensorA const& l0ATensor, TensorB const& l0BTensor, bool initC = true,
        uint8_t unitFlag = 0)
    {
        // Get actual sizes from tensor originShape
        uint32_t m = tla::get<0>(l0CTensor.layout().originShape());
        uint32_t n = tla::get<1>(l0CTensor.layout().originShape());
        uint32_t k = tla::get<1>(l0ATensor.layout().originShape());
        AscendC::MmadParams mmadParams;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
        if constexpr (!std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            if (m == 1)
                m = 16; // avoid gemv mode
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr (std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/tile_mmad.hpp

namespace Catlass::Gemm::Block {
template<class P,class L1,class L0,class A,class B,class C,class Bias=void,
 class Copy=Gemm::Tile::TileCopy<typename P::ArchTag,A,B,C,Bias>,
 class Mmad=Gemm::Tile::TileMmad<typename P::ArchTag,A,B,Bias>>
struct BlockMmad;
}

// BEGIN vendored catlass/gemm/block/block_mmad_preload_async_with_callback.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/arch/resource.hpp
// already embedded: catlass/coord.hpp

// BEGIN vendored catlass/detail/callback.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/detail/macros.hpp

/// @brief Callback is an alternative to std::function<void(void)>, providing a general carrier
/// of callable structure with no parameters and no return value. Compared with function pointers
/// of type void (*)(), Callback can carry lambda expressions with captures, and does not need to
/// pay attention to the captured content. It should be noted that Callback itself does not store
/// the callable structure it carries like std::function<void(void)>, so it is necessary to ensure
/// that it is used within the life cycle of the callable structure.
struct Callback {
    void const* func{nullptr};
    void (*caller)(void const*){nullptr};

    Callback() = default;

    CATLASS_DEVICE
    void operator()() const
    {
        if (func) {
            caller(func);
        }
    }

    CATLASS_DEVICE
    operator bool() const
    {
        return func != nullptr;
    }
};

template <typename Func>
CATLASS_DEVICE void FuncWrapper(void const* func)
{
    (*static_cast<Func const*>(func))();
}

// Use this to make a callback
template <typename Func>
CATLASS_DEVICE Callback MakeCallback(Func* func)
{
    Callback callback;
    callback.func = func;
    callback.caller = &FuncWrapper<Func>;
    return callback;
}


// END vendored catlass/detail/callback.hpp

// already embedded: catlass/gemm_coord.hpp
// already embedded: catlass/gemm/dispatch_policy.hpp
// already embedded: catlass/gemm/helper.hpp
// already embedded: catlass/gemm/tile/tile_copy.hpp
// already embedded: catlass/gemm/tile/tile_mmad.hpp

namespace Catlass::Gemm::Block {

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_, class L1TileShape_, class L0TileShape_, class AType_, class BType_,
    class CType_, class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<
    MmadAtlasA2PreloadAsyncWithCallback<
        PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2PreloadAsyncWithCallback<
        PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);

    // Check LayoutC
    

    // Check L1TileShape
    

    // Check L0TileShape
    
    
    

    
    
    // 32B (256b) aligned
    

    static constexpr auto L1A_LAYOUT = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
    static constexpr auto L1B_LAYOUT = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        InitL1(resource, l1BufAddrStart);
        InitL0A(resource);
        InitL0B(resource);
        InitL0C(resource);
    }

    CATLASS_DEVICE
    ~BlockMmad()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
        AscendC::GlobalTensor<ElementB> const& gmBlockB, LayoutB const& layoutB,
        AscendC::GlobalTensor<ElementC> const& gmBlockC, LayoutC const& layoutC, GemmCoord const& actualShape,
        Callback const& callbackBeforeFixpipe, Callback const& callbackAfterFixpipe)
    {
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t kTileIdx = (startTileIdx + kLoopIdx < kTileCount) ? (startTileIdx + kLoopIdx) :
                                                                         (startTileIdx + kLoopIdx - kTileCount);

            uint32_t kActual =
                (kTileIdx < kTileCount - 1) ? L1TileShape::K : (actualShape.k() - kTileIdx * L1TileShape::K);

            // Emission load instruction from GM to L1
            MatrixCoord gmTileAOffset{0, kTileIdx * L1TileShape::K};
            MatrixCoord gmTileBOffset{kTileIdx * L1TileShape::K, 0};
            auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
            auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
            // Load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmTileA, L1A_LAYOUT, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            // Load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
            copyGmToL1B(l1BTensorList[l1ListId], gmTileB, L1B_LAYOUT, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            // If the number of preload instructions reaches the upper limit, perform an mmad calculation on L1 tile
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }

            // Store the current load status
            uint32_t preloadL1TileMmadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES) ?
                                                     (l1TileMmadParamsId + preloadCount) :
                                                     (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto& l1TileMmadParams = l1TileMmadParamsList[preloadL1TileMmadParamsId];
            l1TileMmadParams.l1ListId = l1ListId;
            l1TileMmadParams.mRound = mRound;
            l1TileMmadParams.nRound = nRound;
            l1TileMmadParams.kActual = kActual;
            l1TileMmadParams.isKLoopFirst = (kLoopIdx == 0);
            l1TileMmadParams.isKLoopLast = (kLoopIdx == kTileCount - 1);
            if (kLoopIdx == kTileCount - 1) {
                l1TileMmadParams.gmBlockC = gmBlockC;
                l1TileMmadParams.layoutCInGm = layoutC.GetTileLayout(actualShape.GetCoordMN());
                l1TileMmadParams.callbackBeforeFixpipe = callbackBeforeFixpipe;
                l1TileMmadParams.callbackAfterFixpipe = callbackAfterFixpipe;
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
        }
    }

    CATLASS_DEVICE
    void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

private:
    struct L1TileMmadParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        bool isKLoopFirst;
        bool isKLoopLast;
        AscendC::GlobalTensor<ElementC> gmBlockC;
        LayoutC layoutCInGm;
        Callback callbackBeforeFixpipe;
        Callback callbackAfterFixpipe;

        CATLASS_DEVICE
        L1TileMmadParams() = default;
    };

    CATLASS_DEVICE
    void InitL1(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
    }

    CATLASS_DEVICE
    void InitL0A(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
    }

    CATLASS_DEVICE
    void InitL0B(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    CATLASS_DEVICE
    void InitL0C(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    CATLASS_DEVICE
    void L1TileMmad(L1TileMmadParams const& params)
    {
        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(params.mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(params.nRound);
        uint32_t kPartLoop = CeilDiv<L0TileShape::K>(params.kActual);
        auto& l1ATensor = l1ATensorList[params.l1ListId];
        auto& l1BTensor = l1BTensorList[params.l1ListId];

        auto& l0CTensor = l0CTensorList[l0CListId];
        LayoutCInL0 layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(params.mRound, params.nRound));

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual =
                (mPartIdx < mPartLoop - 1) ? L0TileShape::M : (params.mRound - mPartIdx * L0TileShape::M);

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPartActual =
                    (kPartIdx < kPartLoop - 1) ? L0TileShape::K : (params.kActual - kPartIdx * L0TileShape::K);

                auto& l0ATile = l0ATensorList[l0AListId];
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                auto l1AOffset = MakeCoord(mPartIdx, kPartIdx) * L0TileShape::ToCoordMK();
                auto l1ATile = l1ATensor[L1A_LAYOUT.GetOffset(l1AOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1ListId]);
                }
                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, L1A_LAYOUT);
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1ListId]);
                }

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                    uint32_t nPartActual =
                        (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (params.nRound - nPartIdx * L0TileShape::N);

                    auto& l0BTile = l0BTensorList[l0BListId];
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                    auto l1BOffset = MakeCoord(kPartIdx, nPartIdx) * L0TileShape::ToCoordKN();
                    auto l1BTile = l1BTensor[L1B_LAYOUT.GetOffset(l1BOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1ListId]);
                    }
                    copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, L1B_LAYOUT);
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1ListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    auto l0COffset = MakeCoord(mPartIdx, nPartIdx) * L0TileShape::ToCoordMN();
                    auto l0CTile = l0CTensor[layoutCInL0.GetOffset(l0COffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (params.isKLoopFirst && (kPartIdx == 0));
                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (params.isKLoopLast && (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) &&
                            (nPartIdx == nPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (params.isKLoopLast) {
            auto layoutCInGm = params.layoutCInGm;

            params.callbackBeforeFixpipe();

            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToGm(params.gmBlockC, l0CTensor, layoutCInGm, layoutCInL0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                copyL0CToGm(params.gmBlockC, l0CTensor, layoutCInGm, layoutCInL0, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;

            params.callbackAfterFixpipe();
        }
    }

    AscendC::LocalTensor<ElementA> l1ATensorList[L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1_STAGES];
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catlass::Gemm::Block


// END vendored catlass/gemm/block/block_mmad_preload_async_with_callback.hpp
