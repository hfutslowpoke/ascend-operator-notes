template <>
struct element_alignment<void> {
    static constexpr auto value = _1{};
};

template <class Element>
inline constexpr auto element_alignment_v = element_alignment<Element>::value;

template <class LayoutTag>
struct FractalSpec {};

template <>
struct FractalSpec<Catlass::layout::zN> {
    using Order = tla::Step<tla::Step<tla::Int<1>, tla::Int<2>>, tla::Step<tla::Int<0>, tla::Int<3>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::C0_NUM_PER_FRACTAL>, tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>>{};
};

template <>
struct FractalSpec<Catlass::layout::nZ> {
    using Order = tla::Step<tla::Step<tla::Int<0>, tla::Int<3>>, tla::Step<tla::Int<1>, tla::Int<2>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>, tla::Int<Catlass::C0_NUM_PER_FRACTAL>>{};
};

template <>
struct FractalSpec<Catlass::layout::zZ> {
    using Order = tla::Step<tla::Step<tla::Int<1>, tla::Int<3>>, tla::Step<tla::Int<0>, tla::Int<2>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::C0_NUM_PER_FRACTAL>, tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>>{};
};

template <>
struct FractalSpec<Catlass::layout::nN> {
    using Order = tla::Step<tla::Step<tla::Int<0>, tla::Int<2>>, tla::Step<tla::Int<1>, tla::Int<3>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>, tla::Int<Catlass::C0_NUM_PER_FRACTAL>>{};
};

template <>
struct FractalSpec<Catlass::layout::L0C> {
    using Order = tla::Step<tla::Step<tla::Int<1>, tla::Int<2>>, tla::Step<tla::Int<0>, tla::Int<3>>>;
    template <class Element>
    static constexpr auto fractal =
        tla::Shape<tla::Int<Catlass::C0_NUM_PER_FRACTAL>, tla::Int<Catlass::C0_NUM_PER_FRACTAL>>{};
};

template <class LayoutTag, class = void>
struct is_fractal_layout : false_type {};

template <class LayoutTag>
struct is_fractal_layout<LayoutTag, std::void_t<typename FractalSpec<LayoutTag>::Order>> : true_type {};

} // end namespace detail

template <class Shape, class Fractal>
CATLASS_HOST_DEVICE constexpr auto make_fractal_shape(Shape const& shape, Fractal const& fractal)
{
    return detail::transform_impl(
        [](auto const& s, auto const& f) { return make_tuple(f, ceil_div(s, f)); }, shape, fractal);
}

template <class LayoutTag, class Element = void, class Shape>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape)
{
    
    if constexpr (is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        
        return make_layout(wrap(shape), make_stride(Int<1>{}));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        
        const auto stride = round_up(compact_row_major(shape), make_stride(detail::element_alignment_v<Element>, _1{}));
        return make_layout(shape, stride);
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        
        const auto stride = round_up(compact_col_major(shape), make_stride(_1{}, detail::element_alignment_v<Element>));
        return make_layout(shape, stride);
    } else if constexpr (detail::is_fractal_layout<LayoutTag>::value) {
        
        const auto fractalShape = make_fractal_shape(shape, detail::FractalSpec<LayoutTag>::template fractal<Element>);
        return make_ordered_layout(fractalShape, typename detail::FractalSpec<LayoutTag>::Order{}, shape);
    } else {
        
    }
}

template <class LayoutTag, class Element = void, class InnerShape, class OuterShape>
CATLASS_HOST_DEVICE constexpr auto make_layout(InnerShape const& innerShape, OuterShape const& outerShape)
{
    
    const auto innerLayout = make_layout<LayoutTag, Element>(innerShape);
    const auto outerLayout = make_layout(outerShape, compact_col_major(outerShape, cosize(innerLayout)));
    return make_layout(
        tuple_cat(innerLayout.shape(), wrap(outerLayout.shape())),
        tuple_cat(innerLayout.stride(), wrap(outerLayout.stride())),
        tuple_cat(innerLayout.originShape(), wrap(outerLayout.originShape())));
}

template <class LayoutTag>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFromTag(LayoutTag const& tag)
{
    

    if constexpr (is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        return MakeLayout(MakeShape(tag.shape(0)), MakeStride(tag.stride(0)), MakeShape(tag.shape(0)));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        return MakeLayout(
            MakeShape(tag.shape(0), tag.shape(1)), MakeStride(tag.stride(0), Int<1>{}),
            MakeShape(tag.shape(0), tag.shape(1)));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        return MakeLayout(
            MakeShape(tag.shape(0), tag.shape(1)), MakeStride(Int<1>{}, tag.stride(1)),
            MakeShape(tag.shape(0), tag.shape(1)));
    } else { // zN or nZ or L0C
        return MakeLayout(
            MakeShape(MakeShape(tag.shape(0), tag.shape(1)), MakeShape(tag.shape(2), tag.shape(3))),
            MakeStride(MakeStride(tag.stride(0), tag.stride(1)), MakeStride(tag.stride(2), tag.stride(3))),
            MakeShape(tag.orgShape(0), tag.orgShape(1)));
    }
}

template <class T>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& len)
{
    return MakeLayout(MakeShape(len), MakeStride(Int<1>{}), MakeShape(len));
}

template <class Element, class LayoutTag, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& rows, U const& cols)
{
    

    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;

    if constexpr (is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        return MakeLayout(MakeShape(cols), MakeStride(Int<1>{}), MakeShape(cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)cols, Int<1>{}), MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)rows), MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::zN>) {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0)),
            MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(
                MakeStride(
                    Int<ELE_NUM_PER_C0>{}, RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::L0C>) {
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(
                    Int<1>{},
                    RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * Catlass::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::Weight4BitnZ>) {
        constexpr uint32_t ELE_NUM_PER_C0 = 32;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 512;
        return MakeLayout(
            MakeShape(
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    }
}


// Specialized layout factories
template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutL0C(T const& rows, U const& cols)
{
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
    return MakeLayout(
        MakeShape(
            MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
            MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(
            MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
            MakeStride(
                Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * Catlass::C0_NUM_PER_FRACTAL)),
        MakeShape(rows, cols));
}





// Layout type predicates
namespace detail {

template <class Layout, class Enable = void>
struct isVector {
    static bool const value = false;
};

template <class Layout>
struct isVector<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 1>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isRowMajor {
    static bool const value = false;
};

template <class Layout>
struct isRowMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<1>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isColumnMajor {
    static bool const value = false;
};

template <class Layout>
struct isColumnMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct iszN {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszN<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<
        rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
        rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<1, 0>(Layout{}) == 1 && stride<0, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

/*
For matmul m axis is not c0 Aligned.
Exp: oriShape(m, k) : (127, 256)
zNUnAlign shape:((127, 1), (16, 256/16))  zN shape: ((16, Ceil(127/16)), (16, 256/16))
*/
template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct iszNUnAlign {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszNUnAlign<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static bool const value =
        (shape<0, 1>(Layout{}) == 1 && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<0, 0>(Layout{}) == ELE_NUM_PER_C0 && stride<1, 0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable = void>
struct iszZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<1, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isnZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct isnZ<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == ELE_NUM_PER_C0 && shape<1, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
         stride<0, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};


} // end namespace detail

// Tile layout utilities
namespace detail {

template <class OriginBase, class TileShape, class Coord, int... Is>
CATLASS_HOST_DEVICE constexpr auto CropOriginShape(
    OriginBase const& originBase, TileShape const& tileShape, Coord const& coord, seq<Is...>)
{
    return MakeShape(tla::min(
        static_cast<uint32_t>(get<Is>(tileShape)),
        (static_cast<uint32_t>(get<Is>(coord)) < static_cast<uint32_t>(get<Is>(originBase))) ?
            (static_cast<uint32_t>(get<Is>(originBase)) - static_cast<uint32_t>(get<Is>(coord))) :
            0u)...);
}

} // namespace detail

/// 创建 tile layout：使用指定的 tile 尺寸用于内存布局计算，同时携带实际逻辑尺寸（origin_shape）。
/// coord 是元素坐标，用于计算实际的 originShape（处理边界情况）。
/// Supports layouts of any rank (rank >= 1) for depth==1 layouts.
/// For depth>1 (fractal) layouts, currently only rank-2 is supported.
template <class Layout, class TileShape, class Coord>
CATLASS_HOST_DEVICE constexpr auto GetTileLayout(Layout const& layout, TileShape const& tileShape, Coord const& coord)
{
    
    

    // 统一计算 tail tile 的逻辑尺寸（originShape 裁剪）
    auto tileOriginShape = detail::CropOriginShape(layout.originShape(), tileShape, coord, tuple_seq<TileShape>{});

    // depth==1 的布局（vector/matrix/tensor）：tile shape 直接作为 memory-layout shape
    // 支持任意 rank >= 1（但必须与 layout.rank 匹配）
    if constexpr (Layout::depth == 1) {
        
        return MakeLayout(tileShape, layout.stride(), tileOriginShape);
    } else {
        // depth>1 的布局（fractal layout）：目前只支持 rank=2
        // 因为 fractal layout 通常用于矩阵（rank-2），需要把 (rows, cols) 转为同结构嵌套 shape
        

        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);

        // MakeMxScaleLayout RowMajor A 等：第一维为行长度，第二维为 (C0, ceil(cols/C0))；与 catlass_dev
        // `MakeLayoutTile` 中 rank(shape<0>)==1 && rank(shape<1>)==2 分支一致。
        if constexpr (
            Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 1 &&
            rank_v<decltype(shape<1>(Layout{}))> == 2 && is_static<decltype(shape<1, 0>(layout))>::value) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(
                MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                layout.stride(), tileOriginShape);
        }
        // MakeMxScaleLayout B 侧等：shape 为 ((C0, ceil(rows/C0)), cols)；与 catlass_dev `MakeLayoutTile` 中
        // rank(shape<0>)==2 && rank(shape<1>)==1 分支一致。
        else if constexpr (
            Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 2 &&
            rank_v<decltype(shape<1>(Layout{}))> == 1 && is_static<decltype(shape<0, 0>(layout))>::value) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<0, 0>(layout))::value;
            return MakeLayout(
                MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                layout.stride(), tileOriginShape);
        }
        // 典型 fractal（zN/nZ 等）：shape<0,0>、shape<1,0> 均为编译期常量；嵌套 ((r0,...),(c0,...))。
        else if constexpr (
            is_static<decltype(shape<0, 0>(layout))>::value && is_static<decltype(shape<1, 0>(layout))>::value) {
            constexpr uint32_t dstInnerShapeRow = decltype(shape<0, 0>(layout))::value;
            constexpr uint32_t dstInnerShapeCol = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(
                MakeShape(
                    MakeShape(Int<dstInnerShapeRow>{}, CeilDiv<dstInnerShapeRow>(rows)),
                    MakeShape(Int<dstInnerShapeCol>{}, CeilDiv<dstInnerShapeCol>(cols))),
                layout.stride(), tileOriginShape);
        }
        // 内层块尺寸非编译期常量：运行期从 layout 读取再分块。
        else {
            const uint32_t dstInnerShapeRow = shape<0, 0>(layout);
            const uint32_t dstInnerShapeCol = shape<1, 0>(layout);
            return MakeLayout(
                MakeShape(
                    MakeShape(dstInnerShapeRow, CeilDiv(rows, dstInnerShapeRow)),
                    MakeShape(dstInnerShapeCol, CeilDiv(cols, dstInnerShapeCol))),
                layout.stride(), tileOriginShape);
        }
    }
}

// Layout transforms
namespace detail {

// Prepend one leading dimension to a layout type (general form of "make batched layout").
// Preserves the *types* of each stride element from the original layout.
template <
    class Layout, class NewShapeT = uint32_t, class NewStrideT = int64_t, class NewOriginT = uint32_t,
    class Seq = tla::make_seq<Layout::rank>>
struct PrependDimLayout;

template <class Layout, class NewShapeT, class NewStrideT, class NewOriginT, int... Is>
struct PrependDimLayout<Layout, NewShapeT, NewStrideT, NewOriginT, tla::seq<Is...>> {
    using ShapeOld = tla::remove_cvref_t<decltype(std::declval<Layout const&>().shape())>;
    using StrideOld = tla::remove_cvref_t<decltype(std::declval<Layout const&>().stride())>;
    using OriginOld = tla::remove_cvref_t<decltype(std::declval<Layout const&>().originShape())>;

    using ShapeNew = tla::Shape<NewShapeT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<ShapeOld>()))>...>;
    using StrideNew =
        tla::Stride<NewStrideT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<StrideOld>()))>...>;
    using OriginNew = tla::Shape<NewOriginT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<OriginOld>()))>...>;

    using type = tla::Layout<ShapeNew, StrideNew, OriginNew>;
};

} // namespace detail

template <class Layout, class NewShapeT = uint32_t, class NewStrideT = int64_t, class NewOriginT = uint32_t>
using PrependDimLayout_t = typename detail::PrependDimLayout<Layout, NewShapeT, NewStrideT, NewOriginT>::type;

template <class Layout>
using MakeBatchedLayout_t = PrependDimLayout_t<Layout>;

} // end namespace tla


// END vendored tla/layout.hpp


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Catlass::detail {
////////////////////////////////////////////////////////////////////////////////////////////////////
// For each Catlass::layout, provides its corresponding tla layout types
template <class Element, class LayoutTag>
struct TagToLayout {
    using type = LayoutTag;
};

template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::ColumnMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<tla::Int<1>, int64_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::VectorLayout> {
    using type = tla::Layout<tla::Shape<uint32_t>, tla::Stride<tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<
            tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>, tla::Stride<tla::Int<1>, int64_t>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<
            tla::Stride<tla::Int<ELE_NUM_PER_C0>, int64_t>, tla::Stride<tla::Int<1>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <class Element>
struct TagToLayout<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
        tla::Stride<
            tla::Stride<tla::Int<1>, int64_t>, tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};






// Convenience aliases
template <class Element, class LayoutTag>
using TagToLayout_t = typename TagToLayout<Element, LayoutTag>::type;

constexpr uint32_t ELE_NUM_PER_FRACTAL_L0C = 256;
using LayoutL0C = tla::Layout<
    tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
    tla::Stride<
        tla::Stride<tla::Int<C0_NUM_PER_FRACTAL>, tla::Int<ELE_NUM_PER_FRACTAL_L0C>>,
        tla::Stride<tla::Int<1>, int64_t>>>;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::detail


// END vendored catlass/detail/tag_to_layout.hpp


// BEGIN vendored catlass/gemm/helper.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/arch/arch.hpp
// already embedded: catlass/catlass.hpp
// already embedded: catlass/numeric_size.hpp
// already embedded: catlass/layout/layout.hpp
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: tla/layout.hpp

namespace Catlass::Gemm::helper {

template <class Element, class Layout>
struct L1AlignHelper {
    
};

template <class Element>
struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::ColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template <class Element>
struct L1AlignHelper<Element, layout::PaddingRowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::PaddingColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template <class Element>
struct L1AlignHelper<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};





template <class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    
};

template <>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template <>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};


template <class Element_, bool isMx = false>
struct GetL0Element {
    using Element = Element_;
};

template <>
struct ElementAccumulatorSelector<AscendC::int4b_t, AscendC::int4b_t> {
    using ElementAccumulator = int32_t;
};

template <class GmAType>
struct L1ATypeSelector {
    
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::VectorLayout>> {
    using L1AType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::zN>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::nZ>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};





template <class GmBType>
struct L1BTypeSelector {
    
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::zN>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::nZ>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};





template <class GmBiasType, class ElementAccumulator>
struct L1BiasTypeSelector {
    
};

template <class ElementAccumulator>
struct L1BiasTypeSelector<void, ElementAccumulator> {
    using GMBiasType = void;
    using L1BiasType = void;
    using L0BiasType = void;
};

template <class Element, class ElementAccumulator>
struct L1BiasTypeSelector<Gemm::GemmType<Element, layout::VectorLayout>, ElementAccumulator> {
    using GMBiasType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>;
    using L1BiasType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
    using L0BiasType = Gemm::GemmType<ElementAccumulator, layout::VectorLayout, AscendC::TPosition::C2>;
};

template <class ArchTag>
struct L0ALayoutSelector {
    
};

template <>
struct L0ALayoutSelector<Arch::AtlasA2> {
    using Layout = layout::zZ;
};

template <>
struct L0ALayoutSelector<Arch::Ascend950> {
    using Layout = layout::zN;
};

template <class Element, class Layout, class Enable = void>
struct L1AlignHelperTla {
    
};

template <class Element, class Layout>
struct L1AlignHelperTla<Element, Layout, std::enable_if_t<tla::detail::isRowMajor<Layout>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element, class Layout>
struct L1AlignHelperTla<Element, Layout, std::enable_if_t<tla::detail::isColumnMajor<Layout>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

///////////////////////////////////////
// new add
template <>
struct ElementAccumulatorSelector<int32_t, int32_t> {
    using ElementAccumulator = int32_t;
};

template <class GmAType, class GmBType>
struct L1AndL0TypeSelectorGemm {
    
    
};

template <class Element>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template <>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<int8_t, layout::RowMajor>, Gemm::GemmType<int8_t, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

template <class Element>
struct L1AndL0TypeSelectorGemm<
    Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template <>
struct L1AndL0TypeSelectorGemm<
    Gemm::GemmType<int8_t, layout::ColumnMajor>, Gemm::GemmType<int8_t, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

template <class Element>
struct L1AndL0TypeSelectorGemm<
    Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template <class Element>
struct L1AndL0TypeSelectorGemm<
    Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template <>
struct L1AndL0TypeSelectorGemm<Gemm::GemmType<int8_t, layout::ColumnMajor>, Gemm::GemmType<int8_t, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>;
    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

///////////////////////////////////////
// Check for whether the tile shape is aligned (Default: 32Bytes).
template <class L1TileShape, class L0TileShape, class ElementA, class ElementB, uint32_t _aligned = 32>
struct TileShapeAlignChecker {
    static constexpr uint32_t _ALIGN = _aligned * 8;

    
    
    
    
    
    
    
    
};

///////////////////////////////////////
} // namespace Catlass::Gemm::helper


// END vendored catlass/gemm/helper.hpp


// BEGIN vendored catlass/gemm/tile/cast_fp8_to_fp16.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/cast_fp8_to_fp16.hpp
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
// already embedded: catlass/arch/resource.hpp
// already embedded: catlass/coord.hpp

// BEGIN vendored catlass/gemm_coord.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/coord.hpp

namespace Catlass {

/// Shape of a matrix multiply-add operation
template <
    /// Rows of matrix product
    uint32_t M_ = 1,
    /// Columns of matrix product
    uint32_t N_ = 1,
    /// Inner dimension of matrix product
    uint32_t K_ = 1>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    static constexpr int64_t MN = M * N;
    static constexpr int64_t MK = M * K;
    static constexpr int64_t KN = N * K;
    static constexpr int64_t MNK = M * N * K;

    static constexpr int64_t COUNT = MNK;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }

    CATLASS_HOST_DEVICE
    static Coord<2> ToCoordMN()
    {
        return MakeCoord(M, N);
    }

    CATLASS_HOST_DEVICE
    static Coord<2> ToCoordMK()
    {
        return MakeCoord(M, K);
    }

    CATLASS_HOST_DEVICE
    static Coord<2> ToCoordKN()
    {
        return MakeCoord(K, N);
    }
};

/// GemmCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a Gemm problem.
struct GemmCoord : public Coord<3, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// Gemm M dimension - rows of the output C matrix
    static constexpr int M_INDEX = 0;

    /// Gemm N dimension - columns of the output C matrix
    static constexpr int N_INDEX = 1;

    /// Gemm K dimension - inner dimension of the Gemm problem
    static constexpr int K_INDEX = 2;

    /// Default ctor
    CATLASS_HOST_DEVICE
    GemmCoord()
    {}

    /// Constructs from Coord<3> and a batch
    CATLASS_HOST_DEVICE
    GemmCoord(Coord<3, Index> const& coord) : Base(coord)
    {}

    /// Helper to construct from a K, N, M, batch variables
    CATLASS_HOST_DEVICE
    GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k))
    {}

    /// Returns the Gemm M coordinate
    CATLASS_HOST_DEVICE
    Index const& m() const
    {
        return this->At(M_INDEX);
    }

    /// Returns reference to the Gemm M coordinate
    CATLASS_HOST_DEVICE
    Index& m()
    {
        return this->At(M_INDEX);
    }

    /// Returns the Gemm N coordinate
    CATLASS_HOST_DEVICE
    Index const& n() const
    {
        return this->At(N_INDEX);
    }

    /// Returns reference to the Gemm N coordinate
    CATLASS_HOST_DEVICE
    Index& n()
    {
        return this->At(N_INDEX);
    }

    /// Returns the Gemm K coordinate
    CATLASS_HOST_DEVICE
    Index const& k() const
    {
        return this->At(K_INDEX);
    }

    /// Returns reference to the Gemm K coordinate
    CATLASS_HOST_DEVICE
    Index& k()
    {
        return this->At(K_INDEX);
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMN() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMK() const
    {
        return this->GetCoordByAxis<M_INDEX, K_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordKN() const
    {
        return this->GetCoordByAxis<K_INDEX, N_INDEX>();
    }
};

} // namespace Catlass


// END vendored catlass/gemm_coord.hpp

// already embedded: catlass/gemm/dispatch_policy.hpp
// already embedded: catlass/gemm/helper.hpp

namespace Catlass::Gemm::Tile {

template <class ArchTag, class SrcType_, class DstType_, uint32_t COMPUTE_LENGTH>
struct TileCastFp8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;
    using LayoutRowMajor = Catlass::layout::RowMajor;

    
    

    static const uint32_t Alignment = 256;
    // static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(int8_t);

    struct Params {
        half scalar;
        half zeroPoint;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(half scalar_, half zeroPoint_)
        {
            scalar = scalar_;
            zeroPoint = zeroPoint_;
        }
    };

    CATLASS_DEVICE
    TileCastFp8ToFp16Dequant()
    {}

    CATLASS_DEVICE
    TileCastFp8ToFp16Dequant(Arch::Resource<ArchTag>& resource, Params const& params_) : params(params_)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc));
            bufferOffset += COMPUTE_LENGTH;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            outputBuffer[i] = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                                  .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            workspace[i] = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                               .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        InitLocalMaskVec(resource, bufferOffset);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> gmDst, LayoutDst const& layoutDst, AscendC::GlobalTensor<ElementSrc> gmSrc,
        LayoutSrc const& layoutSrc, uint32_t& bufferIndex)
    {
        uint32_t tilesNum, tileLen, srcStride, dstStride;
        if constexpr (std::is_same_v<LayoutSrc, layout::RowMajor>) {
            tilesNum = layoutSrc.shape(0);
            tileLen = layoutSrc.shape(1);
            srcStride = layoutSrc.stride(0);
            dstStride = layoutDst.stride(0);
        } else if constexpr (std::is_same_v<LayoutSrc, layout::ColumnMajor>) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            srcStride = layoutSrc.stride(1);
            dstStride = layoutDst.stride(1);
        }

        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        uint32_t tilesRemain = tilesNum % AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < tilesRemain) {
            tilesPerAiv++;
        }

        uint64_t taskSrcOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * srcStride;
        uint64_t taskDstOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * dstStride;
        if (AscendC::GetSubBlockIdx() >= tilesRemain) {
            taskSrcOffset += tilesRemain * srcStride;
            taskDstOffset += tilesRemain * dstStride;
        }

        uint32_t totalLoops = 0;
        uint32_t loopsPerTile, tilesInALoop;
        uint32_t tileLenRoundFp8 = RoundUp<Alignment, uint32_t>(tileLen);
        if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
            // One signle tile length is bigger than COMPUTE_LENGTH, which should be clipped.
            loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH);
            totalLoops = tilesPerAiv * loopsPerTile;
        } else if (tileLenRoundFp8 != 0) {
            // COMPUTE_LENGTH is bigger than tile length, such that more than one tiles can be arranged together.
            tilesInALoop = COMPUTE_LENGTH / tileLenRoundFp8;
            totalLoops = CeilDiv(tilesPerAiv, tilesInALoop);
        } // tileLenRoundFp8 == 0 --> totalLoops = 0

        uint32_t tileTailLen = tileLen % COMPUTE_LENGTH;
        uint64_t srcProcessOffset, dstProcessOffset;
        uint32_t loadLen = COMPUTE_LENGTH, storeLen, loadRepeat = 1, storeRepeat = 1;
        // uint32_t srcLoadStride = srcStride, dstLoadStride = tileLenRoundFp8,
        //         srcStoreStride = tileLenRoundFp8, dstStoreStride = dstStride;
        for (int ldx = 0; ldx < totalLoops; ldx++) {
            // Dynamic compute length
            if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
                uint32_t fullTileRounds = ldx / loopsPerTile;
                uint32_t residueTileRounds = ldx % loopsPerTile;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride + residueTileRounds * COMPUTE_LENGTH;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride + residueTileRounds * COMPUTE_LENGTH;

                if ((residueTileRounds == loopsPerTile - 1) && (tileTailLen != 0)) {
                    loadLen = tileTailLen;
                }
            } else {
                uint32_t fullTileRounds = ldx * tilesInALoop;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride;

                loadLen = tileLen;
                storeLen = tileLen;
                loadRepeat = tilesInALoop;
                storeRepeat = tilesInALoop;

                if ((ldx == totalLoops - 1) && (tilesPerAiv % tilesInALoop != 0)) {
                    loadRepeat = tilesPerAiv % tilesInALoop;
                    storeRepeat = loadRepeat;
                }
            }
            // uint32_t srcLoadStride = srcStride;
            // uint32_t dstLoadStride = tileLenRoundFp8;
            // uint32_t srcStoreStride = tileLenRoundFp8;

            // GM -> UB
            AscendC::DataCopyExtParams dataCopyParamsIn(
                loadRepeat, loadLen * sizeof(ElementSrc),
                (srcStride - loadLen) * sizeof(ElementSrc), //
                (tileLenRoundFp8 - loadLen) * sizeof(ElementSrc) / BYTE_PER_BLK, 0);
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[bufferIndex]);
            AscendC::DataCopyPad(inputBuffer[bufferIndex], gmSrc[srcProcessOffset], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[bufferIndex]);

            Dequant(
                inputBuffer[bufferIndex], outputBuffer[bufferIndex], value_vector1, value_vector2,
                workspace[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[bufferIndex]);

            AscendC::DataCopyExtParams dataCopyParams(
                storeRepeat, storeLen * sizeof(ElementDst),
                (tileLenRoundFp8 - storeLen) * sizeof(ElementDst) / BYTE_PER_C0,
                (dstStride - storeLen) * sizeof(ElementDst), 0);
            AscendC::DataCopyPad(gmDst[dstProcessOffset], outputBuffer[bufferIndex], dataCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[bufferIndex]);
            bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
        }
    }

    /*
     * DATAFLOW: gmSrc<float> -> ubInTensor -> ubOutTensor -> gmDst
     */
    CATLASS_DEVICE
    void EpCastFp32ToFp16(
        AscendC::GlobalTensor<half> gmDst, LayoutRowMajor layoutDst, AscendC::GlobalTensor<float> gmSrc,
        LayoutRowMajor layoutSrc)
    {
        AscendC::LocalTensor<float> ubInTensor[BUFFER_NUM];
        AscendC::LocalTensor<half> ubOutTensor[BUFFER_NUM];

        Arch::Resource<ArchTag> resource;
        int64_t bufferOffset = 0;
        const int64_t CAST_LENGTH = 32 * 1024 / sizeof(half); // 一次处理16K个数据
        for (int i = 0; i < BUFFER_NUM; i++) {
            ubInTensor[i] = resource.ubBuf.template GetBufferByByte<float>(bufferOffset);
            bufferOffset += CAST_LENGTH * 4; // float 4字节
        }
        for (int i = 0; i < BUFFER_NUM; i++) {
            ubOutTensor[i] = resource.ubBuf.template GetBufferByByte<half>(bufferOffset);
            bufferOffset += CAST_LENGTH * 2; // half 2字节
        }

        uint32_t tilesNum, tileLen, srcStride, dstStride;
        tilesNum = layoutSrc.shape(0);
        tileLen = layoutSrc.shape(1);
        srcStride = layoutSrc.stride(0);
        dstStride = layoutDst.stride(0); // Always RowMajor

        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        uint32_t tilesRemain = tilesNum % AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < tilesRemain) {
            tilesPerAiv++;
        }

        uint64_t taskSrcOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * srcStride;
        uint64_t taskDstOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * dstStride;
        if (AscendC::GetSubBlockIdx() >= tilesRemain) {
            taskSrcOffset += tilesRemain * srcStride;
            taskDstOffset += tilesRemain * dstStride;
        }

        uint32_t totalLoops = 0;
        uint32_t loopsPerTile, tilesInALoop;
        uint32_t tileLenRoundFp8 = RoundUp<Alignment, uint32_t>(tileLen);
        if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
            // One signle tile length is bigger than COMPUTE_LENGTH, which should be clipped.
            loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH);
            totalLoops = tilesPerAiv * loopsPerTile;
        } else if (tileLenRoundFp8 != 0) {
            // COMPUTE_LENGTH is bigger than tile length, such that more than one tiles can be arranged together.
            tilesInALoop = COMPUTE_LENGTH / tileLenRoundFp8;
            totalLoops = CeilDiv(tilesPerAiv, tilesInALoop);
        } // tileLenRoundFp8 == 0 --> totalLoops = 0

        uint32_t tileTailLen = tileLen % COMPUTE_LENGTH;
        uint64_t srcProcessOffset, dstProcessOffset;
        uint32_t loadLen = COMPUTE_LENGTH, storeLen, loadRepeat = 1, storeRepeat = 1;
        // uint32_t srcLoadStride = srcStride, dstLoadStride = tileLenRoundFp8,
        //         srcStoreStride = tileLenRoundFp8, dstStoreStride = dstStride;
        for (int ldx = 0; ldx < totalLoops; ldx++) {
            // Dynamic compute length
            if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
                uint32_t fullTileRounds = ldx / loopsPerTile;
                uint32_t residueTileRounds = ldx % loopsPerTile;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride + residueTileRounds * COMPUTE_LENGTH;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride + residueTileRounds * COMPUTE_LENGTH;

                if ((residueTileRounds == loopsPerTile - 1) && (tileTailLen != 0)) {
                    loadLen = tileTailLen;
                }
            } else {
                uint32_t fullTileRounds = ldx * tilesInALoop;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride;

                loadLen = tileLen;
                storeLen = tileLen;
                loadRepeat = tilesInALoop;
                storeRepeat = tilesInALoop;

                if ((ldx == totalLoops - 1) && (tilesPerAiv % tilesInALoop != 0)) {
                    loadRepeat = tilesPerAiv % tilesInALoop;
                    storeRepeat = loadRepeat;
                }
            }

            // copy GM -> UB <fp32>
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::DataCopyExtParams dataCopyParamsIn(
                loadRepeat,
                loadLen * sizeof(float), // float loaded
                (srcStride - loadLen) * sizeof(float), (tileLenRoundFp8 - loadLen) * sizeof(float) / BYTE_PER_BLK, 0);
            AscendC::DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(ubInTensor[bufferIndexForCast], gmSrc[srcProcessOffset], dataCopyParamsIn, padParams);
            // Begin casting ...
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::Cast(
                ubOutTensor[bufferIndexForCast], ubInTensor[bufferIndexForCast], AscendC::RoundMode::CAST_RINT,
                CAST_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBufferForCast[bufferIndexForCast]);
            // End casting ...

            AscendC::DataCopyExtParams dataCopyParams(
                storeRepeat, storeLen * sizeof(half), (tileLenRoundFp8 - storeLen) * sizeof(half) / BYTE_PER_C0,
                (dstStride - storeLen) * sizeof(half), 0);
            AscendC::DataCopyPad(gmDst[dstProcessOffset], ubOutTensor[bufferIndexForCast], dataCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventIdBufferForCast[bufferIndexForCast]);
            bufferIndexForCast = (bufferIndexForCast + 1) % BUFFER_NUM;
        }
    }

private:
    CATLASS_DEVICE
    void InitLocalMaskVec(Arch::Resource<ArchTag>& resource, int64_t& bufferOffset)
    {
        int16_t value_uint = 0x4000;
        value_vector1 = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector1, value_uint, 128);
        AscendC::PipeBarrier<PIPE_V>();
        // AscendC::PipeBarrier<PIPE_V>();
        value_uint = 0x3FFF;
        value_vector2 = (resource.ubBuf.template GetBufferByByte<ElementSrc>(bufferOffset * sizeof(ElementSrc)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector2, value_uint, 128);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void Dequant(
        AscendC::LocalTensor<int8_t>& src, AscendC::LocalTensor<half>& dst,
        AscendC::LocalTensor<int16_t>& value_vector1, AscendC::LocalTensor<int16_t>& value_vector2,
        AscendC::LocalTensor<half>& workspace)
    {
        pipe_barrier(PIPE_V);
        uint32_t num = COMPUTE_LENGTH;
        num = (num + 128 - 1) / 128 * 128;
        AscendC::Cast<half, uint8_t>(
            dst.template ReinterpretCast<half>(), src.template ReinterpretCast<uint8_t>(),
            AscendC::RoundMode::CAST_NONE, num);
        pipe_barrier(PIPE_V);

        AscendC::Adds<half>(dst, dst, 1024, num);
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(
            dst.template ReinterpretCast<uint16_t>(), dst.template ReinterpretCast<uint16_t>(), 7, num);
        pipe_barrier(PIPE_V);

        uint64_t mask = 128;
        AscendC::And<int16_t>(
            workspace.template ReinterpretCast<int16_t>(), dst.template ReinterpretCast<int16_t>(), value_vector1, mask,
            num / 128, {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(
            workspace.template ReinterpretCast<uint16_t>(), workspace.template ReinterpretCast<uint16_t>(), 1, num);
        pipe_barrier(PIPE_V);

        AscendC::And<int16_t>(
            dst.template ReinterpretCast<int16_t>(), dst.template ReinterpretCast<int16_t>(), value_vector2, mask,
            num / 128, {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::Or<int16_t>(
            dst.template ReinterpretCast<int16_t>(), dst.template ReinterpretCast<int16_t>(),
            workspace.template ReinterpretCast<int16_t>(), num);
        pipe_barrier(PIPE_V);

        AscendC::Muls<half>(dst.template ReinterpretCast<half>(), dst.template ReinterpretCast<half>(), 1 << 8, num);
        pipe_barrier(PIPE_V);

        AscendC::Adds(dst, dst, params.zeroPoint, num);
        pipe_barrier(PIPE_V);

        AscendC::Muls(dst, dst, params.scalar, num);
        pipe_barrier(PIPE_V);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<int8_t> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<int16_t> value_vector1;
    AscendC::LocalTensor<int16_t> value_vector2;
    AscendC::LocalTensor<half> outputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<half> workspace[BUFFER_NUM];
    AscendC::TEventID EventIdBuffer[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID EventIdBufferForCast[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    uint32_t bufferIndexForCast{0};

    Params params;
};

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/cast_fp8_to_fp16.hpp



// END vendored catlass/gemm/tile/cast_fp8_to_fp16.hpp


// BEGIN vendored catlass/gemm/tile/cast_int4_to_int8.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/cast_int4_to_int8.hpp
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
// already embedded: catlass/arch/resource.hpp
// already embedded: catlass/coord.hpp
// already embedded: catlass/gemm_coord.hpp
// already embedded: catlass/gemm/dispatch_policy.hpp
// already embedded: catlass/gemm/helper.hpp

namespace Catlass::Gemm::Tile {

template <
    class ArchTag, class SrcType_, class DstType_,
    // Length of the compute elements
    uint32_t COMPUTE_LEN_, uint32_t STAGES = 2>
struct TileCastInt4ToInt8 {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;

    static constexpr uint32_t ELE_NUM_PER_BLK_INT8 = BYTE_PER_BLK / sizeof(ElementDst);
    static constexpr uint32_t ELE_NUM_PER_BLK_INT4 = BYTE_PER_BLK / sizeof(ElementDst) * 2;

    

    

    static constexpr uint32_t COMPUTE_LEN = COMPUTE_LEN_;
    

    struct Params {};

    /// Construct
    CATLASS_DEVICE
    TileCastInt4ToInt8(Arch::Resource<ArchTag> const& resource, Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            uint32_t ubOffset = 0;
            uint32_t ubInSize = COMPUTE_LEN * sizeof(int8_t) / 2;
            uint32_t ubOutSize = COMPUTE_LEN * sizeof(ElementDst);
            uint32_t ubWorkspaceSize = COMPUTE_LEN * sizeof(half);
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                ubInTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
                ubOffset += ubInSize;
                ubOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
                ubOffset += ubOutSize;
                ubWorkspaceList[i] = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
                ubOffset += ubWorkspaceSize;

                ubEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~TileCastInt4ToInt8()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc)
    {
        uint32_t tilesNum = layoutSrc.shape(0);
        uint32_t tileLen = layoutSrc.shape(1);
        uint32_t tileLenRoundInt8 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT8);
        uint32_t tileLenRoundInt4 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT4);
        uint64_t tileStrideSrc = layoutSrc.stride(0);
        uint64_t tileStrideDst = layoutDst.stride(0);
        if constexpr (std::is_same_v<LayoutSrc, layout::ColumnMajor>) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            tileLenRoundInt8 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT8);
            tileLenRoundInt4 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT4);
            tileStrideSrc = layoutSrc.stride(1);
            tileStrideDst = layoutDst.stride(1);
        }
        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < (tilesNum % AscendC::GetSubBlockNum())) {
            tilesPerAiv++;
        }
        uint64_t taskOffsetSrc = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideSrc;
        uint64_t taskOffsetDst = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideDst;
        if (AscendC::GetSubBlockIdx() >= (tilesNum % AscendC::GetSubBlockNum())) {
            taskOffsetSrc += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideSrc;
            taskOffsetDst += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideDst;
        }
        uint32_t tilesPerLoop = 32;
        uint32_t loops = CeilDiv(tilesPerAiv, tilesPerLoop);
        uint32_t pingpong = 0;
        for (uint32_t loopIdx = 0; loopIdx < loops; loopIdx++) {
            uint32_t actualTiles = tilesPerLoop;
            if (loopIdx == loops - 1) {
                actualTiles = tilesPerAiv - loopIdx * tilesPerLoop;
            }
            uint64_t tileOffsetSrc = loopIdx * tilesPerLoop * tileStrideSrc;
            AscendC::DataCopyExtParams dataCopyParamsIn(
                actualTiles, (tileLen + 1) / 2 * sizeof(int8_t), (tileStrideSrc - tileLen) * sizeof(int8_t) / 2,
                (tileLenRoundInt4 - tileLen) / ELE_NUM_PER_BLK_INT4, 0);
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);
            AscendC::DataCopyPad(
                ubInTensorList[pingpong], gmSrc[taskOffsetSrc + tileOffsetSrc], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            AscendC::Cast(
                ubWorkspaceList[pingpong], ubInTensorList[pingpong], AscendC::RoundMode::CAST_NONE,
                actualTiles * tileLenRoundInt4);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);

            AscendC::Cast(
                ubOutTensorList[pingpong], ubWorkspaceList[pingpong], AscendC::RoundMode::CAST_NONE,
                actualTiles * tileLenRoundInt4);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);

            uint64_t tileOffsetDst = loopIdx * tilesPerLoop * tileStrideDst;
            AscendC::DataCopyExtParams dataCopyParamsOut(
                actualTiles, tileLen * sizeof(ElementDst), (tileLenRoundInt4 - tileLen) / ELE_NUM_PER_BLK_INT8,
                (tileStrideDst - tileLen) * sizeof(ElementDst), 0);
            AscendC::DataCopyPad(gmDst[taskOffsetDst + tileOffsetDst], ubOutTensorList[pingpong], dataCopyParamsOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            pingpong = (pingpong + 1) % STAGES;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementSrc> ubInTensorList[STAGES];
    AscendC::LocalTensor<ElementDst> ubOutTensorList[STAGES];
    AscendC::LocalTensor<half> ubWorkspaceList[STAGES];

    int32_t ubEventList[STAGES];
};

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/cast_int4_to_int8.hpp



// END vendored catlass/gemm/tile/cast_int4_to_int8.hpp


// BEGIN vendored catlass/gemm/tile/cast_int8_to_fp16.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/cast_int8_to_fp16.hpp
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
// already embedded: catlass/arch/resource.hpp
// already embedded: catlass/coord.hpp
// already embedded: catlass/gemm_coord.hpp
// already embedded: catlass/gemm/dispatch_policy.hpp
// already embedded: catlass/gemm/helper.hpp

namespace Catlass::Gemm::Tile {

template <
    class ArchTag, class SrcType_, class DstType_,
    // Length of the compute elements
    uint32_t COMPUTE_LEN_, uint32_t STAGES = 2>
struct TileCastInt8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;

    
    

    static constexpr uint32_t ELE_NUM_PER_BLK_INT8 = BYTE_PER_BLK / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_BLK_HALF = BYTE_PER_BLK / sizeof(ElementDst);

    static constexpr uint32_t COMPUTE_LEN = COMPUTE_LEN_;

    

    struct Params {
        half deqScalar;
        half deqZeroPoint;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(half deqScalar_, half deqZeroPoint_)
        {
            deqScalar = deqScalar_;
            deqZeroPoint = deqZeroPoint_;
        }
    };

    /// Construct
    CATLASS_DEVICE
    TileCastInt8ToFp16Dequant(Arch::Resource<ArchTag> const& resource, Params const& params_) : params(params_)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            uint32_t ubOffset = 0;
            uint32_t ubInSize = COMPUTE_LEN * sizeof(ElementSrc);
            uint32_t ubOutSize = COMPUTE_LEN * sizeof(ElementDst);
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                ubInTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
                ubOffset += ubInSize;
                ubOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
                ubOffset += ubOutSize;

                ubEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~TileCastInt8ToFp16Dequant()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc)
    {
        uint32_t tilesNum = layoutSrc.shape(0);
        uint32_t tileLen = layoutSrc.shape(1);
        uint32_t tileLenRoundInt8 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT8);
        uint64_t tileStrideSrc = layoutSrc.stride(0);
        uint64_t tileStrideDst = layoutDst.stride(0);
        if constexpr (std::is_same_v<LayoutSrc, layout::ColumnMajor>) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            tileLenRoundInt8 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT8);
            tileStrideSrc = layoutSrc.stride(1);
            tileStrideDst = layoutDst.stride(1);
        }
        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < (tilesNum % AscendC::GetSubBlockNum())) {
            tilesPerAiv++;
        }
        uint64_t taskOffsetSrc = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideSrc;
        uint64_t taskOffsetDst = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideDst;
        if (AscendC::GetSubBlockIdx() >= (tilesNum % AscendC::GetSubBlockNum())) {
            taskOffsetSrc += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideSrc;
            taskOffsetDst += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideDst;
        }
        uint32_t tilesPerLoop = COMPUTE_LEN / tileLenRoundInt8;
        uint32_t loops = CeilDiv(tilesPerAiv, tilesPerLoop);
        uint32_t pingpong = 0;
        for (uint32_t loopIdx = 0; loopIdx < loops; loopIdx++) {
            uint32_t actualTiles = tilesPerLoop;
            if (loopIdx == loops - 1) {
                actualTiles = tilesPerAiv - loopIdx * tilesPerLoop;
            }
            uint64_t tileOffsetSrc = loopIdx * tilesPerLoop * tileStrideSrc;
            AscendC::DataCopyExtParams dataCopyParamsIn(
                actualTiles, tileLen * sizeof(ElementSrc), (tileStrideSrc - tileLen) * sizeof(ElementSrc),
                (tileLenRoundInt8 - tileLen) / ELE_NUM_PER_BLK_INT8, 0);
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);
            AscendC::DataCopyPad(
                ubInTensorList[pingpong], gmSrc[taskOffsetSrc + tileOffsetSrc], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            AscendC::Cast(
                ubOutTensorList[pingpong], ubInTensorList[pingpong], AscendC::RoundMode::CAST_NONE,
                actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);

            AscendC::Adds(
                ubOutTensorList[pingpong], ubOutTensorList[pingpong], params.deqZeroPoint,
                actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(
                ubOutTensorList[pingpong], ubOutTensorList[pingpong], params.deqScalar, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);

            uint64_t tileOffsetDst = loopIdx * tilesPerLoop * tileStrideDst;
            AscendC::DataCopyExtParams dataCopyParamsOut(
                actualTiles, tileLen * sizeof(ElementDst), (tileLenRoundInt8 - tileLen) / ELE_NUM_PER_BLK_HALF,
                (tileStrideDst - tileLen) * sizeof(ElementDst), 0);
            AscendC::DataCopyPad(gmDst[taskOffsetDst + tileOffsetDst], ubOutTensorList[pingpong], dataCopyParamsOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            pingpong = (pingpong + 1) % STAGES;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementSrc> ubInTensorList[STAGES];
    AscendC::LocalTensor<ElementDst> ubOutTensorList[STAGES];
    int32_t ubEventList[STAGES];

    Params params;
};

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/cast_int8_to_fp16.hpp



// END vendored catlass/gemm/tile/cast_int8_to_fp16.hpp


// BEGIN vendored catlass/gemm/tile/copy_gm_to_l1.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp
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

// BEGIN vendored catlass/gemm/tile/tile_copy_tla.hpp
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

namespace Catlass::Gemm::Tile {

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla {
    
};

// Extended template for TileCopyTla that supports manually specifying LayoutTagSrc and LayoutTagDst.
// Users can specialize the copy class by LayoutTagSrc and LayoutTagDst.
template <class ArchTag, class TensorSrc, class TensorDst, class LayoutTagSrc, class LayoutTagDst>
struct TileCopyTlaExt {
    
};

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopySparseTla {
    
};

template <class ArchTag, class Element, class TensorSrc, class TensorDst, class TensorIdx, class Enable = void>
struct CopyL1ToL0BSparseTla {
    
};

// used for FA LoadQ, multi matrix datacopy ND->NZ
template <class ArchTag, class TensorSrc, class TensorDst>
struct TileCopyFAQTla {
    
};

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/tile_copy_tla.hpp


// BEGIN vendored tla/tensor.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/arch/arch.hpp
// already embedded: tla/layout.hpp
// already embedded: tla/integral/integral_constant.hpp
// already embedded: tla/integral/integral_sequence.hpp
// already embedded: tla/integral/integral_math.hpp
// already embedded: tla/tuple/tuple_math.hpp
// already embedded: tla/tuple/tuple_algorithms.hpp

namespace tla {

//
// Underscore slicing utilities
//
namespace detail {

// Safe "element type at index I" helper that avoids instantiating get<I> for invalid indices.
template <class Coord, int I, class Enable = void>
struct coord_elem_type {
    using type = void;
};
template <class Coord, int I>
struct coord_elem_type<
    Coord, I, std::enable_if_t<(I >= 0) && (I < (int)tla::tuple_size<tla::remove_cvref_t<Coord>>::value)>> {
    using type = tla::remove_cvref_t<decltype(tla::get<I>(std::declval<Coord>()))>;
};

template <class Coord, int I>
struct coord_elem_is_underscore : tla::is_underscore<typename coord_elem_type<Coord, I>::type> {};

// Count underscores.
template <class Coord, int I, class Enable = void>
struct underscore_count_from : tla::C<0> {};
template <class Coord, int I>
struct underscore_count_from<Coord, I, std::enable_if_t<(I >= 0)>> {
    static constexpr int value =
        (coord_elem_is_underscore<Coord, I>::value ? 1 : 0) + underscore_count_from<Coord, I - 1>::value;
};
template <class Coord>
struct underscore_count
    : tla::C<underscore_count_from<Coord, (int)tla::tuple_size<tla::remove_cvref_t<Coord>>::value - 1>::value> {};

// Build index sequences for underscore dims (stable 0..R-1 recursion).
template <class Coord, int I, int R, int... Is>
struct underscore_indices_impl;
template <class Coord, int R, int... Is>
struct underscore_indices_impl<Coord, R, R, Is...> {
    using type = seq<Is...>;
};
template <class Coord, int I, int R, int... Is>
struct underscore_indices_impl
    : std::conditional_t<
          coord_elem_is_underscore<Coord, I>::value, underscore_indices_impl<Coord, I + 1, R, Is..., I>,
          underscore_indices_impl<Coord, I + 1, R, Is...>> {};

template <class Coord>
using underscore_indices =
    typename underscore_indices_impl<Coord, 0, (int)tla::tuple_size<tla::remove_cvref_t<Coord>>::value>::type;

// Replace every tla::_ with 0 for offset computation.
template <class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) underscore_to_zero(T const& x)
{
    if constexpr (tla::is_underscore<T>::value) {
        return tla::_0{};
    } else {
        return x;
    }
}

template <class Coord, int... I>
CATLASS_HOST_DEVICE constexpr auto replace_underscore_with_zero_impl(Coord const& c, seq<I...>)
{
    return tla::MakeCoord(underscore_to_zero(tla::get<I>(c))...);
}

template <class Coord>
CATLASS_HOST_DEVICE constexpr auto replace_underscore_with_zero(Coord const& c)
{
    
    return replace_underscore_with_zero_impl(c, tuple_seq<Coord>{});
}

// Build a layout from selected top-level indices of `layout`.
template <class Layout, int... Is>
CATLASS_HOST_DEVICE constexpr auto select_layout(Layout const& layout, seq<Is...>)
{
    auto shape_new = tla::MakeTuple(tla::get<Is>(layout.shape())...);
    auto stride_new = tla::MakeTuple(tla::get<Is>(layout.stride())...);
    auto origin_new = tla::MakeTuple(tla::get<Is>(layout.originShape())...);
    return tla::MakeLayout(shape_new, stride_new, origin_new);
}

template <size_t N>
using MakeZeroTuple = decltype(tla::tuple_repeat<N>(tla::Int<0>{}));

} // namespace detail

//
// slice_and_offset
//
// A lightweight helper that factors underscore slicing into:
// - a projected layout (keeping only underscored dimensions, in-order), and
// - a base offset computed at the fixed indices (underscored dims treated as 0).
//
// Notes:
// - `coord_arg` must be a one-level `tla::tuple` and contain at least one `tla::_`.
// - This function does not perform runtime bounds checks against originShape(); out-of-bounds is undefined behavior.
// - Returned `offset` is an element offset intended for `BuiltinTensor::operator[](offset)` view creation.
template <class CoordArg, class Layout, class BaseCoord>
CATLASS_HOST_DEVICE constexpr auto slice_and_offset(
    CoordArg const& coord_arg, Layout const& layout, BaseCoord const& base_coord)
{
    
    
    
    
    

    constexpr int k = detail::underscore_count<CoordArg>::value;
    
    

    // Compute base offset using zeros for underscores
    auto coord0 = detail::replace_underscore_with_zero(coord_arg);
    auto full0 = base_coord + coord0;
    auto offset = (int64_t)layout(full0);

    // Determine output dims (underscored dims) and build projected layout.
    using Us = detail::underscore_indices<CoordArg>;
    auto layout_proj = detail::select_layout(layout, Us{});

    return tla::MakeTuple(layout_proj, offset);
}

// Convenience overload (no base_coord): assume base_coord == 0.
template <class CoordArg, class Layout>
CATLASS_HOST_DEVICE constexpr auto slice_and_offset(CoordArg const& coord_arg, Layout const& layout)
{
    using Z = detail::MakeZeroTuple<Layout::rank>;
    return slice_and_offset(coord_arg, layout, Z{});
}

//
// Tensor
//

namespace detail {

template <class A, class B, int... Is>
CATLASS_DEVICE constexpr auto HadamardU32(A const& a, B const& b, seq<Is...>)
{
    return MakeCoord((static_cast<uint32_t>(get<Is>(a)) * static_cast<uint32_t>(get<Is>(b)))...);
}

template <class TensorT, class CoordT, class ShapeT, int R>
CATLASS_DEVICE constexpr auto GetTileImpl(TensorT const& tensor, CoordT const& coord, ShapeT const& shape, Int<R>)
{
    
    

    auto layoutNew = GetTileLayout(tensor.layout(), shape, coord);
    auto coordNew = tensor.coord() + coord;
    return MakeTensor(tensor.data(), layoutNew, coordNew, Catlass::Arch::PositionType<TensorT::position>{});
}

template <class TensorT, class TileCoord, class TileShape, int R>
CATLASS_DEVICE constexpr auto TileViewImpl(
    TensorT const& tensor, TileCoord const& tileCoord, TileShape const& tileShape, Int<R>)
{
    
    

    auto elementOffset = HadamardU32(tileCoord, tileShape, tuple_seq<TileCoord>{});
    auto layoutNew = GetTileLayout(tensor.layout(), tileShape, elementOffset);
    auto coordNew = tensor.coord() + elementOffset;
    return MakeTensor(tensor.data(), layoutNew, coordNew, Catlass::Arch::PositionType<TensorT::position>{});
}

} // namespace detail

template <class BuiltinTensor, class Layout_, class Coord_, AscendC::TPosition Position>
struct Tensor {
    using Element = typename BuiltinTensor::PrimType;
    using Layout = Layout_;
    using Coord = Coord_;
    static constexpr AscendC::TPosition position = Position;

    CATLASS_HOST_DEVICE constexpr Tensor()
    {}

    CATLASS_HOST_DEVICE constexpr Tensor(
        BuiltinTensor const& builtinTensor, Layout const& layout, Coord const& coord = {})
        : rep_(builtinTensor, layout, coord)
    {}

    //
    // Accessors
    //

    static constexpr int rank = Layout::rank;

    CATLASS_HOST_DEVICE constexpr decltype(auto) tensor() const
    {
        return *this;
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) data() const
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) data()
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) layout() const
    {
        return get<1>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) coord() const
    {
        return get<2>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) shape() const
    {
        return layout().shape();
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) stride() const
    {
        return layout().stride();
    }

    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape() const
    {
        return layout().originShape();
    }

    //
    // Indexing / slicing
    //
    // - No underscore: returns `data()[layout()(coord()+coord_arg)]`
    // - Underscores (0..rank, one-level coord): returns a subtensor view over the underscored dimensions (kept
    // in-order)
    //   Notes:
    //   - Coord must be one-level (no nested tuples in coord elements).
    //   - Fixed (non-underscore) indices are expected to be within originShape(). This implementation does not
    //     generally perform runtime bounds checks/cropping; out-of-bounds indices result in undefined behavior.
    template <class CoordArg>
    CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(CoordArg const& coord_arg) const
    {
        if constexpr (tla::is_tuple<tla::remove_cvref_t<CoordArg>>::value) {
            
            

            constexpr int k = detail::underscore_count<CoordArg>::value;
            if constexpr (k > 0) {
                

                auto sliced = tla::slice_and_offset(coord_arg, layout(), coord());
                auto layout_proj = tla::get<0>(sliced);
                auto offset = (int64_t)tla::get<1>(sliced);

                using CoordZ = detail::MakeZeroTuple<(size_t)k>;
                auto data_new = data()[static_cast<uint64_t>(offset)];
                return Tensor<decltype(data_new), decltype(layout_proj), CoordZ, position>(
                    data_new, layout_proj, CoordZ{});
            } else {
                // No underscore: point view at coord() + coord_arg
                auto full = coord() + coord_arg;
                return data()[layout()(full)];
            }
        } else {
            // Scalar coordinate convenience (rank-1): treat it as a 1D coord tuple.
            
            auto full = coord() + MakeCoord(coord_arg);
            return data()[layout()(full)];
        }
    }

    template <class Coord0, class Coord1, class... Coords>
    CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(
        Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const
    {
        return operator()(MakeCoord(c0, c1, cs...));
    }

    tla::tuple<BuiltinTensor, Layout, Coord> rep_;
};

template <class BuiltinTensor, class Layout, class PositionType>
CATLASS_HOST_DEVICE constexpr auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout, PositionType)
{
    using Coord = detail::MakeZeroTuple<Layout::rank>;
    return Tensor<BuiltinTensor, Layout, Coord, PositionType::value>(builtinTensor, layout);
}

template <class BuiltinTensor, class Layout, class Coord, class PositionType>
CATLASS_HOST_DEVICE constexpr auto MakeTensor(
    BuiltinTensor const& builtinTensor, Layout const& layout, Coord const& coord, PositionType)
{
    return Tensor<BuiltinTensor, Layout, Coord, PositionType::value>(builtinTensor, layout, coord);
}

// Get a tile from tensor, automatically handling boundary cases (tail tile).
// coord is element coordinate; shape is tile size for memory layout.
// The returned Tensor's layout.shape() is used for memory layout calculation,
// and layout.originShape() is the actual logical size (may be smaller than shape).
// Supports tensors of any rank (rank >= 1).
template <class Tensor, class Coord, class Shape>
CATLASS_DEVICE constexpr auto GetTile(Tensor const& tensor, Coord const& coord, Shape const& shape)
{
    
    
    return detail::GetTileImpl(tensor, coord, shape, Int<Tensor::rank>{});
}

// 从 tensor 中获取一个 tile，自动处理边界情况（tail tile）。
// tileCoord 是 tile 单位坐标；tileShape 是用于内存布局的 tile 尺寸。
// 返回的 Tensor 的 layout.shape() 用于内存布局计算，layout.originShape() 是实际逻辑尺寸（可能小于 tileShape）。
// TileView(tensor, tileCoord, tileShape) = GetTile(tensor, tileCoord ⊙ tileShape, tileShape)
// Supports tensors of any rank (rank >= 1).
template <class TensorT, class TileCoord, class TileShape>
CATLASS_DEVICE constexpr auto TileView(TensorT const& tensor, TileCoord const& tileCoord, TileShape const& tileShape)
{
    
    
    return detail::TileViewImpl(tensor, tileCoord, tileShape, Int<TensorT::rank>{});
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 目标 layout 根据 LayoutTagDst 构造，从 LikeTensor::Element 推断 ElementDst，从 likeTensor 的 originShape 提取尺寸。
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType>
CATLASS_HOST_DEVICE constexpr auto MakeTensorLike(
    BuiltinTensor const& builtinTensor, LikeTensor const& likeTensor, PositionType)
{
    using ElementDst = typename LikeTensor::Element;
    
    return MakeTensorLike<LayoutTagDst, ElementDst>(builtinTensor, likeTensor, PositionType{});
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 使用 layoutBase 的 shape/stride，但继承 likeTensor 的 originShape，从 LikeTensor::Element 推断 ElementDst。
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
CATLASS_HOST_DEVICE constexpr auto MakeTensorLike(
    BuiltinTensor const& builtinTensor, LikeTensor const& likeTensor, PositionType, LayoutBase const& layoutBase)
{
    using ElementDst = typename LikeTensor::Element;
    
    return MakeTensorLike<LayoutTagDst, ElementDst>(builtinTensor, likeTensor, PositionType{}, layoutBase);
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 目标 layout 根据 LayoutTagDst 构造，从 LikeTensor::Element 推断 ElementDst，从 likeTensor 的 originShape
// 提取尺寸。(调用MakeLayout，可能会因分型布局合法要求对shape进行以分型为粒度的向上取整) 允许 BuiltinTensor 的元素类型与
// LikeTensor 的元素类型不同（例如 L0C 使用 ElementAccumulator 而不是 ElementC）。
template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType>
CATLASS_HOST_DEVICE constexpr auto MakeTensorLike(
    BuiltinTensor const& builtinTensor, LikeTensor const& likeTensor, PositionType)
{
    
    
    // rank-1 uses VectorLayout; rank-2 uses the requested layout and element type.
    if constexpr (LikeTensor::rank == 1) {
        
        auto layoutNominal = MakeLayout(get<0>(likeTensor.layout().originShape()));
        using Coord0 = detail::MakeZeroTuple<decltype(layoutNominal)::rank>;
        return Tensor<BuiltinTensor, decltype(layoutNominal), Coord0, PositionType::value>(
            builtinTensor, layoutNominal);
    } else {
        
        auto layoutNominal = MakeLayout<ElementDst, LayoutTagDst>(
            get<0>(likeTensor.layout().originShape()), get<1>(likeTensor.layout().originShape()));
        using Coord0 = detail::MakeZeroTuple<decltype(layoutNominal)::rank>;
        return Tensor<BuiltinTensor, decltype(layoutNominal), Coord0, PositionType::value>(
            builtinTensor, layoutNominal);
    }
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 使用 layoutBase 的 shape/stride，但继承 likeTensor 的 originShape。允许 BuiltinTensor 的元素类型与 LikeTensor
// 的元素类型不同。
template <
    class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
CATLASS_HOST_DEVICE constexpr auto MakeTensorLike(
    BuiltinTensor const& builtinTensor, LikeTensor const& likeTensor, PositionType, LayoutBase const& layoutBase)
{
    
    

    auto layoutFixedStride = MakeLayout(layoutBase.shape(), layoutBase.stride(), likeTensor.originShape());
    using Coord0 = detail::MakeZeroTuple<decltype(layoutFixedStride)::rank>;
    return Tensor<BuiltinTensor, decltype(layoutFixedStride), Coord0, PositionType::value>(
        builtinTensor, layoutFixedStride);
}

} // end namespace tla


// END vendored tla/tensor.hpp


namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType, class L1Type = void>
struct CopyGmToL1 {
    
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType, class L1Type = void>
struct CopyGmToL1IntervalDataCopy {
    
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType, class L1Type = void>
struct CopyGmToL1GMMPTD {
    
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType, class L1Type = void>
struct CopyGmToL1DynamicOptimized {
    
};

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1DynamicOptimized() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(0) <= 16) {
            // If the number of matrix rows is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            for (int i = 0; i < layoutSrc.shape(0); ++i) {
                AscendC::DataCopyParams dataCopyParams(
                    CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)), layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                    (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
                AscendC::DataCopy(
                    dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            if (layoutSrc.stride(0) < STRIDE_LIMIT) {
                if (layoutSrc.shape(1) != ELE_NUM_PER_C0 || layoutSrc.stride(0) != ELE_NUM_PER_C0) {
                    intriParams.nValue = layoutSrc.shape(0);
                    intriParams.srcDValue = layoutSrc.stride(0);
                    intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                } else {
                    // If the matrix has ELE_NUM_PER_C0, columns and a stride of ELE_NUM_PER_C0, it follows a row-major
                    // layout in L1, allowing the use of standard contiguous DataCopy interface for more efficient
                    // transfers.
                    AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
                }
            } else {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        }
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1DynamicOptimized() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(1) <= 16) {
            // If the number of matrix cols is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            for (int i = 0; i < layoutSrc.shape(1); ++i) {
                AscendC::DataCopyParams dataCopyParams(
                    CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)), layoutDst.shape(0) / ELE_NUM_PER_C0, 0,
                    (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
                AscendC::DataCopy(
                    dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
            }
        } else {
            if (layoutSrc.stride(1) < STRIDE_LIMIT) {
                if (layoutSrc.shape(0) != ELE_NUM_PER_C0 || layoutSrc.stride(1) != ELE_NUM_PER_C0) {
                    intriParams.nValue = layoutSrc.shape(1);
                    intriParams.srcDValue = layoutSrc.stride(1);
                    intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                } else {
                    // If the matrix has ELE_NUM_PER_C0, rows and a stride of ELE_NUM_PER_C0, it follows a col-major
                    // layout in L1, allowing the use of standard contiguous DataCopy interface for more efficient
                    // transfers.
                    AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
                }
            } else {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
                }
            }
        }
    }
};

/// Partial specialization for AtlasA2, zN in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::zN>>
    : public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::zN>> {};

/// Partial specialization for AtlasA2, nZ in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::nZ>>
    : public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::nZ>> {};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};

/// Partial specialization for AtlasA2, PaddingColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1GMMPTD<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1GMMPTD() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(0) == 1) {
            // If the number of matrix rows is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)), layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor, srcTensor, dataCopyParams);
        } else {
            if (layoutSrc.stride(0) < STRIDE_LIMIT) {
                if (layoutSrc.shape(1) != ELE_NUM_PER_C0 || layoutSrc.stride(0) != ELE_NUM_PER_C0) {
                    intriParams.nValue = layoutSrc.shape(0);
                    intriParams.srcDValue = layoutSrc.stride(0);
                    intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                } else {
                    // If the matrix has ELE_NUM_PER_C0 columns and a stride of ELE_NUM_PER_C0, it follows a row-major
                    // layout in L1, allowing the use of the standard contiguous DataCopy interface for more efficient
                    // transfers.
                    AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
                }
            } else {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc, uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride, uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};

////////////////////////////////////////
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
/// Partial specialization for AtlasA2, half, RowMajor in and zN out.
template <>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)), layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingRowMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
template <>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.orgShape(1), layoutDst.shape(2)), layoutDst.shape(2) / ELE_NUM_PER_C0, 0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, ColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template <>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)), layoutDst.shape(0) / ELE_NUM_PER_C0, 0,
                (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template <>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.orgShape(0), layoutDst.shape(0)), layoutDst.shape(0) / ELE_NUM_PER_C0, 0,
                (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(2)], dataCopyParams);
        }
    }
};

/// new add gemm
template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(0);
        uint32_t ndNum = layoutSrc.shape(0) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(0) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(1);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0; //`

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(0);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};
////////////////////////////////////////

///////////////////////////////////////
/// new add gemv, VectorLayout -> zN
template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::VectorLayout>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;
        intriParams.nValue = 1;
        intriParams.srcDValue = layoutSrc.shape(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};





///////////////////////////////////////
/// new add gemv, ColumnMajor -> nN
template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};
/////////////////////////////////

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        // There is an issue with AscenC in the data migration of nd2nz of type int4b_t: dValue needs to be passed
        // with the length of int8, and it is uncertain whether this will be fixed in subsequent versions.
        if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
                intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
            }
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc, uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride, uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};
/////////////////////////////////

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        // There is an issue with AscenC in the data migration of nd2nz of type int4b_t: dValue needs to be passed
        // with the length of int8, and it is uncertain whether this will be fixed in subsequent versions.
        if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
                intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
            }
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

/// Partial specialization for zN in and zN out.
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
        uint32_t blockLen = layoutSrc.orgShape(0);

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for nZ in and nZ out.
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
        uint32_t blockLen = layoutSrc.orgShape(1);

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(1) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(1) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(1);
                uint64_t srcOffset = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(0);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(1);
        intriParams.srcDValue = layoutSrc.stride(2);
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class Element>
struct CopyGmToL1<
    Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride);
                DataCopy(
                    dstTensor[i * MAX_REPEAT * layoutDst.stride(0)], srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)],
                    dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyGmToL1, AtlasA2, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t nValue = tla::get<0>(srcTensor.originShape());
        const uint32_t dValue = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        if (srcDValue < STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * ELE_NUM_PER_C0], srcTensor.data()[srcOffset + i * srcDValue],
                    intriParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t nValue = tla::get<1>(srcTensor.originShape());
        const uint32_t dValue = tla::get<0>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        if (srcDValue < STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * ELE_NUM_PER_C0], srcTensor.data()[srcOffset + i * srcDValue],
                    intriParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcTensor.originShape()));
        uint32_t blockLen = tla::get<0>(srcTensor.originShape());

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopyParams repeatParams;
        if (srcOuterStrideCol / ELE_NUM_PER_C0 <= STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = dstOuterStrideCol / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * dstOuterStrideCol],
                    srcTensor.data()[srcOffset + i * srcOuterStrideCol], repeatParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcTensor.originShape()));
        uint32_t blockLen = tla::get<1>(srcTensor.originShape());

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopyParams repeatParams;
        if (srcOuterStrideRow / ELE_NUM_PER_C0 <= STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = srcOuterStrideRow / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = dstOuterStrideRow / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                    srcTensor.data()[srcOffset + i * srcOuterStrideRow], repeatParams);
            }
        }
    }
};

///////////////////////////////////////////TileCopyTlaExt//////////////////////////////////////////////////////
/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingRowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>, layout::RowMajor,
    layout::zN> {
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, ActualShape actualShape)
    {
        

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<1>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<0>(actualShape);
        intriParams.srcDValue = tla::get<0>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingRowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>, layout::PaddingRowMajor,
    layout::zN> {
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, ActualShape actualShape)
    {
        
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<1>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<0>(actualShape);
        intriParams.srcDValue = tla::get<0, 0>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>, layout::ColumnMajor,
    layout::nZ> {
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, ActualShape actualShape)
    {
        

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<0>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<1>(actualShape);
        intriParams.srcDValue = tla::get<1>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<1, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, PaddingColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    layout::PaddingColumnMajor, layout::nZ> {
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, ActualShape actualShape)
    {
        

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = tla::get<0>(actualShape);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = tla::get<1>(actualShape);
        intriParams.srcDValue = tla::get<1, 0>(srcTensor.stride());
        intriParams.dstNzNStride = tla::get<1, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>, layout::zN,
    layout::zN> {
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, ActualShape actualShape)
    {
        

        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(actualShape));
        uint32_t blockLen = tla::get<0>(actualShape);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopyParams repeatParams;
        if (srcOuterStrideCol / ELE_NUM_PER_C0 <= STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = dstOuterStrideCol / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * dstOuterStrideCol],
                    srcTensor.data()[srcOffset + i * srcOuterStrideCol], repeatParams);
            }
        }
    }
};

/// Partial specialization for TileCopyTlaExt, CopyGmToL1, AtlasA2, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>, layout::nZ,
    layout::nZ> {
    using ActualShape = tla::Shape<uint32_t, uint32_t>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, ActualShape actualShape)
    {
        

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(actualShape));
        uint32_t blockLen = tla::get<1>(actualShape);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopyParams repeatParams;
        if (srcOuterStrideRow / ELE_NUM_PER_C0 <= STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = srcOuterStrideRow / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = dstOuterStrideRow / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                    srcTensor.data()[srcOffset + i * srcOuterStrideRow], repeatParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

// For copying fused per channel / group dequant scale from GM to L1.
// Scale data is stored as RowMajor in GM and loaded as VectorLayout in L1.
template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag, Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::GM>,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv<ELE_NUM_PER_C0>(layoutDst.shape(0));
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, RowMajor or ColumnMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopySparseTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        (tla::detail::isRowMajor<LayoutSrc>::value || tla::detail::isColumnMajor<LayoutSrc>::value) &&
        tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopySparseTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        auto srcShape = srcTensor.shape();
        auto srcStride = srcTensor.stride();
        auto dstShape = dstTensor.shape();
        auto coord = srcTensor.coord();

        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = tla::min<int, int>(
            tla::get<0, 0>(dstShape) * tla::get<0, 1>(dstShape), tla::get<0>(srcShape) - tla::get<0>(coord));
        nd2nzParams.dValue = tla::min<int, int>(
            tla::get<1, 0>(dstShape) * tla::get<1, 1>(dstShape), tla::get<1>(srcShape) - tla::get<1>(coord));
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = tla::get<0>(srcStride);
        nd2nzParams.dstNzC0Stride = tla::get<0, 0>(dstShape) * tla::get<0, 1>(dstShape);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        auto offset = srcTensor.layout()(coord);

        AscendC::DataCopy(dstTensor.data(), srcTensor.data()[offset], nd2nzParams);
    }
};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopySparseTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopySparseTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        
        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        if (srcDValue < STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * ELE_NUM_PER_C0], srcTensor.data()[srcOffset + i * srcDValue],
                    intriParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, sparse zn in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopySparseTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<uint32_t, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopySparseTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        using TransT = typename TensorSrc::Element;

        auto srcShape = srcTensor.shape();
        auto dstShape = dstTensor.shape();
        auto coord = srcTensor.coord();
        int32_t alignedGRow = tla::get<0, 0>(srcShape) * tla::get<0, 1>(srcShape);
        // height direction need to be 16 aligned
        auto alignHeight = tla::get<0, 0>(dstShape) * tla::get<0, 1>(dstShape);
        int32_t blockLen = alignHeight * tla::get<1, 0>(dstShape) * sizeof(TransT) / BYTE_PER_C0;
        int32_t srcStride = (alignedGRow - alignHeight) * tla::get<1, 0>(srcShape) * sizeof(TransT) / BYTE_PER_C0;
        uint16_t nburst = tla::get<1, 1>(dstShape);
        auto offset = srcTensor.layout()(coord);
        AscendC::DataCopy(
            dstTensor.data(), srcTensor.data()[offset],
            {nburst, static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride), static_cast<uint16_t>(0)});
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, sparse nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopySparseTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isnZ<uint32_t, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopySparseTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        using TransT = typename TensorSrc::Element;
        auto srcShape = srcTensor.shape();
        auto dstShape = dstTensor.shape();
        auto coord = srcTensor.coord();
        int32_t alignedGCol = tla::get<1, 0>(srcShape) * tla::get<1, 1>(srcShape);
        auto alignCol = tla::get<1, 0>(dstShape) * tla::get<1, 1>(dstShape);
        int32_t blockLen = alignCol * tla::get<0, 0>(dstShape) * sizeof(TransT) / BYTE_PER_C0;
        int32_t srcStride = (alignedGCol - alignCol) * tla::get<0, 0>(srcShape) * sizeof(TransT) / BYTE_PER_C0;
        uint16_t nburst = tla::get<0, 1>(dstShape);
        auto offset = srcTensor.layout()(coord);
        AscendC::DataCopy(
            dstTensor.data(), srcTensor.data()[offset],
            {nburst, static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride), static_cast<uint16_t>(0)});
    }
};

///////////////////////////////////////////TileCopyFAQTla//////////////////////////////////////////////////////
/// used for FA LoadQ, multi matrix datacopy ND->NZ
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyFAQTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyFAQTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<2>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t ndNum = tla::get<0>(srcTensor.shape());
        const uint32_t srcNdMatrixStride = tla::get<0>(srcTensor.stride());
        const uint32_t dstNzNStride = tla::get<0>(srcTensor.shape());
        const uint32_t dstNzMatrixStride = ELE_NUM_PER_C0;
        const uint32_t dstNzC0Stride = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(
                    dstTensor.data()[dstOffset + i * ELE_NUM_PER_C0], srcTensor.data()[srcOffset + i * srcDValue],
                    intriParams);
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp



// END vendored catlass/gemm/tile/copy_gm_to_l1.hpp


// BEGIN vendored catlass/gemm/tile/copy_gm_to_ub.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_gm_to_ub.hpp
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
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        

        const uint16_t row = tla::get<0>(srcTensor.originShape());
        const uint16_t col = tla::get<1>(srcTensor.originShape());

        AscendC::DataCopyExtParams dataCopyParams(
            row, col * sizeof(ElementSrc), (tla::get<0>(srcTensor.stride()) - col) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - col) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    };
};

template <class ArchTag, class GmType>
struct CopyGm2Ub {
    
};

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::VectorLayout const& layoutDst, layout::VectorLayout const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, layoutSrc.shape(0) * sizeof(Element), 0, 0, 0);
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

} // namespace Catlass::Gemm::Tile


// END vendored catlass/gemm/tile/atlasa2/copy_gm_to_ub.hpp



// END vendored catlass/gemm/tile/copy_gm_to_ub.hpp


// BEGIN vendored catlass/gemm/tile/copy_l0c_to_gm.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// BEGIN vendored catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp
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
// already embedded: catlass/gemm/gemm_type.hpp
// already embedded: tla/tensor.hpp

namespace Catlass::Gemm::Tile {

enum class ScaleGranularity
{
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag, class ElementSrc, class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    
};

// CopyL0CToGm fp32 to fp32
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, float, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, half, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, half, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, bfloat16_t, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, bfloat16_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

// CopyL0CToGm cast float to uint8/int8
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, uint8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::QF322B8_PRE;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, uint8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VQF322B8_PRE;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, int8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::QF322B8_PRE;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, int8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VQF322B8_PRE;
};

// CopyL0CToGm output int32
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, int32_t, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm cast int32_t to fp16
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, half, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, half, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

// CopyL0CToGm cast int32 to uint8/int8
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, uint8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, int8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int32_t, int8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <
    class ArchTag, class ElementAccumulator, class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGm {
    
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::NO_QUANT, ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const& params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& dst, AscendC::LocalTensor<ElementSrc> const& src,
        LayoutDst const& dstLayout, LayoutSrc const& srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::PER_TENSOR, ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(float scalar)
        {
            scale = scalar;
        }
    };
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const& params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& dst, AscendC::LocalTensor<ElementSrc> const& src,
        LayoutDst const& dstLayout, LayoutSrc const& srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const& params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& dst, AscendC::LocalTensor<ElementSrc> const& src,
        AscendC::LocalTensor<uint64_t> const& scale, LayoutDst const& dstLayout, LayoutSrc const& srcLayout,
        uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::SetFixPipeConfig<uint64_t, false>(scale, false);
        AscendC::PipeBarrier<PIPE_FIX>();
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::zN>, ScaleGranularity::NO_QUANT,
    ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const& params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& dst, AscendC::LocalTensor<ElementSrc> const& src,
        LayoutDst const& dstLayout, LayoutSrc const& srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.isChannelSplit = true;
        }

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};



///////////////////////////////////////////CopyL0CToGmTla/////////////////////////////////////////////////
// L0C copy mode
struct CopyToGM {};
struct CopyToL1 {};

template <
    class ArchTag, class TensorSrc, class TensorDst, ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false, class Enable = void>
struct CopyL0CToGmTla {
    
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::AtlasA2, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        

        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::AtlasA2, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

