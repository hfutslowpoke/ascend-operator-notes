// BEGIN vendored catlass/arch/arch.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// BEGIN vendored catlass/catlass.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstdint>

#if defined(__CCE__)
#include <kernel_operator.h>
#endif

// BEGIN vendored catlass/detail/alignment.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// BEGIN vendored catlass/detail/macros.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#if defined(__CCE__)
#include <kernel_operator.h>
#endif

#define CATLASS_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
#define CATLASS_HOST_DEVICE __forceinline__[host, aicore]
#else
#define CATLASS_HOST_DEVICE
#endif
#define CATLASS_GLOBAL __global__ __aicore__


// END vendored catlass/detail/macros.hpp


// BEGIN vendored tla/integral/integral_constant.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/detail/macros.hpp

// BEGIN vendored tla/utils/type_traits.hpp
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

#include <cstddef>
#include <type_traits>
#include <tuple>

// SFINAE helper macros for template constraints.
// TLA_REQUIRES: default template parameter (pointer form).
// TLA_REQUIRES_T: type form for non-type template parameters.
#define TLA_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr
#define TLA_REQUIRES_T(...) typename std::enable_if<(__VA_ARGS__)>::type

namespace tla {

// ---------------------------------------------------------------------------
// Re-exports from <type_traits>
// ---------------------------------------------------------------------------

using std::enable_if;
using std::enable_if_t;
using std::void_t;

using std::is_arithmetic;
using std::is_arithmetic_v;

using std::is_empty;
using std::is_empty_v;

using std::is_same;
using std::is_same_v;

using std::is_signed;
using std::is_signed_v;
using std::is_unsigned;
using std::is_unsigned_v;

// std::is_integral intentionally not re-exported: tla::is_integral (integral_constant.hpp) shadows it.
template <class T>
using is_std_integral = std::is_integral<T>;

// ---------------------------------------------------------------------------
// Re-exports from <utility>: declval, forward, move
// ---------------------------------------------------------------------------

template <class T>
CATLASS_HOST_DEVICE std::add_rvalue_reference_t<T> declval() noexcept;

template <class T>
CATLASS_HOST_DEVICE constexpr T&& forward(std::remove_reference_t<T>& arg) noexcept
{
    return static_cast<T&&>(arg);
}

template <class T>
CATLASS_HOST_DEVICE constexpr T&& forward(std::remove_reference_t<T>&& arg) noexcept
{
    
    return static_cast<T&&>(arg);
}

template <class T>
CATLASS_HOST_DEVICE constexpr std::remove_reference_t<T>&& move(T&& arg) noexcept
{
    return static_cast<std::remove_reference_t<T>&&>(arg);
}

// ---------------------------------------------------------------------------
// remove_cvref (C++20 polyfill for C++17)
// ---------------------------------------------------------------------------

template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

// ---------------------------------------------------------------------------
// tuple_size / same_tuple_size: tla-local wrappers around std::tuple_size
// ---------------------------------------------------------------------------

template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T, std::void_t<typename std::tuple_size<T>::type>>
    : std::integral_constant<std::size_t, std::tuple_size<T>::value> {};

template <class T>
inline constexpr std::size_t tuple_size_v = tuple_size<T>::value;

// same_tuple_size: true if all Ts have the same tuple_size as T0
template <class T0, class... Ts>
struct same_tuple_size : std::bool_constant<((tuple_size<Ts>::value == tuple_size<T0>::value) && ...)> {};

template <class... Ts>
inline constexpr bool same_tuple_size_v = same_tuple_size<Ts...>::value;

// ---------------------------------------------------------------------------
// tuple_element / tuple_element_t: tla-local wrapper around std::tuple_element
// ---------------------------------------------------------------------------

template <size_t I, class T, class = void>
struct tuple_element;

template <size_t I, class T>
struct tuple_element<I, T, std::void_t<typename std::tuple_element<I, T>::type>> : std::tuple_element<I, T> {};

template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

// ---------------------------------------------------------------------------
// dependent_false: always-false value dependent on template parameters
// ---------------------------------------------------------------------------

template <class... Args>
inline constexpr bool dependent_false = false;

} // namespace tla

#define TLA_ASSERT_SAME_TUPLE_SIZE(...)


// END vendored tla/utils/type_traits.hpp


namespace tla {

// Base type: std-conforming integral constant with explicit type enforcement
template <class T, T v>
struct integral_constant {
    using type = integral_constant<T, v>;
    static constexpr T value = v;
    using value_type = T;
    CATLASS_HOST_DEVICE constexpr operator value_type() const noexcept
    {
        return value;
    }
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept
    {
        return value;
    }
};

// Simplified aliases: C (auto deduction), Bool (bool), Int (int), Int64 (long long)
template <auto v>
using C = integral_constant<decltype(v), v>;

// Legacy alias; prefer integral_constant or C in new code.
template <class T, T v>
using constant = C<v>;

template <bool b>
using Bool = C<b>;

template <bool b>
using bool_constant = Bool<b>;

template <int v>
using Int = C<v>;

template <long long v>
using Int64 = C<v>;

// Bool type aliases
using true_type = Bool<true>;
using false_type = Bool<false>;

// Int type aliases
#define TLA_INT_ALIAS(N) \
    using _##N = Int<N>; \
    using _##N##LL = Int64<N##LL>

TLA_INT_ALIAS(0);
TLA_INT_ALIAS(1);
TLA_INT_ALIAS(2);
TLA_INT_ALIAS(3);
TLA_INT_ALIAS(4);
TLA_INT_ALIAS(5);
TLA_INT_ALIAS(6);
TLA_INT_ALIAS(7);
TLA_INT_ALIAS(8);
TLA_INT_ALIAS(9);
TLA_INT_ALIAS(10);
TLA_INT_ALIAS(12);
TLA_INT_ALIAS(16);
TLA_INT_ALIAS(24);
TLA_INT_ALIAS(32);
TLA_INT_ALIAS(48);
TLA_INT_ALIAS(64);
TLA_INT_ALIAS(96);
TLA_INT_ALIAS(128);
TLA_INT_ALIAS(192);
TLA_INT_ALIAS(256);
TLA_INT_ALIAS(384);
TLA_INT_ALIAS(512);
TLA_INT_ALIAS(768);
TLA_INT_ALIAS(1024);
TLA_INT_ALIAS(2048);
TLA_INT_ALIAS(4096);
TLA_INT_ALIAS(8192);
TLA_INT_ALIAS(16384);
TLA_INT_ALIAS(32768);
TLA_INT_ALIAS(65536);

#undef TLA_INT_ALIAS

// Type traits: classify and test integral constant types
template <class T>
struct is_integral : Bool<is_std_integral<T>::value> {};
template <class T, T v>
struct is_integral<integral_constant<T, v>> : true_type {};
template <class T>
inline constexpr bool is_integral_v = is_integral<T>::value;

template <class T>
struct is_static : Bool<is_empty_v<remove_cvref_t<T>>> {};
template <class T>
inline constexpr bool is_static_v = is_static<T>::value;

template <auto n, class T>
struct is_constant_base : false_type {};
template <auto n, class T, T v>
struct is_constant_base<n, integral_constant<T, v>> : Bool<v == n> {};
template <auto n, class T>
struct is_constant : is_constant_base<n, remove_cvref_t<T>> {};
template <auto n, class T>
inline constexpr bool is_constant_v = is_constant<n, T>::value;

// Underscore placeholder: empty tag for "take the whole dimension" in Coord/tensor indexing
struct Underscore {
    using type = Underscore;
};

template <>
struct is_integral<Underscore> : true_type {};

inline constexpr Underscore _{};

template <class T>
struct is_underscore : Bool<is_same_v<remove_cvref_t<T>, Underscore>> {};
template <class T>
inline constexpr bool is_underscore_v = is_underscore<T>::value;

} // namespace tla

#define TLA_STATIC_CHECK(...)


// END vendored tla/integral/integral_constant.hpp


// BEGIN vendored tla/integral/integral_math.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/detail/macros.hpp
// already embedded: tla/integral/integral_constant.hpp

// BEGIN vendored tla/utils/math.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: tla/utils/type_traits.hpp
// already embedded: catlass/detail/macros.hpp

namespace tla {

template <class T, class U, TLA_REQUIRES((is_arithmetic<T>::value) && (is_arithmetic<U>::value))>
CATLASS_HOST_DEVICE constexpr auto max(T const& t, U const& u)
{
    return t < u ? u : t;
}

template <class T, class U, TLA_REQUIRES((is_arithmetic<T>::value) && (is_arithmetic<U>::value))>
CATLASS_HOST_DEVICE constexpr auto min(T const& t, U const& u)
{
    return t < u ? t : u;
}

template <class T, class U, TLA_REQUIRES((is_std_integral<T>::value) && (is_std_integral<U>::value))>
CATLASS_HOST_DEVICE constexpr auto clip_sub(T const& t, U const& u)
{
    return (t > u) ? (t - u) : T(0);
}

template <class T, class U, TLA_REQUIRES((is_std_integral<T>::value) && (is_std_integral<U>::value))>
CATLASS_HOST_DEVICE constexpr auto ceil_div(T const& a, U const& b)
{
    return (a + b - T(1)) / b;
}

template <class T, class U, TLA_REQUIRES((is_std_integral<T>::value) && (is_std_integral<U>::value))>
CATLASS_HOST_DEVICE constexpr auto round_up(T const& t, U const& u)
{
    return ceil_div(t, u) * u;
}

template <class T, TLA_REQUIRES(is_arithmetic<T>::value)>
CATLASS_HOST_DEVICE constexpr auto abs(T const& t)
{
    if constexpr (is_signed<T>::value) {
        return t < T(0) ? -t : t;
    } else {
        return t;
    }
}

} // namespace tla


// END vendored tla/utils/math.hpp

// already embedded: tla/utils/type_traits.hpp

namespace tla {

/***************/
/** Operators **/
/***************/

#define TLA_LEFT_UNARY_OP(OP)                                 \
    template <auto t>                                         \
    CATLASS_HOST_DEVICE constexpr C<(OP t)> operator OP(C<t>) \
    {                                                         \
        return {};                                            \
    }
#define TLA_BINARY_OP(OP)                                             \
    template <auto t, auto u>                                         \
    CATLASS_HOST_DEVICE constexpr C<(t OP u)> operator OP(C<t>, C<u>) \
    {                                                                 \
        return {};                                                    \
    }

TLA_LEFT_UNARY_OP(+);
TLA_LEFT_UNARY_OP(-);
TLA_LEFT_UNARY_OP(~);
TLA_LEFT_UNARY_OP(!);
TLA_LEFT_UNARY_OP(*);

TLA_BINARY_OP(+);
TLA_BINARY_OP(-);
TLA_BINARY_OP(*);
TLA_BINARY_OP(/);
TLA_BINARY_OP(%);
TLA_BINARY_OP(&);
TLA_BINARY_OP(|);
TLA_BINARY_OP(^);
TLA_BINARY_OP(<<);
TLA_BINARY_OP(>>);
// Logical overloads propagate static results; operands do not short-circuit.
TLA_BINARY_OP(&&);
TLA_BINARY_OP(||);
TLA_BINARY_OP(==);
TLA_BINARY_OP(!=);
TLA_BINARY_OP(<);
TLA_BINARY_OP(>);
TLA_BINARY_OP(<=);
TLA_BINARY_OP(>=);

#undef TLA_BINARY_OP
#undef TLA_LEFT_UNARY_OP

//
// Mixed static-dynamic special cases
// When one operand is C<t> and the other is a runtime integral,
// the result is statically known for certain values of t.
//

// 0 * x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator*(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator*(U, C<t>)
{
    return {};
}

// 0 / x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator/(C<t>, U)
{
    return {};
}

// x % (1 or -1) = 0
template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (t == 1 || t == -1))>
CATLASS_HOST_DEVICE constexpr C<0> operator%(U, C<t>)
{
    return {};
}

// 0 % x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator%(C<t>, U)
{
    return {};
}

// 0 & x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator&(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator&(U, C<t>)
{
    return {};
}

// Like constant-constant overloads, mixed logical overloads do not short-circuit.
// false && x = false
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (!bool(t)))>
CATLASS_HOST_DEVICE constexpr C<false> operator&&(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (!bool(t)))>
CATLASS_HOST_DEVICE constexpr C<false> operator&&(U, C<t>)
{
    return {};
}

// true || x = true
template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (bool(t)))>
CATLASS_HOST_DEVICE constexpr C<true> operator||(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (bool(t)))>
CATLASS_HOST_DEVICE constexpr C<true> operator||(U, C<t>)
{
    return {};
}

//
// Named functions from math.hpp
//

#define TLA_NAMED_BINARY_FN(OP)                                         \
    template <auto t, auto u>                                           \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>, C<u>)                   \
    {                                                                   \
        return C<OP(t, u)>{};                                           \
    }                                                                   \
    template <auto t, class U, TLA_REQUIRES(is_std_integral<U>::value)> \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>, U u)                    \
    {                                                                   \
        return OP(t, u);                                                \
    }                                                                   \
    template <class T, auto u, TLA_REQUIRES(is_std_integral<T>::value)> \
    CATLASS_HOST_DEVICE constexpr auto OP(T t, C<u>)                    \
    {                                                                   \
        return OP(t, u);                                                \
    }

TLA_NAMED_BINARY_FN(max);
TLA_NAMED_BINARY_FN(min);
TLA_NAMED_BINARY_FN(clip_sub);
TLA_NAMED_BINARY_FN(ceil_div);
TLA_NAMED_BINARY_FN(round_up);

#undef TLA_NAMED_BINARY_FN

// Variadic max/min: 2+args recursive, delegates to binary overloads.
// SFINAE on is_integral<T0> avoids conflict with tuple-aware overloads.
template <class T0, class T1, class... Ts, TLA_REQUIRES(is_integral<remove_cvref_t<T0>>::value)>
CATLASS_HOST_DEVICE constexpr auto max(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    return max(max(t0, t1), ts...);
}

template <class T0, class T1, class... Ts, TLA_REQUIRES(is_integral<remove_cvref_t<T0>>::value)>
CATLASS_HOST_DEVICE constexpr auto min(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    return min(min(t0, t1), ts...);
}

#define TLA_NAMED_UNARY_FN(OP)                  \
    template <auto t>                           \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>) \
    {                                           \
        return C<OP(t)>{};                      \
    }

TLA_NAMED_UNARY_FN(abs);

#undef TLA_NAMED_UNARY_FN

// conditional_return: compile-time conditional that preserves static-ness.
// When both branches are the same C<v>, returns C<v> (no runtime cost).
// When branches differ, falls back to runtime ternary.
template <class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr decltype(auto) conditional_return(true_type, TrueType&& t, FalseType&&)
{
    return tla::forward<TrueType>(t);
}

template <class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr decltype(auto) conditional_return(false_type, TrueType&&, FalseType&& f)
{
    return tla::forward<FalseType>(f);
}

template <auto v>
CATLASS_HOST_DEVICE constexpr auto conditional_return(bool, C<v> const&, C<v> const&)
{
    return C<v>{};
}

template <auto v, auto u>
CATLASS_HOST_DEVICE constexpr auto conditional_return(bool b, C<v> const&, C<u> const&)
{
    return b ? v : u;
}

template <class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr auto conditional_return(bool b, TrueType const& t, FalseType const& f)
{
    return b ? t : f;
}

template <bool b, class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr auto conditional_return(TrueType const& t, FalseType const& f)
{
    if constexpr (b) {
        return t;
    } else {
        return f;
    }
}

} // namespace tla


// END vendored tla/integral/integral_math.hpp


template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE constexpr T RoundUp(const T& val)
{
    
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto RoundUp(T const& val, U const& align)
{
    if constexpr (tla::is_static<T>::value && tla::is_static<U>::value) { // Int, Int
        constexpr uint32_t res = (T::value + U::value - 1) / U::value * U::value;
        return tla::Int<res>{};
    } else if constexpr (tla::is_static<T>::value) { // Int, int
        return (T::value + align - 1) / align * align;
    } else if constexpr (tla::is_static<U>::value) { // int, Int
        return (val + U::value - 1) / U::value * U::value;
    } else { // int, int
        return (val + align - 1) / align * align;
    }
}

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE constexpr T RoundDown(const T val)
{
    
    return val / ALIGN * ALIGN;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto RoundDown(T const& val, U const& align)
{
    if constexpr (tla::is_static<T>::value && tla::is_static<U>::value) { // Int, Int
        constexpr uint32_t res = T::value / U::value * U::value;
        return tla::Int<res>{};
    } else if constexpr (tla::is_static<T>::value) { // Int, int
        return T::value / align * align;
    } else if constexpr (tla::is_static<U>::value) { // int, Int
        return val / U::value * U::value;
    } else { // int, int
        return val / align * align;
    }
}

template <uint32_t DIVISOR, typename T>
CATLASS_HOST_DEVICE constexpr T CeilDiv(const T dividend)
{
    
    return (dividend + DIVISOR - 1) / DIVISOR;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto CeilDiv(T const& dividend, U const& divisor)
{
    if constexpr (tla::is_static<T>::value && tla::is_static<U>::value) { // Int, Int
        constexpr uint32_t res = (T::value + U::value - 1) / U::value;
        return tla::Int<res>{};
    } else if constexpr (tla::is_static<T>::value) { // Int, int
        return (T::value + divisor - 1) / divisor;
    } else if constexpr (tla::is_static<U>::value) { // int, Int
        return (dividend + U::value - 1) / U::value;
    } else { // int, int
        return (dividend + divisor - 1) / divisor;
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto Max(T const& a, U const& b)
{
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto Min(T const& a, U const& b)
{
    if (a < b) {
        return a;
    } else {
        return b;
    }
}


// END vendored catlass/detail/alignment.hpp


// BEGIN vendored catlass/detail/dependent_false.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;


// END vendored catlass/detail/dependent_false.hpp

// already embedded: catlass/detail/macros.hpp

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

constexpr uint32_t BYTE_PER_BLK_FP = 128; /// datablock size of A1->C2PiPE2GM

class EmptyClass {};

} // namespace Catlass


// END vendored catlass/catlass.hpp


namespace Catlass::Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

// Ascend950
struct Ascend950 {
    static constexpr uint32_t BIAS_SIZE = 4 * 1024;
    static constexpr uint32_t FIXBUF_SIZE = 16 * 1024;
    static constexpr uint32_t UB_SIZE = 248 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 256 * 1024;
};

template <AscendC::TPosition POS>
using PositionType = std::integral_constant<AscendC::TPosition, POS>;

using PositionGM = PositionType<AscendC::TPosition::GM>;
using PositionL1 = PositionType<AscendC::TPosition::A1>;
using PositionL0A = PositionType<AscendC::TPosition::A2>;
using PositionL0B = PositionType<AscendC::TPosition::B2>;
using PositionL0C = PositionType<AscendC::TPosition::CO1>;
using PositionBias = PositionType<AscendC::TPosition::C2>;
using PositionUB = PositionType<AscendC::TPosition::VECCALC>;

} // namespace Catlass::Arch


// END vendored catlass/arch/arch.hpp


// BEGIN vendored catlass/arch/resource.hpp
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

// BEGIN vendored catlass/arch/local_tensor_buffer.hpp
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

namespace Catlass::Arch {

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    CATLASS_DEVICE AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE
    LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
    
};

/// Partial specialization for TPosition::A1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
        GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
        tensor = tbufA1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
        GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
        tensor = tbufA2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B1> tbufB1;
        GetTPipePtr()->InitBuffer(tbufB1, ArchTag::L1_SIZE);
        tensor = tbufB1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
        GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
        tensor = tbufB2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C1> tbufC1;
        GetTPipePtr()->InitBuffer(tbufC1, ArchTag::L1_SIZE);
        tensor = tbufC1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
        GetTPipePtr()->InitBuffer(tbufC2, ArchTag::BIAS_SIZE);
        tensor = tbufC2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
        GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
        tensor = tbufCO1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2PIPE2GM
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2PIPE2GM;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2PIPE2GM> tbufC2PIPE2GM;
        GetTPipePtr()->InitBuffer(tbufC2PIPE2GM, ArchTag::FIXBUF_SIZE);
        tensor = tbufC2PIPE2GM.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECIN
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECIN> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECIN;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECIN> tbufVECIN;
        GetTPipePtr()->InitBuffer(tbufVECIN, ArchTag::UB_SIZE);
        tensor = tbufVECIN.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECOUT
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECOUT> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECOUT;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECOUT> tbufVECOUT;
        GetTPipePtr()->InitBuffer(tbufVECOUT, ArchTag::UB_SIZE);
        tensor = tbufVECOUT.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECCALC
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};

} // namespace Catlass::Arch


// END vendored catlass/arch/local_tensor_buffer.hpp


namespace Catlass::Arch {

template <class ArchTag>
struct Resource {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> fpBuf;

    CATLASS_DEVICE
    Resource()
    {
        // The initialization of AscendC::Tpipe will insert some synchronization interfaces,
        // which may conflict with the usage by users. Therefore, the "destroy" interface is used for releasing.
        pipe.Destroy();
    }
};

} // namespace Catlass::Arch


// END vendored catlass/arch/resource.hpp


// BEGIN vendored catlass/gemm/gemm_type.hpp
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

namespace Catlass::Gemm {

////////////////////////////////////////////////////////////////////

template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = POSITION_;
};

} // namespace Catlass::Gemm


// END vendored catlass/gemm/gemm_type.hpp


// BEGIN vendored catlass/gemm/dispatch_policy.hpp
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

namespace Catlass::Gemm {

// Block Mmad Policies

template <class ArchTag_, bool ASYNC_ = false>
struct MmadBase {
    using ArchTag = ArchTag_;
    static constexpr uint32_t ASYNC = ASYNC_;
};

template <class BaseDispatchPolicy_>
struct MmadDualSharedA : public BaseDispatchPolicy_ {
    using BaseDispatchPolicy = BaseDispatchPolicy_;
    using ArchTag = typename BaseDispatchPolicy::ArchTag;
    static constexpr bool USE_FULL_LOAD_A = BaseDispatchPolicy::USE_FULL_LOAD_A;
};

using MmadAtlasA2 = MmadBase<Arch::AtlasA2, false>;
using MmadAtlasA2Async = MmadBase<Arch::AtlasA2, true>;

// Now ENABLE_UNIT_FLAG_ must be false when intput element is int8
template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingPongWithPrologue : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2SingleCoreSplitk : public MmadAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongSliceKWithPrologue : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Preload : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

struct MmadAtlasA2UnsharedFAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2UnsharedFAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAQKTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAPVTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2AMLAPVTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsync : public MmadAtlasA2Async {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_; // Stages of emitting load instruction in advance
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsyncWithCallback
    : public MmadAtlasA2PreloadAsync<
          PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2DynamicPreloadAsyncWithCallback
    : public MmadAtlasA2PreloadAsync<
          PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2W4A4MatmulPerTokenPerChannelDequant
    : public MmadAtlasA2PreloadAsyncWithCallback<
          PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};
////////////////////
// new add
template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false, bool ENABLE_ABBA_ = false>
struct GemmAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool ENABLE_ABBA = ENABLE_ABBA_;
};

struct GemvAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};
////////////////////

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongBias : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongSymmLeft : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongSymmRight : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2XFAIQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2XFAIPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAITailQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAITailPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIQKSplitRow : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIPVSplitRow : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FullLoadA : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false, bool USE_HF32_MODE_ = false,
    uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 1, uint32_t L1B_STAGES_ = 2,
    uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadFullLoadA : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 1, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadAscend950FullLoadA : public MmadBase<ArchTag_, false> {
    static constexpr bool USE_FULL_LOAD_A = true;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, uint32_t OUTPUT_UB_STAGES_, bool AIV_GATHER_, uint32_t MAX_GATHER_K_>
struct MatmulGatherScatterDispatchPolicy : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t OUTPUT_UB_STAGES = OUTPUT_UB_STAGES_;
    static constexpr bool AIV_GATHER = AIV_GATHER_;
    static constexpr uint32_t MAX_GATHER_K = MAX_GATHER_K_;
    
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2W8A16 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicCommon : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2DynamicSmall : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Streamk : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicStreamk : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2DynamicSingleCoreSplitk : public MmadAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    
};

template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256, uint32_t STAGES_ = 2>
struct MmadAtlasA2DynamicAiv : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
};

// UB must have a capacity of UBA(mTile) + UBB(nTile) + UBC(mTile, nTile) size.
// MmadAtlasA2AivSimple is suitable for cases: mTile <= mTileUbLimits && nTile >= params.n
template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256, bool IS_TILE_M_ = true>
struct MmadAtlasA2DynamicAivSimple : public MmadAtlasA2 {
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
    static constexpr bool IS_TILE_M = IS_TILE_M_;
};

// mTile and nTile must be aligned to 16.
// UB must have a capacity of UBA(mTile) + UBB(nTile) + 2 * UBC(mTile, nTile) size.
// MmadAtlasA2AivTrans is suitable for cases: M > N
template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256>
struct MmadAtlasA2DynamicAivTrans : public MmadAtlasA2 {
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Small : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};
// Now ENABLE_UNIT_FLAG_ must be false when intput element is int8
template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadPingpong : public MmadBase<ArchTag_, false> {
    static constexpr bool USE_FULL_LOAD_A = false;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadPingpongMutex : public MmadBase<ArchTag_, false> {
    
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadPingpongPreLoad : public MmadBase<ArchTag_, true> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadPingpongSymmLeft : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadPingpongSymmRight : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 2>
struct MmadMultiBatch : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
};

template <
    class ArchTag_, uint32_t PRELOAD_STAGES_, uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_,
    bool USE_HF32_MODE_ = false, bool ENABLE_L1_RESIDENT_ = false>
struct MmadPreloadAsyncWithCallback : public MmadBase<ArchTag_, true> {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <
    class ArchTag_, uint32_t PRELOAD_STAGES_, uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_,
    bool USE_HF32_MODE_ = false, bool ENABLE_L1_RESIDENT_ = false>
struct MmadPreloadAsyncWithCallbackL0CToUB : public MmadBase<ArchTag_, true> {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
    static constexpr bool ENABLE_DYNAMIC_TILE = true;
};

// 基于TLA提供block层不感知尾块逻辑，tile层感知originShape的流程。使用TileView与MakeTensorLike。
template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadPingpongTlaV2 : public MmadBase<ArchTag_, ENABLE_UNIT_FLAG_> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct SparseMatmulMultiBlockOnKAxis : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAITailQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIQKMx : public MmadFAIQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_> {};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAITailPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIPVMx : public MmadFAIPV<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_> {};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L1_SCALE_FACTOR_K_ = 16, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadMx : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
    /// GM→L1 MX scale 一次驻留的 L1 K 条带个数。设为 1 时每条 L1 K 条带各搬一次 scale。
    static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_L1_RESIDENT_ = false, uint32_t L1_SCALE_FACTOR_K_ = 1,
    uint32_t L0C_STAGES_ = 1, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 1, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadA8W4Mx : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
    static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
};

template <class ArchTag_, uint32_t L1B_STAGES_ = 2>
struct MxA8W4Prologue : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadDequant : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadPingpongPertile : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false,
    uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadSvd1 : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};
template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false,
    uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadSvd2 : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
template <
    class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L1_SCALE_FACTOR_K_ = 8, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadSvd3 : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
};

template <class ArchTag_>
struct MmadAtlasA5RainQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L0_STAGES = 2;
    // Q: persistent in L1
    static constexpr uint32_t L1A_STAGES = 1;
    // K: double buffer
    static constexpr uint32_t L1B_STAGES = 2;
};

template <class ArchTag_>
struct MmadAtlasA5RainPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L0_STAGES = 2;
    // P: PRE_LAUNCH+1 pipeline buffers
    static constexpr uint32_t L1A_STAGES = 3;
    // V: double buffer
    static constexpr uint32_t L1B_STAGES = 2;
};

template <
    class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool NZ_LAYOUT_FLAG_ = false, bool PA_BNNBSD_FLAG_ = 0,
    bool ENABLE_DN_ = false>
struct MmadFlashAttentionQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t PA_BNNBSD_FLAG = PA_BNNBSD_FLAG_;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool NZ_LAYOUT_FLAG = NZ_LAYOUT_FLAG_;
    static constexpr bool ENABLE_DN = ENABLE_DN_;
};

template <
    class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool NZ_LAYOUT_FLAG_ = false, bool PA_BNNBSD_FLAG_ = 0,
    bool ENABLE_DN_ = false>
struct MmadFlashAttentionPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t PA_BNNBSD_FLAG = PA_BNNBSD_FLAG_;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool NZ_LAYOUT_FLAG = NZ_LAYOUT_FLAG_;
    static constexpr bool ENABLE_DN = ENABLE_DN_;
};

template <class ArchTag_, bool ENABLE_SHUFFLE_K_ = false>
struct MmadPlanarComplexFused : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <class ArchTag_>
struct MmadXASharedQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

template <class ArchTag_>
struct MmadXAUnsharedQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

template <class ArchTag_>
struct MmadXAUnsharedPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

template <class ArchTag_>
struct MmadXASharedPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

template <class ArchTag_, bool ENABLE_PAGED_KV_CACHE_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadHstuQK : public MmadBase<ArchTag_, false> {
    static constexpr bool ENABLE_PAGED_KV_CACHE = ENABLE_PAGED_KV_CACHE_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_PV_RESIDENT_L0C = true;

    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr uint32_t L1A_STAGES = 1;
    static constexpr uint32_t L1B_STAGES = 2;
    static constexpr uint32_t UBC_STAGES = 2;
};

template <class ArchTag_, bool ENABLE_PAGED_KV_CACHE_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadHstuPV : public MmadBase<ArchTag_, false> {
    static constexpr bool ENABLE_PAGED_KV_CACHE = ENABLE_PAGED_KV_CACHE_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_PV_RESIDENT_L0C = true;

    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr uint32_t L1A_STAGES = 2;
    static constexpr uint32_t L1B_STAGES = 2;
    static constexpr uint32_t UBC_STAGES = 2;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FagSdp : public MmadBase<Arch::AtlasA2, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FagdQKV : public MmadBase<Arch::AtlasA2, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

} // namespace Catlass::Gemm


// END vendored catlass/gemm/dispatch_policy.hpp


// BEGIN vendored catlass/gemm/tile/tile_copy.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <type_traits>
// already embedded: catlass/catlass.hpp

// BEGIN vendored catlass/detail/tag_to_layout.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// BEGIN vendored catlass/layout/layout.hpp
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

// BEGIN vendored catlass/layout/matrix.hpp
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

// BEGIN vendored catlass/coord.hpp
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

namespace Catlass {

/// Statically-sized array specifying Coords within a tensor
template <
    int RANK_,                 ///< Logical rank of coordinate
    class Index_ = uint32_t,   ///< Index type used for each dimension
    class LongIndex_ = int64_t ///< Long index type used for linear offsets
    >
struct Coord {
public:
    // Number of elements in Coord
    static const int RANK = RANK_;

    // Index typen used to store elements
    using Index = Index_;

    // Type used to represent linear offsets
    using LongIndex = LongIndex_;

    // Default ctor initializes uniformly
    CATLASS_HOST_DEVICE constexpr explicit Coord(Index value = Index(0)) : idx{}
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    // Constructs from an array of integers
    CATLASS_HOST_DEVICE constexpr Coord(Index const (&idx_)[RANK]) : idx{}
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    CATLASS_HOST_DEVICE
    constexpr int Argmin() const
    {
        return ArgminImpl<1>(0);
    }

    // Returns the index of the dimension with greatest value
    CATLASS_HOST_DEVICE
    constexpr int Argmax() const
    {
        return ArgmaxImpl<1>(0);
    }

    // Returns true if Coord is non-zero
    CATLASS_HOST_DEVICE
    constexpr explicit operator bool() const
    {
        return AnyImpl<0>();
    }

    // Return true if Coord is uniformly zero.
    CATLASS_HOST_DEVICE
    constexpr bool operator!() const
    {
        return !AnyImpl<0>();
    }

    // Element-wise addition
    CATLASS_HOST_DEVICE
    constexpr Coord operator+(Coord const& b) const
    {
        Coord c;
        AddCoordImpl<0>(c, b);
        return c;
    }

    // Add a scalar to each element
    CATLASS_HOST_DEVICE
    constexpr Coord operator+(const Index val) const
    {
        Coord c;
        AddScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise subtraction
    CATLASS_HOST_DEVICE
    constexpr Coord operator-(Coord const& b) const
    {
        Coord c;
        SubCoordImpl<0>(c, b);
        return c;
    }

    // Subtract a scalar from each element
    CATLASS_HOST_DEVICE
    constexpr Coord operator-(Index const val) const
    {
        Coord c;
        SubScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise multiply
    CATLASS_HOST_DEVICE
    constexpr Coord operator*(Coord const& b) const
    {
        Coord c;
        MulCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise division
    CATLASS_HOST_DEVICE
    constexpr Coord operator/(Coord const& b) const
    {
        Coord c;
        DivCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise mod
    CATLASS_HOST_DEVICE
    constexpr Coord operator%(Coord const& b) const
    {
        Coord c;
        ModCoordImpl<0>(c, b);
        return c;
    }

    // In-place addition
    CATLASS_HOST_DEVICE
    constexpr Coord& operator+=(Coord const& b)
    {
        PlusEqualImpl<0>(b);
        return *this;
    }

    // In-place equal
    CATLASS_HOST_DEVICE
    constexpr bool operator==(Coord const& b) const
    {
        return EqualCoordImpl<0>(b);
    }

    // In-place equal
    CATLASS_HOST_DEVICE
    constexpr bool operator==(Index const val) const
    {
        return EqualScalarImpl<0>(val);
    }

    // Member acces operator
    CATLASS_HOST_DEVICE constexpr Index& operator[](int dim)
    {
        return idx[dim];
    }

    // Member access operator
    CATLASS_HOST_DEVICE constexpr Index const& operator[](int dim) const
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    CATLASS_HOST_DEVICE Index& At()
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    CATLASS_HOST_DEVICE
    Index& At(int dim)
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    CATLASS_HOST_DEVICE Index const& At() const
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    CATLASS_HOST_DEVICE
    Index const& At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    CATLASS_HOST_DEVICE auto GetCoordByAxis() const
    {
        Index idx_[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{idx_};
    }

    CATLASS_HOST_DEVICE
    static constexpr Coord Min(Coord const& a, Coord const& b)
    {
        Coord res;
        for (int i = 0; i < RANK; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

private:
    template <int N>
    CATLASS_HOST_DEVICE constexpr int ArgminImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        } else {
            return ArgminImpl<N + 1>(idx[N] < idx[i] ? N : i);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr int ArgmaxImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        } else {
            return ArgmaxImpl<N + 1>(idx[N] > idx[i] ? N : i);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr bool AnyImpl() const
    {
        if constexpr (N == RANK) {
            return false;
        } else {
            return idx[N] || AnyImpl<N + 1>();
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void AddCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + b.idx[N];
            AddCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void AddScalarImpl(Coord& c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + val;
            AddScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void SubCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - b.idx[N];
            SubCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void SubScalarImpl(Coord& c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - val;
            SubScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void MulCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] * b.idx[N];
            MulCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void DivCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] / b.idx[N];
            DivCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void ModCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] % b.idx[N];
            ModCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr void PlusEqualImpl(Coord const& b)
    {
        if constexpr (N < RANK) {
            idx[N] += b.idx[N];
            PlusEqualImpl<N + 1>(b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr bool EqualCoordImpl(Coord const& b) const
    {
        if constexpr (N == RANK) {
            return true;
        } else {
            return idx[N] == b.idx[N] && EqualCoordImpl<N + 1>(b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE constexpr bool EqualScalarImpl(Index const val) const
    {
        if constexpr (N == RANK) {
            return true;
        } else {
            return idx[N] == val && EqualScalarImpl<N + 1>(val);
        }
    }

    // Indices
    Index idx[RANK];
};

// Helper to make a 1-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<1, T> MakeCoord(T dim0)
{
    T values[1] = {dim0};
    return Coord<1, T>(values);
}

/// Helper to make a 2-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<2, T> MakeCoord(T dim0, T dim1)
{
    T values[2] = {dim0, dim1};
    return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<3, T> MakeCoord(T dim0, T dim1, T dim2)
{
    T values[3] = {dim0, dim1, dim2};
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<4, T> MakeCoord(T dim0, T dim1, T dim2, T dim3)
{
    T values[4] = {dim0, dim1, dim2, dim3};
    return Coord<4, T>(values);
}

/// Helper to make a 5-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<5, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4)
{
    T values[5] = {dim0, dim1, dim2, dim3, dim4};
    return Coord<5, T>(values);
}

/// Helper to make a 6-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<6, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5)
{
    T values[6] = {dim0, dim1, dim2, dim3, dim4, dim5};
    return Coord<6, T>(values);
}

/// Helper to make a 7-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<7, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5, T dim6)
{
    T values[7] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6};
    return Coord<7, T>(values);
}

} // namespace Catlass


// END vendored catlass/coord.hpp


// BEGIN vendored catlass/numeric_size.hpp
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
// already embedded: tla/integral/integral_constant.hpp

#include <cstddef>
#include <type_traits>

namespace Catlass {

#if defined(__CCE__)
// Atlas A2 等：沿用 AscendC 对子字节类型（如 int4b_t）的逻辑位宽，避免 sizeof(T)*8 与设备布局不一致。
using AscendC::SizeOfBits;
#else
template <typename T>
struct SizeOfBits {
    static constexpr size_t value = sizeof(T) * 8;
};
#endif

// Host / non-kernel TU with CATLASS_ARCH 3510: Ascend FP4 typedefs may still use sizeof==1; keep logical 4-bit count.
#if !defined(__CCE__)
#endif

/// Returns the number of bytes required to hold a specified number of bits
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE constexpr ReturnType BitsToBytes(T bits)
{
    return (static_cast<ReturnType>(bits) + static_cast<ReturnType>(7)) / static_cast<ReturnType>(8);
}

template <auto t>
CATLASS_HOST_DEVICE constexpr auto BitsToBytes(tla::Int<t>)
{
    return tla::Int<BitsToBytes(t)>{};
}

/// Returns the number of bits required to hold a specified number of bytes
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE constexpr ReturnType BytesToBits(T bytes)
{
    return static_cast<ReturnType>(bytes) * static_cast<ReturnType>(8);
}

template <auto t>
CATLASS_HOST_DEVICE constexpr auto BytesToBits(tla::Int<t>)
{
    return tla::Int<BytesToBits(t)>{};
}

template <typename Element, typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE constexpr ReturnType BytesToElements(T bytes)
{
    return BytesToBits<ReturnType>(bytes) / ReturnType(SizeOfBits<Element>::value);
}

template <typename Element, auto t>
CATLASS_HOST_DEVICE constexpr auto BytesToElements(tla::Int<t>)
{
    return tla::Int<BytesToElements<Element>(t)>{};
}

template <typename Element, typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE constexpr ReturnType ElementsToBytes(T elements)
{
    return BitsToBytes<ReturnType>(elements * SizeOfBits<Element>::value);
}

template <typename Element, auto t>
CATLASS_HOST_DEVICE constexpr auto ElementsToBytes(tla::Int<t>)
{
    return tla::Int<ElementsToBytes<Element>(t)>{};
}

} // namespace Catlass


// END vendored catlass/numeric_size.hpp

// already embedded: catlass/detail/alignment.hpp

// BEGIN vendored catlass/matrix_coord.hpp
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

template <uint32_t ROW_ = 1, uint32_t COLUMN_ = 1>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;

    static constexpr int64_t COUNT = ROW * COLUMN;

    CATLASS_HOST_DEVICE
    static Coord<2> ToCoord()
    {
        return MakeCoord(ROW, COLUMN);
    }
};

/// MatrixCoord wraps Coord<2, uint32_t> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=2
    using Base = Coord<2, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// Rows dimension
    static constexpr uint32_t ROW_INDEX = 0;

    /// Columns dimension
    static constexpr uint32_t COLUMN_INDEX = 1;

    /// Default ctor
    CATLASS_HOST_DEVICE
    MatrixCoord()
    {}

    /// Constructs from Coord<2>
    CATLASS_HOST_DEVICE
    constexpr MatrixCoord(Coord<2, Index> const& coord) : Base(coord)
    {}

    /// Helper to construct from a row and column
    CATLASS_HOST_DEVICE
    constexpr MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column))
    {}

    /// Helper to construct from a row and column, which are LongIndex based
    CATLASS_HOST_DEVICE
    constexpr MatrixCoord(LongIndex row, LongIndex column) : Base(MakeCoord(Index(row), Index(column)))
    {}

    /// Returns the row of the coordinate
    CATLASS_HOST_DEVICE
    Index const& row() const
    {
        return this->At(ROW_INDEX);
    }

    /// Returns the row of the coordinate
    CATLASS_HOST_DEVICE
    Index& row()
    {
        return this->At(ROW_INDEX);
    }

    /// Returns the column of the coordinate
    CATLASS_HOST_DEVICE
    Index const& column() const
    {
        return this->At(COLUMN_INDEX);
    }

    /// Returns the column of the coordinate
    CATLASS_HOST_DEVICE
    Index& column()
    {
        return this->At(COLUMN_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    constexpr MatrixCoord operator+(Base const& b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    constexpr MatrixCoord& operator+=(Base const& b)
    {
        Base::operator+=(b);
        return *this;
    }
};

} // namespace Catlass


// END vendored catlass/matrix_coord.hpp


// BEGIN vendored catlass/conv_coord.hpp
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

/// Computes the conv output size along one axis:
/// (inSize + totalPad - dilation * (kernel - 1) - 1) / stride + 1.
/// The intermediate arithmetic is done in int64_t to avoid uint32_t underflow when the padded
/// input is smaller than the dilated kernel. Returns 0 (no valid output position) in that case
/// or when stride is 0.
CATLASS_HOST_DEVICE
constexpr int64_t ConvOutputSize(int64_t inSize, int64_t totalPad, int64_t kernel, int64_t stride, int64_t dilation)
{
    int64_t numerator = inSize + totalPad - dilation * (kernel - 1) - 1;
    if (numerator < 0 || stride <= 0) {
        return 0;
    }
    return numerator / stride + 1;
}

/// Shape of conv3d operation














/////////////////// Shapes and Coords for Conv2d ///////////////////



















} // namespace Catlass


// END vendored catlass/conv_coord.hpp


namespace Catlass::layout {

/// Mapping function for row-major matrices
struct RowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE constexpr RowMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1)))
    {}

    /// Constructor
    CATLASS_HOST_DEVICE constexpr RowMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1)))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride)
    {}

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayout(Index rows, Index cols)
    {
        return RowMajor(rows, cols);
    }

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayoutInUb(MatrixCoord const& shape)
    {
        constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
        return RowMajor(shape.row(), shape.column(), RoundUp<ELE_NUM_PER_BLK>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    RowMajor GetTileLayout(MatrixCoord const& tileShape) const
    {
        return RowMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(shape_[0]) * stride_[0];
    }

protected:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for col-major matrices
struct ColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows)))
    {}

    /// Constructor
    CATLASS_HOST_DEVICE constexpr ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride)
    {}

    template <class Element>
    CATLASS_HOST_DEVICE static ColumnMajor MakeLayout(Index rows, Index cols)
    {
        return ColumnMajor(rows, cols);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    ColumnMajor GetTileLayout(MatrixCoord const& tileShape) const
    {
        return ColumnMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(shape_[1]) * stride_[1];
    }

protected:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for nZ matrices which is col-major inside fractal and row-major between fractal
struct nZ {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr nZ(
        Index orgRows = 0,                 /// Number of rows of origin matrices
        Index orgCols = 0,                 /// Number of cols of origin matrices
        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal
        LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
        LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
        LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
        LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr nZ(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static nZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nZ(
            orgRows, orgCols, ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL,
            colsRound / C0_NUM_PER_FRACTAL, 1, colsRound * ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    nZ GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return nZ(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE constexpr typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(stride_[1]) * shape_[1];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for zN matrices which is row-major inside fractal and col-major between fractal
struct zN {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr zN(
        Index orgRows = 0,                 /// Number of rows of origin matrices
        Index orgCols = 0,                 /// Number of cols of origin matrices
        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal
        LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
        LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
        LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
        LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr zN(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(
            orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
            colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL, 1, rowsRound * ELE_NUM_PER_C0);
    }

    CATLASS_HOST_DEVICE
    static zN MakeLayoutInL0C(MatrixCoord const& shape)
    {
        return zN(
            shape.row(), shape.column(), C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
            C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()), C0_NUM_PER_FRACTAL,
            C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL, 1, RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    zN GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return zN(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE constexpr typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(stride_[3]) * shape_[3];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for zN matrices which is row-major inside fractal and row-major between fractal
struct zZ {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr zZ(
        Index orgRows = 0,                 /// Number of rows of origin matrices
        Index orgCols = 0,                 /// Number of cols of origin matrices
        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal
        LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
        LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
        LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
        LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr zZ(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static zZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zZ(
            orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
            colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, colsRound * C0_NUM_PER_FRACTAL, 1, ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for L0C matrices
/// A special data layout for L0C memory, used for accumulator storage.
/// This layout similar to zN layout, but uses C0_NUM_PER_FRACTAL for both row and column inner dimensions.
struct L0C {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr L0C(
        Index orgRows = 0,                 /// Number of rows of origin matrices
        Index orgCols = 0,                 /// Number of cols of origin matrices
        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal
        LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
        LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
        LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
        LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr L0C(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static L0C MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return L0C(
            orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, C0_NUM_PER_FRACTAL,
            colsRound / C0_NUM_PER_FRACTAL, C0_NUM_PER_FRACTAL, ELE_NUM_PER_FRACTAL, 1, rowsRound * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    L0C GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return L0C(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE constexpr typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(stride_[3]) * shape_[3];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for padding rowmajor matrices
/// A special data layout designed to improve the efficiency of matrix operations in non-512B aligned scenarios.
/// This layout is row-major within blocks and also row-major between blocks.
struct PaddingRowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE
    PaddingRowMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0)
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(blockRows, CeilDiv(orgRows, blockRows), blockCols, CeilDiv(orgCols, blockCols))),
          stride_(MakeCoord(
              (LongIndex)blockCols, (LongIndex)blockRows * (LongIndex)RoundUp(orgCols, blockCols), (LongIndex)1,
              (LongIndex)blockRows * (LongIndex)blockCols))
    {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1] + (LongIndex)coord.column() / blockCols * stride_[3] +
               (LongIndex)coord.row() % blockRows * stride_[0] + (LongIndex)coord.column() % blockCols;
    }

    CATLASS_HOST_DEVICE
    PaddingRowMajor GetTileLayout(MatrixCoord const& tileShape) const
    {
        return PaddingRowMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

private:
    //
    // Data members
    //

    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for padding columnmajor matrices
/// A special data layout designed to improve the efficiency of matrix operations in non-512B aligned scenarios.
/// This layout is column-major within blocks and also column-major between blocks.
struct PaddingColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE
    PaddingColumnMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0)
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(blockRows, CeilDiv(orgRows, blockRows), blockCols, CeilDiv(orgCols, blockCols))),
          stride_(MakeCoord(
              (LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols, (LongIndex)blockRows,
              (LongIndex)RoundUp(orgRows, blockRows) * (LongIndex)blockCols))
    {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1] + (LongIndex)coord.column() / blockCols * stride_[3] +
               (LongIndex)coord.row() % blockRows + (LongIndex)coord.column() % blockCols * stride_[2];
    }

    CATLASS_HOST_DEVICE
    PaddingColumnMajor GetTileLayout(MatrixCoord const& tileShape) const
    {
        return PaddingColumnMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

private:
    //
    // Data members
    //

    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

///////////////////////
// new add layout nN
// nN layout
struct nN {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE
    nN(Index orgRows = 0, /// Number of rows of origin matrices
       Index orgCols = 0, /// Number of cols of origin matrices

       Index rowsInFractal = 0, /// Number of rows inside the fractal
       Index rowsByFractal = 0, /// number of rows by the fractal
       Index colsInFractal = 0, /// number of cols inside the fractal
       Index colsByFractal = 0, /// number of cols by the fractal

       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE
    nN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE static nN MakeLayout(Index orgRows, Index orgCols)
    {
        static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nN(
            orgRows, orgCols,

            ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL, colsRound / C0_NUM_PER_FRACTAL,

            1, ELE_NUM_PER_FRACTAL, ELE_NUM_PER_C0, rowsRound * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};





/// Mapping function for nZ matrices which is col-major inside fractal and row-major between fractal
struct Weight4BitnZ {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr Weight4BitnZ(
        Index orgRows = 0,                 /// Number of rows of origin matrices
        Index orgCols = 0,                 /// Number of cols of origin matrices
        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal
        LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
        LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
        LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
        LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr Weight4BitnZ(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static Weight4BitnZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = 16;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 512;
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return Weight4BitnZ(
            orgRows, orgCols, ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL,
            colsRound / C0_NUM_PER_FRACTAL, 1, colsRound * ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    Weight4BitnZ GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return Weight4BitnZ(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(stride_[1]) * shape_[1];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};
} // namespace Catlass::layout


// END vendored catlass/layout/matrix.hpp


// BEGIN vendored catlass/layout/vector.hpp
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
// already embedded: catlass/coord.hpp
// already embedded: catlass/numeric_size.hpp

namespace Catlass::layout {

struct VectorLayout {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 1;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Shape vector
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

    /// Logical coordinate
    using TensorCoord = Coord<RANK, Index>;

public:
    // Methods

    CATLASS_HOST_DEVICE constexpr VectorLayout(Index size = 0)
        : shape_(MakeCoord(size)), stride_(MakeCoord(LongIndex(1)))
    {}

    CATLASS_HOST_DEVICE constexpr VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride)
    {}

    template <class Element>
    CATLASS_HOST_DEVICE static VectorLayout MakeLayoutInUb(TensorCoord const& tileShape)
    {
        constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
        return VectorLayout{ELE_NUM_PER_BLK > (tileShape[0])};
    }

    CATLASS_HOST_DEVICE
    LongIndex GetOffset(TensorCoord const& coord) const
    {
        return stride_[0] * coord[0];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    VectorLayout GetTileLayout(TensorCoord const& tileShape) const
    {
        return VectorLayout(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE constexpr typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE constexpr typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Stride data member
    Shape shape_;
    Stride stride_;
};

} // namespace Catlass::layout


// END vendored catlass/layout/vector.hpp


// BEGIN vendored catlass/layout/tensor.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: catlass/catlass.hpp
// already embedded: catlass/conv_coord.hpp
// already embedded: catlass/detail/alignment.hpp

namespace Catlass::layout {





} // namespace Catlass::layout


// END vendored catlass/layout/tensor.hpp



// END vendored catlass/layout/layout.hpp

// already embedded: catlass/numeric_size.hpp

// BEGIN vendored tla/layout.hpp
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
// already embedded: catlass/numeric_size.hpp
// already embedded: tla/integral/integral_constant.hpp
// already embedded: tla/integral/integral_math.hpp
// already embedded: tla/utils/math.hpp

// BEGIN vendored tla/tuple/tuple.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstddef>
#include <tuple>
// already embedded: catlass/detail/macros.hpp
// already embedded: tla/integral/integral_constant.hpp

// BEGIN vendored tla/integral/integral_sequence.hpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstddef>
// already embedded: tla/integral/integral_constant.hpp
// already embedded: tla/utils/type_traits.hpp

namespace tla {

template <typename T, T... Ns>
struct IntegerSequence {
    using value_type = T;
    static constexpr size_t size()
    {
        return sizeof...(Ns);
    }
};

template <typename Sequence, typename T, size_t N, typename = void>
struct MakeIntegerSequenceImpl;

template <typename T, typename NS>
struct MakeIntegerSequenceImpl<NS, T, 0> {
    typedef NS type;
};

template <typename T, T... Ns, size_t N>
struct MakeIntegerSequenceImpl<IntegerSequence<T, Ns...>, T, N, TLA_REQUIRES_T(N > 0)> {
    typedef typename MakeIntegerSequenceImpl<IntegerSequence<T, N - 1, Ns...>, T, N - 1>::type type;
};

template <typename T, T N>
using make_integer_sequence = typename MakeIntegerSequenceImpl<IntegerSequence<T>, T, N>::type;

// integer_sequence
template <class T, T... Ints>
using integer_sequence = IntegerSequence<T, Ints...>;

// index_sequence
template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

// int_sequence
template <int... Ints>
using int_sequence = integer_sequence<int, Ints...>;

template <int N>
using make_int_sequence = make_integer_sequence<int, N>;

// Shortcuts
template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

// make_range<Begin, End>: integer sequence Begin, Begin+1, ..., End-1
namespace detail {

template <int Begin, class S>
struct range_impl;

template <int Begin, int... Is>
struct range_impl<Begin, seq<Is...>> {
    using type = seq<(Is + Begin)...>;
};

} // end namespace detail

template <int Begin, int End>
using make_range = typename detail::range_impl<Begin, make_seq<(End > Begin) ? (End - Begin) : 0>>::type;

// make_rseq: reverse integer sequence N-1, N-2, ..., 1, 0
namespace detail {

template <int N, int... Is>
struct make_rseq_impl : make_rseq_impl<N - 1, Is..., N - 1> {};

template <int... Is>
struct make_rseq_impl<0, Is...> {
    using type = seq<Is...>;
};

} // end namespace detail

template <int N>
using make_rseq = typename detail::make_rseq_impl<N>::type;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

template <class Tuple>
using tuple_rseq = make_rseq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

// tuple traits for integer_sequence
template <class T, T... Ints>
struct tuple_size<integer_sequence<T, Ints...>> : std::integral_constant<size_t, sizeof...(Ints)> {};

template <size_t I, class T, T... Is>
struct tuple_element<I, integer_sequence<T, Is...>> {
    constexpr static T idx[sizeof...(Is)] = {Is...};
    using type = C<idx[I]>;
};

template <size_t I, class T, T... Ints>
CATLASS_HOST_DEVICE constexpr tuple_element_t<I, integer_sequence<T, Ints...>> get(integer_sequence<T, Ints...>)
{
    
    return {};
}

} // end namespace tla


// END vendored tla/integral/integral_sequence.hpp

// already embedded: tla/utils/type_traits.hpp

namespace tla {

namespace detail {

// EBO stands for "empty base optimization."
template <size_t N, class T, bool IsEmpty = std::is_empty<T>::value>
struct EBO;

// Specialization for types T that are empty;
template <size_t N, class T>
struct EBO<N, T, true> {
    CATLASS_HOST_DEVICE constexpr EBO()
    {}

    CATLASS_HOST_DEVICE constexpr EBO(T const&)
    {}
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T getv(EBO<N, T, true> const&)
{
    return {};
}

// Specialization for types T that are not empty;
template <size_t N, class T>
struct EBO<N, T, false> {
    CATLASS_HOST_DEVICE constexpr EBO() : t_{}
    {}

    CATLASS_HOST_DEVICE constexpr EBO(T const& t) : t_{t}
    {}

    T t_;
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T const& getv(EBO<N, T, false> const& x)
{
    return x.t_;
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T& getv(EBO<N, T, false>& x)
{
    return x.t_;
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T&& getv(EBO<N, T, false>&& x)
{
    return static_cast<T&&>(x.t_);
}

// TupleBase
template <class IdxSeq, class... T>
struct TupleBase;

template <size_t... I, class... T>
struct TupleBase<index_sequence<I...>, T...> : EBO<I, T>... {
    CATLASS_HOST_DEVICE constexpr TupleBase()
    {}

    CATLASS_HOST_DEVICE constexpr TupleBase(T const&... t) : EBO<I, T>(t)...
    {}
};

} // end namespace detail

// tla::tuple class.
template <class... T>
struct tuple : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...> {
    CATLASS_HOST_DEVICE constexpr tuple()
    {}

    CATLASS_HOST_DEVICE constexpr tuple(T const&... t)
        : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...>(t...)
    {}
};

template <>
struct tuple<> {};

// get for tla::tuple
template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...> const& t) noexcept
{
    
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>& t) noexcept
{
    
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>&& t) noexcept
{
    
    return detail::getv<I>(static_cast<tuple<T...>&&>(t));
}

// Integral SFINAE get: for scalar integral types, get<0> returns the value
template <size_t I, class T, TLA_REQUIRES(is_integral<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(T&& t) noexcept
{
    
    return static_cast<T&&>(t);
}

// Multi-index get: recursively index into nested tuples
template <size_t I0, size_t I1, size_t... Is, class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(T&& t) noexcept
{
    return get<I1, Is...>(get<I0>(static_cast<T&&>(t)));
}

namespace detail {

template <class T>
auto has_tuple_size(T*) -> Bool<(0 <= tuple_size<T>::value)>;
auto has_tuple_size(...) -> false_type;

} // end namespace detail

template <class T>
struct is_tuple : decltype(detail::has_tuple_size((T*)0)){};

template <class T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

template <class... Ts>
struct all_tuples : Bool<(is_tuple<Ts>::value && ...)> {};

template <class... Ts>
inline constexpr bool all_tuples_v = all_tuples<Ts...>::value;

template <class... T>
struct tuple_size<tla::tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

template <class... T>
struct tuple_size<const tla::tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

// make_tuple
template <class... T>
CATLASS_HOST_DEVICE constexpr tuple<T...> MakeTuple(T const&... t)
{
    return {t...};
}

// make_tuple (lowercase alias of MakeTuple)
template <class... T>
CATLASS_HOST_DEVICE constexpr tuple<T...> make_tuple(T const&... t)
{
    return {t...};
}

namespace detail {

template <class... T, class... U, size_t... I>
CATLASS_HOST_DEVICE constexpr bool tuple_equal_impl(tuple<T...> const& a, tuple<U...> const& b, index_sequence<I...>)
{
    return ((get<I>(a) == get<I>(b)) && ...);
}

} // namespace detail

template <class... T, class... U>
CATLASS_HOST_DEVICE constexpr bool operator==(tuple<T...> const& a, tuple<U...> const& b)
{
    
    return detail::tuple_equal_impl(a, b, make_index_sequence<sizeof...(T)>{});
}

template <class... T, class... U>
CATLASS_HOST_DEVICE constexpr bool operator!=(tuple<T...> const& a, tuple<U...> const& b)
{
    return !(a == b);
}

} // end namespace tla

#define TLA_ASSERT_ALL_TUPLES(...)

// Structured bindings support
namespace std {
template <class... T>
struct tuple_size<tla::tuple<T...>> : integral_constant<size_t, sizeof...(T)> {};

template <size_t I, class... T>
struct tuple_element<I, tla::tuple<T...>> {
    using type = tuple_element_t<I, tuple<T...>>;
};
} // namespace std


// END vendored tla/tuple/tuple.hpp


// BEGIN vendored tla/tuple/tuple_algorithms.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <utility>
// already embedded: catlass/detail/macros.hpp
// already embedded: tla/integral/integral_constant.hpp
// already embedded: tla/integral/integral_math.hpp
// already embedded: tla/integral/integral_sequence.hpp
// already embedded: tla/tuple/tuple.hpp
// already embedded: tla/utils/type_traits.hpp

namespace tla {

// ===========================================================================
// Apply: apply functions to tuple elements
// ===========================================================================

namespace detail {

template <class F, class T, int... Is>
CATLASS_HOST_DEVICE constexpr auto apply_unpack(F&& f, T&& t, seq<Is...>)
{
    return f(get<Is>(tla::forward<T>(t))...);
}

// apply: unpack tuple into function call — f(t_0, t_1, ..., t_n)
template <class F, class T>
CATLASS_HOST_DEVICE constexpr auto apply_impl(F&& f, T&& t)
{
    TLA_ASSERT_ALL_TUPLES(remove_cvref_t<T>);
    return apply_unpack(f, tla::forward<T>(t), tuple_seq<T>{});
}

// for_each: apply f to each element (no return value)
template <class F, class T>
CATLASS_HOST_DEVICE constexpr void for_each_impl(F&& f, T&& t)
{
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return apply_unpack([&](auto&&... a) { (f(tla::forward<decltype(a)>(a)), ...); }, t, tuple_seq<T>{});
    } else {
        return f(tla::forward<T>(t));
    }
}

} // end namespace detail

// Public algorithms keep the tuple-first argument order.
template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto apply(T&& t, F&& f)
{
    return detail::apply_impl(tla::forward<F>(f), tla::forward<T>(t));
}

template <class T, class F>
CATLASS_HOST_DEVICE constexpr void for_each(T&& t, F&& f)
{
    return detail::for_each_impl(tla::forward<F>(f), tla::forward<T>(t));
}

// ===========================================================================
// Construction: wrap, select, take, repeat, tuple_cat, append, prepend
// ===========================================================================

// wrap: wrap non-tuples into a rank-1 tuple, leave tuples unchanged
template <class T>
CATLASS_HOST_DEVICE constexpr auto wrap(T const& t)
{
    if constexpr (is_tuple<T>::value) {
        return t;
    } else {
        return make_tuple(t);
    }
}

// select<I...>(t): select elements at the given indices into a new tuple
template <size_t... I, class T>
CATLASS_HOST_DEVICE constexpr auto select(T const& t)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return make_tuple(get<I>(t)...);
}

// take<B, E>(t): take elements in range [B, E)
template <int B, int E, class T>
CATLASS_HOST_DEVICE constexpr auto take(T const& t)
{
    TLA_ASSERT_ALL_TUPLES(T);
    
    return detail::apply_unpack([](auto const&... a) { return make_tuple(a...); }, t, make_range<B, E>{});
}

// repeat<N>(t): N=1 returns t as-is; N!=1 wraps N copies in a tuple
// tuple_repeat<N>(t): always wraps N copies in a tuple
namespace detail {

template <size_t N, class T, int... Is>
CATLASS_HOST_DEVICE constexpr auto tuple_repeat_impl(T const& t, seq<Is...>)
{
    return make_tuple((void(Is), t)...);
}

} // end namespace detail

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr auto tuple_repeat(T const& t)
{
    return detail::tuple_repeat_impl<N>(t, make_seq<N>{});
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr auto repeat(T const& t)
{
    if constexpr (N == 1) {
        return t;
    } else {
        return tuple_repeat<N>(t);
    }
}

// tuple_cat: concatenate tuples
namespace detail {

template <class T0, class T1, int... Is0, int... Is1>
CATLASS_HOST_DEVICE constexpr auto tuple_cat_impl(T0 const& t0, T1 const& t1, seq<Is0...>, seq<Is1...>)
{
    return make_tuple(get<Is0>(t0)..., get<Is1>(t1)...);
}

} // end namespace detail

CATLASS_HOST_DEVICE constexpr tuple<> tuple_cat()
{
    return {};
}

template <class T>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T const& t)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return t;
}

template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1)
{
    TLA_ASSERT_ALL_TUPLES(T0, T1);
    return detail::tuple_cat_impl(t0, t1, tuple_seq<T0>{}, tuple_seq<T1>{});
}

template <class T0, class T1, class... Ts>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    TLA_ASSERT_ALL_TUPLES(T0, T1, Ts...);
    return tuple_cat(tuple_cat(t0, t1), ts...);
}

// append: append elements to a tuple
namespace detail {

template <class T, class X, int... Is>
CATLASS_HOST_DEVICE constexpr auto append_impl(T const& t, X const& x, seq<Is...>)
{
    return make_tuple(get<Is>(t)..., x);
}

} // end namespace detail

// append(a, x): append single element to a tuple
template <class T, class X>
CATLASS_HOST_DEVICE constexpr auto append(T const& a, X const& x)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return detail::append_impl(a, x, tuple_seq<T>{});
}

// append<N>(a, x): pad a tuple to rank N by appending copies of x
template <int N, class T, class X>
CATLASS_HOST_DEVICE constexpr auto append(T const& a, X const& x)
{
    TLA_ASSERT_ALL_TUPLES(T);
    constexpr int S = tuple_size<T>::value;
    
    return tuple_cat(a, tuple_repeat<N - S>(x));
}

// prepend: prepend a single element to the beginning of a tuple
namespace detail {

template <class T, class X, int... Is>
CATLASS_HOST_DEVICE constexpr auto prepend_impl(T const& t, X const& x, seq<Is...>)
{
    return make_tuple(x, get<Is>(t)...);
}

} // end namespace detail

template <class T, class X>
CATLASS_HOST_DEVICE constexpr auto prepend(T const& t, X const& x)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return detail::prepend_impl(t, x, tuple_seq<T>{});
}

// ===========================================================================
// Transform: map elements to produce new values/tuples
// ===========================================================================

namespace detail {

template <int I, class F, class... Ts>
CATLASS_HOST_DEVICE constexpr decltype(auto) get_apply_impl(F&& f, Ts&&... ts)
{
    // Preserve intermediate references from f when passing its result to g.
    return f(get<I>(tla::forward<Ts>(ts))...);
}

template <class F, class G, int... Is, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_apply_unpack(F&& f, G&& g, seq<Is...>, Ts&&... ts)
{
    return g(get_apply_impl<Is>(f, tla::forward<Ts>(ts)...)...);
}

// transform_apply: transform each element with f, then pack with g
template <class F, class G, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_apply_impl(F&& f, G&& g, T0&& t0, Ts&&... ts)
{
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(remove_cvref_t<T0>, remove_cvref_t<Ts>...);
        return transform_apply_unpack(f, g, tuple_seq<T0>{}, tla::forward<T0>(t0), tla::forward<Ts>(ts)...);
    } else {
        return g(f(tla::forward<T0>(t0), tla::forward<Ts>(ts)...));
    }
}

// transform: apply f to each element, return new tuple
template <class F, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_impl(F&& f, T0 const& t0, Ts const&... ts)
{
    if constexpr (is_tuple<T0>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, Ts...);
        return transform_apply_unpack(
            f, [](auto const&... a) { return tla::make_tuple(a...); }, tuple_seq<T0>{}, t0, ts...);
    } else {
        return f(t0, ts...);
    }
}

} // end namespace detail

template <class T, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T&& t, F&& f, G&& g)
{
    return detail::transform_apply_impl(tla::forward<F>(f), tla::forward<G>(g), tla::forward<T>(t));
}

template <class T0, class T1, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T0&& t0, T1&& t1, F&& f, G&& g)
{
    return detail::transform_apply_impl(
        tla::forward<F>(f), tla::forward<G>(g), tla::forward<T0>(t0), tla::forward<T1>(t1));
}

template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto transform(T const& t, F&& f)
{
    return detail::transform_impl(tla::forward<F>(f), t);
}

template <class T0, class T1, class F>
CATLASS_HOST_DEVICE constexpr auto transform(T0 const& t0, T1 const& t1, F&& f)
{
    return detail::transform_impl(tla::forward<F>(f), t0, t1);
}

// transform_leaf: recursively apply f to leaf (non-tuple) elements
template <class F, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_leaf(F&& f, T0 const& t0, Ts const&... ts)
{
    if constexpr (is_tuple<T0>::value) {
        return detail::transform_impl([&](auto const&... a) { return transform_leaf(f, a...); }, t0, ts...);
    } else {
        return f(t0, ts...);
    }
}

// filter_tuple: transform each element with f, then concatenate results
template <class F, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto filter_tuple(F&& f, T0 const& t0, Ts const&... ts)
{
    return detail::transform_apply_impl(f, [](auto const&... a) { return tuple_cat(a...); }, t0, ts...);
}

// ===========================================================================
// Fold: left fold and reverse fold over tuple elements
// ===========================================================================

namespace detail {

template <class F, class V, class T, int I0, int... Is>
CATLASS_HOST_DEVICE constexpr auto fold_impl(F&& f, V const& v, T const& t, seq<I0, Is...>)
{
    return fold_impl(f, f(v, get<I0>(t)), t, seq<Is...>{});
}

template <class F, class V, class T>
CATLASS_HOST_DEVICE constexpr auto fold_impl(F&& f, V const& v, T const& t, seq<>)
{
    return v;
}

} // end namespace detail

// fold: f(...f(f(v, t_0), t_1),...,t_n)
template <class F, class V, class T>
CATLASS_HOST_DEVICE constexpr auto fold(F&& f, V const& v, T const& t)
{
    if constexpr (is_tuple<T>::value) {
        return detail::fold_impl(f, v, t, tuple_seq<T>{});
    } else {
        return f(v, t);
    }
}

// fold_reverse: f(...f(f(v, t_n), t_{n-1}),...,t_0)
template <class F, class V, class T>
CATLASS_HOST_DEVICE constexpr auto fold_reverse(F&& f, V const& v, T const& t)
{
    if constexpr (is_tuple<T>::value) {
        return detail::fold_impl(f, v, t, tuple_rseq<T>{});
    } else {
        return f(v, t);
    }
}

// ===========================================================================
// Properties: rank, depth, is_flat
// ===========================================================================

// rank: number of elements at the top level
template <int... Is, class T>
CATLASS_HOST_DEVICE constexpr auto rank(T const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<T>::value) {
            return Int<tuple_size<T>::value>{};
        } else {
            return Int<1>{};
        }
    } else {
        return rank(get<Is...>(t));
    }
}

template <class T>
using rank_t = decltype(rank(std::declval<T>()));

template <class T>
constexpr auto rank_v = rank_t<T>::value;

// depth: maximum nesting level (scalar=0, flat tuple=1, ...)
template <int... Is, class T>
CATLASS_HOST_DEVICE constexpr auto depth(T const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<T>::value) {
            return Int<1>{} + detail::apply_impl(
                                  [](auto const&... v) {
                                      if constexpr (sizeof...(v) == 0) {
                                          return Int<0>{};
                                      } else if constexpr (sizeof...(v) == 1) {
                                          return depth(v...);
                                      } else {
                                          return max(depth(v)...);
                                      }
                                  },
                                  t);
        } else {
            return Int<0>{};
        }
    } else {
        return depth(get<Is...>(t));
    }
}

template <class T>
using depth_t = decltype(depth(std::declval<T>()));

template <class T>
constexpr auto depth_v = depth_t<T>::value;

// is_flat: true if tuple has no nested tuples
template <class T>
struct is_flat : true_type {};

template <class... Ts>
struct is_flat<tuple<Ts...>> : Bool<(true && ... && (!is_tuple<Ts>::value))> {};

// ===========================================================================
// Flatten: flatten and unflatten hierarchical tuples
// ===========================================================================

// flatten_to_tuple: flatten a hierarchical tuple to depth 1, wrap non-tuples
template <class T>
CATLASS_HOST_DEVICE constexpr auto flatten_to_tuple(T const& t)
{
    if constexpr (is_tuple<T>::value && !is_flat<T>::value) {
        return filter_tuple([](auto const& a) { return flatten_to_tuple(a); }, t);
    } else {
        return wrap(t);
    }
}

// unflatten: reconstruct a hierarchical tuple from a flat tuple, guided by target_profile
namespace detail {

template <class FlatTuple, class TargetProfile>
CATLASS_HOST_DEVICE constexpr auto unflatten_impl(FlatTuple const& flat_tuple, TargetProfile const& target_profile)
{
    if constexpr (is_tuple<TargetProfile>::value) {
        return fold(
            [](auto const& v, auto const& t) {
                auto sub = unflatten_impl(get<1>(v), t);
                return make_tuple(append(get<0>(v), get<0>(sub)), get<1>(sub));
            },
            make_tuple(make_tuple(), flat_tuple), target_profile);
    } else {
        return make_tuple(get<0>(flat_tuple), take<1, rank_v<FlatTuple>>(flat_tuple));
    }
}

} // end namespace detail

template <class FlatTuple, class TargetProfile>
CATLASS_HOST_DEVICE constexpr auto unflatten(FlatTuple const& flat_tuple, TargetProfile const& target_profile)
{
    TLA_ASSERT_ALL_TUPLES(FlatTuple);
    auto result = detail::unflatten_impl(flat_tuple, target_profile);
    
    return get<0>(result);
}

} // end namespace tla


// END vendored tla/tuple/tuple_algorithms.hpp


// BEGIN vendored tla/tuple/tuple_math.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: tla/tuple/tuple.hpp
// already embedded: tla/tuple/tuple_algorithms.hpp
// already embedded: tla/utils/math.hpp

// BEGIN vendored tla/utils/functional.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: tla/utils/type_traits.hpp
// already embedded: tla/utils/math.hpp
// already embedded: catlass/detail/macros.hpp

namespace tla {

// ---------------------------------------------------------------------------
// identity — returns input unchanged
// ---------------------------------------------------------------------------

struct identity {
    template <class T>
    CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const
    {
        return tla::forward<T>(arg);
    }
};

// ---------------------------------------------------------------------------
// Macro-generated unary functors
// ---------------------------------------------------------------------------

#define TLA_LEFT_UNARY_OP(NAME, OP)                                            \
    struct NAME {                                                              \
        template <class T>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const \
        {                                                                      \
            return (OP tla::forward<T>(arg));                                  \
        }                                                                      \
    }

#define TLA_NAMED_UNARY_OP(NAME, OP)                                           \
    struct NAME {                                                              \
        template <class T>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const \
        {                                                                      \
            return OP(tla::forward<T>(arg));                                   \
        }                                                                      \
    }

TLA_LEFT_UNARY_OP(negate, -);
TLA_NAMED_UNARY_OP(abs_fn, abs);

#undef TLA_LEFT_UNARY_OP
#undef TLA_NAMED_UNARY_OP

// ---------------------------------------------------------------------------
// Macro-generated binary functors
// ---------------------------------------------------------------------------

#define TLA_BINARY_OP(NAME, OP)                                                         \
    struct NAME {                                                                       \
        template <class T, class U>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& lhs, U&& rhs) const \
        {                                                                               \
            return (tla::forward<T>(lhs) OP tla::forward<U>(rhs));                      \
        }                                                                               \
    }

#define TLA_NAMED_BINARY_OP(NAME, OP)                                                   \
    struct NAME {                                                                       \
        template <class T, class U>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& lhs, U&& rhs) const \
        {                                                                               \
            return OP(tla::forward<T>(lhs), tla::forward<U>(rhs));                      \
        }                                                                               \
    }

TLA_BINARY_OP(plus, +);
TLA_BINARY_OP(minus, -);
TLA_BINARY_OP(greater, >);
TLA_BINARY_OP(greater_equal, >=);
TLA_BINARY_OP(less, <);
TLA_BINARY_OP(less_equal, <=);
TLA_NAMED_BINARY_OP(min_fn, min);
TLA_NAMED_BINARY_OP(max_fn, max);

#undef TLA_BINARY_OP
#undef TLA_NAMED_BINARY_OP

// ---------------------------------------------------------------------------
// Fold functors (left-fold only)
// ---------------------------------------------------------------------------

#define TLA_FOLD_OP(NAME, OP)                                         \
    struct NAME##_unary_lfold {                                       \
        template <class... T>                                         \
        CATLASS_HOST_DEVICE constexpr auto operator()(T&&... t) const \
        {                                                             \
            return (... OP t);                                        \
        }                                                             \
    }

TLA_FOLD_OP(multiplies, *);

#undef TLA_FOLD_OP

} // namespace tla


// END vendored tla/utils/functional.hpp

// already embedded: tla/utils/type_traits.hpp
// already embedded: tla/integral/integral_constant.hpp
// already embedded: tla/integral/integral_sequence.hpp
// already embedded: catlass/detail/macros.hpp

namespace tla {

// is_int_tuple: true if T is an integral, or a tuple whose elements are all is_int_tuple (recursive)
template <class T>
struct is_int_tuple : is_integral<T> {};

template <class... Ts>
struct is_int_tuple<tla::tuple<Ts...>> : Bool<(is_int_tuple<Ts>::value && ...)> {};

template <class T>
inline constexpr bool is_int_tuple_v = is_int_tuple<T>::value;

// ---------------------------------------------------------------------------
// Product / product / size / product_like: aggregation over tuple elements
// ---------------------------------------------------------------------------

struct Product {
    template <class T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const& a) const
    {
        if constexpr (is_tuple_v<T>) {
            if constexpr (tuple_size_v<T> == 0) {
                return Int<1>{};
            } else {
                return detail::transform_apply_impl(Product{}, multiplies_unary_lfold{}, a);
            }
        } else if constexpr (is_integral_v<T>) {
            return a;
        } else {
            
        }
    }
};

// product: multiply all elements (recursive, empty tuple = 1)
template <class T>
CATLASS_HOST_DEVICE constexpr auto product(T const& t)
{
    return Product{}(t);
}

// size: total element count (alias for product)
template <class T>
CATLASS_HOST_DEVICE constexpr auto size(T const& t)
{
    return product(t);
}

// product_like: product of leaves in t, guided by structure of g
template <class T, class TG>
CATLASS_HOST_DEVICE constexpr auto product_like(T const& t, TG const& g)
{
    return transform_leaf([](auto const&, auto const& x) { return product(x); }, g, t);
}

// product_each: product of each element (one-level flatten)
template <class T>
CATLASS_HOST_DEVICE constexpr auto product_each(T const& t)
{
    return detail::transform_impl(Product{}, wrap(t));
}

// ---------------------------------------------------------------------------
// Arithmetic operators: element-wise +, -, *, / with scalar broadcasting
// ---------------------------------------------------------------------------

#define TLA_TUPLE_BINARY_OP(OP)                                                                       \
    template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>                 \
    CATLASS_HOST_DEVICE constexpr auto operator OP(T0 const& a, T1 const& b)                          \
    {                                                                                                 \
        if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {                                             \
            TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);                                                       \
            return detail::transform_impl([](auto const& x, auto const& y) { return x OP y; }, a, b); \
        } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {                                   \
            return detail::transform_impl([&](auto const& x) { return x OP b; }, a);                  \
        } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {                                   \
            return detail::transform_impl([&](auto const& x) { return a OP x; }, b);                  \
        } else {                                                                                      \
                     \
        }                                                                                             \
    }

TLA_TUPLE_BINARY_OP(+);
TLA_TUPLE_BINARY_OP(-);
TLA_TUPLE_BINARY_OP(*);
TLA_TUPLE_BINARY_OP(/);

#undef TLA_TUPLE_BINARY_OP

// minimum: element-wise min (recursive, supports broadcasting)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto minimum(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_impl([](auto const& x, auto const& y) { return minimum(x, y); }, t0, t1);
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return minimum(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return minimum(t0, x); }, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return min(t0, t1);
    } else {
        
    }
}

// maximum: element-wise max (recursive, supports broadcasting)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto maximum(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_impl([](auto const& x, auto const& y) { return maximum(x, y); }, t0, t1);
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return maximum(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return maximum(t0, x); }, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return max(t0, t1);
    } else {
        
    }
}

// clip_sub: element-wise clamping subtraction (recursive, supports broadcasting)
template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>
CATLASS_HOST_DEVICE constexpr auto clip_sub(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_impl([](auto const& x, auto const& y) { return clip_sub(x, y); }, t0, t1);
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return clip_sub(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return clip_sub(t0, x); }, t1);
    } else {
        
    }
}

// ceil_div: tiling-aware ceiling division (recursive)
// - both tuples: pad shorter with 1, element-wise
// - tuple / scalar: cascading (divisor consumed across elements)
// - scalar / tuple: product then divide
template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>
CATLASS_HOST_DEVICE constexpr auto ceil_div(T0 const& a, T1 const& b)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        
        constexpr int R = tuple_size<T0>::value;
        return detail::transform_impl(
            [](auto const& x, auto const& y) { return ceil_div(x, y); }, a, append<R>(b, Int<1>{}));
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        auto result = fold(
            [](auto const& init, auto const& ai) {
                return make_tuple(append(get<0>(init), ceil_div(ai, get<1>(init))), ceil_div(get<1>(init), ai));
            },
            make_tuple(make_tuple(), b), a);
        return get<0>(result);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return ceil_div(a, product(b));
    } else {
        
    }
}

// round_up: element-wise round up to multiple (recursive, supports broadcasting)
template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>
CATLASS_HOST_DEVICE constexpr auto round_up(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        
        constexpr int R = tuple_size<T0>::value;
        return detail::transform_impl(
            [](auto const& x, auto const& y) { return round_up(x, y); }, t0, append<R>(t1, Int<1>{}));
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return round_up(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return round_up(t0, x); }, t1);
    } else {
        
    }
}

// inner_product: dot product (recursive). For tuples, element-wise multiply then sum;
// for scalars, returns the product.
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto inner_product(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_apply_impl(
            [](auto const& x, auto const& y) { return inner_product(x, y); },
            [](auto const&... v) { return (Int<0>{} + ... + v); }, t0, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return t0 * t1;
    } else {
        
    }
}

// evenly_divides: check if t0 is evenly divisible by t1 (element-wise)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto evenly_divides(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_apply_impl(
            [](auto const& x, auto const& y) { return (x % y) == 0; }, [](auto... bs) { return (bs && ...); }, t0, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return (t0 % t1) == 0;
    } else {
        
    }
}

} // end namespace tla


// END vendored tla/tuple/tuple_math.hpp

// already embedded: tla/utils/functional.hpp

// BEGIN vendored tla/stride.hpp
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// already embedded: tla/tuple/tuple.hpp
// already embedded: tla/tuple/tuple_algorithms.hpp
// already embedded: tla/tuple/tuple_math.hpp
// already embedded: tla/utils/type_traits.hpp
// already embedded: tla/integral/integral_constant.hpp
// already embedded: tla/integral/integral_sequence.hpp
// already embedded: catlass/detail/macros.hpp

namespace tla {

template <class Shape, class Current = _1LL>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current = {});

template <class Shape, class Current = _1LL>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current = {});

namespace detail {

template <class Shape, class Current, int I, int... Is>
CATLASS_HOST_DEVICE constexpr auto compact_col_major_impl(Shape const& shape, Current const& current, seq<I, Is...>)
{
    auto stride = compact_col_major(get<I>(shape), current);
    if constexpr (sizeof...(Is) == 0) {
        return make_tuple(stride);
    } else {
        auto next = current * product(get<I>(shape));
        return prepend(compact_col_major_impl(shape, next, seq<Is...>{}), stride);
    }
}

template <class Shape, class Current, int I, int... Is>
CATLASS_HOST_DEVICE constexpr auto compact_row_major_impl(Shape const& shape, Current const& current, seq<I, Is...>)
{
    auto stride = compact_row_major(get<I>(shape), current);
    if constexpr (sizeof...(Is) == 0) {
        return make_tuple(stride);
    } else {
        auto next = current * product(get<I>(shape));
        return append(compact_row_major_impl(shape, next, seq<Is...>{}), stride);
    }
}

} // end namespace detail

// compact_col_major: column-major compact strides.
// stride[0] = current, stride[i] = stride[i-1] * shape[i-1].
// Supports nested shapes (recurses into tuple elements) and custom starting stride.
template <class Shape, class Current>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current)
{
    if constexpr (is_tuple_v<Shape> && is_tuple_v<Current>) {
        return detail::transform_impl(
            [](auto const& s, auto const& c) { return compact_col_major(s, c); }, shape, current);
    } else if constexpr (is_tuple_v<Shape> && is_integral_v<Current>) {
        return detail::compact_col_major_impl(shape, current, tuple_seq<Shape>{});
    } else if constexpr (is_integral_v<Shape> && is_integral_v<Current>) {
        return current;
    } else {
        
    }
}

// compact_row_major: row-major compact strides.
// stride[N-1] = current, stride[i] = stride[i+1] * shape[i+1].
// Supports nested shapes and custom starting stride.
template <class Shape, class Current>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current)
{
    if constexpr (is_tuple_v<Shape> && is_tuple_v<Current>) {
        return detail::transform_impl(
            [](auto const& s, auto const& c) { return compact_row_major(s, c); }, shape, current);
    } else if constexpr (is_tuple_v<Shape> && is_integral_v<Current>) {
        return detail::compact_row_major_impl(shape, current, tuple_rseq<Shape>{});
    } else if constexpr (is_integral_v<Shape> && is_integral_v<Current>) {
        return current;
    } else {
        
    }
}

namespace detail {

template <class FlatShape, class FlatOrder, class Order, int... Is>
CATLASS_HOST_DEVICE constexpr auto order_stride_start(
    FlatShape const& flat_shape, FlatOrder const& flat_order, Order const& order, seq<Is...>)
{
    return (_1LL{} * ... * conditional_return(get<Is>(flat_order) < order, get<Is>(flat_shape), _1LL{}));
}

template <class Shape, class Order, class FlatShape, class FlatOrder>
CATLASS_HOST_DEVICE constexpr auto compact_order_impl(
    Shape const& shape, Order const& order, FlatShape const& flat_shape, FlatOrder const& flat_order)
{
    if constexpr (is_tuple<Order>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Shape, Order);
        return detail::transform_impl(
            [&](auto const& s, auto const& o) { return compact_order_impl(s, o, flat_shape, flat_order); }, shape,
            order);
    } else {
        auto stride_start = order_stride_start(flat_shape, flat_order, order, tuple_seq<FlatShape>{});
        return compact_col_major(shape, stride_start);
    }
}

} // end namespace detail

template <class Shape, class Order>
CATLASS_HOST_DEVICE constexpr auto compact_order(Shape const& shape, Order const& order)
{
    
    auto flat_shape = flatten_to_tuple(product_like(shape, order));
    auto flat_order = flatten_to_tuple(order);
    constexpr int n = tuple_size<remove_cvref_t<decltype(flat_order)>>::value;
    auto unique_order =
        detail::transform_impl([](auto I, auto v) { return v * Int<n>{} + I; }, make_range<0, n>{}, flat_order);
    auto new_order = unflatten(unique_order, order);
    return detail::compact_order_impl(shape, new_order, flat_shape, unique_order);
}

template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple_v<Coord> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Coord, Shape, Stride);
        return detail::transform_apply_impl(
            [](auto const& c, auto const& s, auto const& d) { return crd2idx(c, s, d); },
            [](auto const&... xs) { return (_0LL{} + ... + xs); }, coord, shape, stride);
    } else if constexpr (is_integral_v<Coord> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Shape, Stride);
        if constexpr (is_constant<0, Coord>::value) {
            return _0LL{};
        } else {
            auto zipped =
                detail::transform_impl([](auto const& s, auto const& d) { return make_tuple(s, d); }, shape, stride);
            constexpr int N = tuple_size<Shape>::value;
            const auto [remaining, result] = fold(
                [](auto const& acc, auto const& sd) {
                    auto prod = product(get<0>(sd));
                    return make_tuple(
                        get<0>(acc) / prod, get<1>(acc) + crd2idx(get<0>(acc) % prod, get<0>(sd), get<1>(sd)));
                },
                make_tuple(coord, _0LL{}), take<0, N - 1>(zipped));
            const auto [s, d] = get<N - 1>(zipped);
            return result + crd2idx(remaining, s, d);
        }
    } else if constexpr (is_integral_v<Coord> && is_integral_v<Shape> && is_integral_v<Stride>) {
        return coord * stride;
    } else {
        
    }
}

// crd2offset: alias for crd2idx (backward compatibility)
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2offset(Coord const& coord, Shape const& shape, Stride const& stride)
{
    return crd2idx(coord, shape, stride);
}

template <class Coord, class Shape>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape)
{
    if constexpr (is_tuple_v<Coord> && is_tuple_v<Shape>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Coord, Shape);
        auto flat_coord = flatten_to_tuple(coord);
        auto flat_shape = flatten_to_tuple(product_like(shape, coord));
        auto zipped = detail::transform_impl(
            [](auto const& c, auto const& s) { return make_tuple(c, s); }, flat_coord, flat_shape);
        return fold_reverse(
            [](auto const& acc, auto const& cs) { return get<0>(cs) + get<1>(cs) * acc; }, _0LL{}, zipped);
    } else if constexpr (is_integral_v<Coord> && (is_tuple_v<Shape> || is_integral_v<Shape>)) {
        return coord;
    } else {
        
    }
}

// idx2crd: inverse of crd2idx. Decomposes a flat index into a coordinate within <Shape, Stride>.
// Shape and Stride are assumed congruent (same tuple structure).
// - (tuple, tuple, tuple): each mode decomposed independently via element-wise recursion
// - (int, tuple, tuple): the same index is applied to every mode
// - (int, int, int): 1D base case: (idx / stride) % shape; shape == 1 short-circuits to 0
template <class Index, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto idx2crd(Index const& idx, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple_v<Index> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Index, Shape, Stride);
        return detail::transform_impl(
            [](auto const& i, auto const& s, auto const& d) { return idx2crd(i, s, d); }, idx, shape, stride);
    } else if constexpr (is_integral_v<Index> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Shape, Stride);
        return detail::transform_impl([&](auto const& s, auto const& d) { return idx2crd(idx, s, d); }, shape, stride);
    } else if constexpr (is_integral_v<Index> && is_integral_v<Shape> && is_integral_v<Stride>) {
        if constexpr (is_constant<1, Shape>::value) {
            return _0LL{};
        } else {
            return (idx / stride) % shape;
        }
    } else {
        
    }
}

// idx2crd: inverse of crd2idx 2-arg. Decomposes a flat index into a coordinate within Shape
// assuming compact column-major layout.
template <class Index, class Shape>
CATLASS_HOST_DEVICE constexpr auto idx2crd(Index const& idx, Shape const& shape)
{
    if constexpr (is_tuple_v<Index> && is_tuple_v<Shape>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Index, Shape);
        return detail::transform_impl([](auto const& i, auto const& s) { return idx2crd(i, s); }, idx, shape);
    } else if constexpr (is_integral_v<Index> && is_tuple_v<Shape>) {
        return idx2crd(idx, shape, compact_col_major(shape));
    } else if constexpr (is_integral_v<Index> && is_integral_v<Shape>) {
        return idx;
    } else {
        
    }
}

} // namespace tla


// END vendored tla/stride.hpp

// already embedded: catlass/layout/layout.hpp

namespace tla {

// Type aliases and factory functions
template <class... Shapes>
using Shape = tla::tuple<Shapes...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Shape<Ts...> make_shape(Ts const&... t)
{
    return {t...};
}
// PascalCase alias
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Shape<Ts...> MakeShape(Ts const&... t)
{
    return make_shape(t...);
}

template <class... Strides>
using Stride = tla::tuple<Strides...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Stride<Ts...> make_stride(Ts const&... t)
{
    return {t...};
}
// PascalCase alias
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Stride<Ts...> MakeStride(Ts const&... t)
{
    return make_stride(t...);
}

template <class... Coords>
using Coord = tla::tuple<Coords...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Coord<Ts...> make_coord(Ts const&... t)
{
    return {t...};
}
// PascalCase alias
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Coord<Ts...> MakeCoord(Ts const&... t)
{
    return make_coord(t...);
}

template <class... Strides>
using Step = tla::tuple<Strides...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Step<Ts...> make_step(Ts const&... t)
{
    return {t...};
}

template <class... Layouts>
using Tile = tla::tuple<Layouts...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Tile<Ts...> make_tile(Ts const&... t)
{
    return {t...};
}

// Layout core type
namespace detail {

template <class Shape>
CATLASS_HOST_DEVICE constexpr auto make_origin_shape(Shape const& shape)
{
    return conditional_return(is_tuple<Shape>{}, product_each(shape), shape);
}

} // namespace detail

template <class Shape, class Stride, class OriginShape = decltype(detail::make_origin_shape(tla::declval<Shape>()))>
struct Layout : private tla::tuple<Shape, Stride, OriginShape> {
    CATLASS_HOST_DEVICE constexpr Layout(
        Shape const& shape = {}, Stride const& stride = {}, OriginShape const& originShape = {})
        : tla::tuple<Shape, Stride, OriginShape>(shape, stride, originShape)
    {}

    static constexpr int rank = rank_v<Stride>;
    static constexpr int depth = depth_v<Stride>;

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) shape()
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) shape() const
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) stride()
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) stride() const
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape()
    {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape() const
    {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const&>(*this));
    }

    template <class Coord>
    CATLASS_HOST_DEVICE constexpr auto operator()(Coord const& coord) const
    {
        return crd2offset(coord, shape(), stride());
    }
};

template <class Layout>
struct is_layout : false_type {};
template <class Shape, class Stride, class OriginShape>
struct is_layout<Layout<Shape, Stride, OriginShape>> : true_type {};
template <class T>
inline constexpr bool is_layout_v = is_layout<T>::value;

// Layout construction
template <
    class Shape, class Stride, class OriginShape,
    TLA_REQUIRES((is_int_tuple_v<Shape>) && (is_int_tuple_v<Stride>) && (is_int_tuple_v<OriginShape>))>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape, Stride const& stride, OriginShape const& originShape)
{
    return Layout<Shape, Stride, OriginShape>(shape, stride, originShape);
}

// PascalCase alias
template <class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(Shape const& shape, Stride const& stride, OriginShape const& originShape)
{
    return make_layout(shape, stride, originShape);
}

template <class Shape, class Stride, TLA_REQUIRES((is_int_tuple_v<Shape>) && (is_int_tuple_v<Stride>))>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape, Stride const& stride)
{
    const auto originShape = detail::make_origin_shape(shape);
    return make_layout(shape, stride, originShape);
}

// PascalCase alias
template <class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(Shape const& shape, Stride const& stride)
{
    return make_layout(shape, stride);
}

template <class Shape, TLA_REQUIRES(is_int_tuple_v<Shape>)>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape)
{
    const auto stride = compact_col_major(shape);
    return make_layout(shape, stride);
}

template <class Layout0, class... Layouts, TLA_REQUIRES((is_layout_v<Layout0>) && (is_layout_v<Layouts> && ...))>
CATLASS_HOST_DEVICE constexpr auto make_layout(Layout0 const& layout0, Layouts const&... layouts)
{
    return make_layout(
        make_shape(layout0.shape(), layouts.shape()...), make_stride(layout0.stride(), layouts.stride()...),
        make_shape(layout0.originShape(), layouts.originShape()...));
}

template <class Shape, class Order>
CATLASS_HOST_DEVICE constexpr auto make_ordered_layout(Shape const& shape, Order const& order)
{
    return make_layout(shape, compact_order(shape, order));
}

template <class Shape, class Order, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto make_ordered_layout(
    Shape const& shape, Order const& order, OriginShape const& originShape)
{
    return make_layout(shape, compact_order(shape, order), originShape);
}

// Layout accessors (free functions)
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape>& layout)
{
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape> const& layout)
{
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape>& layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape> const& layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape>& layout)
{
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape> const& layout)
{
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto rank(Layout<Shape, Stride, OriginShape> const& layout)
{
    return rank(shape<Is...>(layout));
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto depth(Layout<Shape, Stride, OriginShape> const& layout)
{
    return depth(shape<Is...>(layout));
}

// Layout manipulation (like a tuple of pairs)

// get<Is...>(layout): extract sublayout at recursive indices Is...
// For Is... = <I0,I1,...,IN>, equivalent to get<IN>(...get<I1>(get<I0>(layout)))
template <size_t... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto get(Layout<Shape, Stride, OriginShape> const& layout)
{
    return make_layout(get<Is...>(shape(layout)), get<Is...>(stride(layout)), get<Is...>(originShape(layout)));
}

// take<B, E>(layout): extract sublayout with modes in range [B, E)
template <int B, int E, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto take(Layout<Shape, Stride, OriginShape> const& layout)
{
    return make_layout(take<B, E>(shape(layout)), take<B, E>(stride(layout)), take<B, E>(originShape(layout)));
}

// select<Is...>(layout): extract sublayout with modes at indices Is...
template <size_t... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto select(Layout<Shape, Stride, OriginShape> const& layout)
{
    return make_layout(select<Is...>(shape(layout)), select<Is...>(stride(layout)), select<Is...>(originShape(layout)));
}

// Codomain operations

// coshape = (shape - 1) * |stride| + 1 (per mode, recursive)
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto coshape(Layout<Shape, Stride, OriginShape> const& layout)
{
    auto m1_shapes = transform_leaf([](auto s) { return s - Int<1>{}; }, shape<Is...>(layout));
    auto abs_strides = transform_leaf(abs_fn{}, stride<Is...>(layout));
    auto co_coord = inner_product(m1_shapes, abs_strides);
    return transform_leaf([](auto c) { return c + Int<1>{}; }, co_coord);
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto cosize(Layout<Shape, Stride, OriginShape> const& layout)
{
    return size(coshape<Is...>(layout));
}

template <class Layout_>
using cosize_t = decltype(cosize(std::declval<Layout_>()));
// Tag-dispatched layout construction

namespace detail {

template <class Element>
struct element_alignment {
    static constexpr auto value = max(Catlass::BytesToElements<Element>(_1{}), _1{});
};

