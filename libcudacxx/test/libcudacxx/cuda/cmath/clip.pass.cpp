//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T, class = void>
inline constexpr bool can_clip = false;

template <class T>
inline constexpr bool can_clip<T, cuda::std::void_t<decltype(cuda::clip(T{}, T{}, T{}))>> = true;

struct non_numeric
{
  TEST_FUNC constexpr bool operator<(non_numeric) const
  {
    return false;
  }
};

static_assert(!can_clip<non_numeric>);
static_assert(!can_clip<int*>);

template <class T>
TEST_FUNC constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::clip(T{}, T{}, T{})), T>);
  static_assert(cuda::std::is_same_v<decltype(cuda::clip(cuda::std::declval<const T&>(), T{}, T{})), T>);
  static_assert(noexcept(cuda::clip(T{}, T{}, T{})));

  assert(cuda::clip(T(0), T(1), T(3)) == T(1));
  assert(cuda::clip(T(1), T(1), T(3)) == T(1));
  assert(cuda::clip(T(2), T(1), T(3)) == T(2));
  assert(cuda::clip(T(3), T(1), T(3)) == T(3));
  assert(cuda::clip(T(4), T(1), T(3)) == T(3));
  assert(cuda::clip(T(0), T(2), T(2)) == T(2));
  assert(cuda::clip(T(2), T(2), T(2)) == T(2));
  assert(cuda::clip(T(4), T(2), T(2)) == T(2));

  constexpr auto lowest = cuda::std::numeric_limits<T>::lowest();
  constexpr auto max    = cuda::std::numeric_limits<T>::max();
  assert(cuda::clip(lowest, lowest, max) == lowest);
  assert(cuda::clip(max, lowest, max) == max);
  assert(cuda::clip(lowest, T(1), T(3)) == T(1));
  assert(cuda::clip(max, T(1), T(3)) == T(3));

  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(cuda::clip(T(-4), T(-3), T(-1)) == T(-3));
    assert(cuda::clip(T(-2), T(-3), T(-1)) == T(-2));
    assert(cuda::clip(T(0), T(-3), T(-1)) == T(-1));
  }

  // Values above double's exact integer range must retain all their bits.
  if constexpr (cuda::std::is_integral_v<T> && cuda::std::numeric_limits<T>::digits > 53)
  {
    constexpr T large = (T(1) << 53) + 1;
    assert(cuda::clip(large, T(0), max) == large);
    assert(cuda::clip(large - 1, large, max) == large);
    assert(cuda::clip(max, T(0), large) == large);
    if constexpr (cuda::std::is_signed_v<T>)
    {
      assert(cuda::clip(-large, lowest, T(0)) == -large);
      assert(cuda::clip(lowest, -large, T(0)) == -large);
    }
  }
}

TEST_FUNC constexpr bool test()
{
  test_type<char>();
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
  test_type<wchar_t>();
  test_type<char16_t>();
  test_type<char32_t>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_type<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  static_assert(cuda::std::is_same_v<decltype(cuda::clip(false, false, true)), bool>);
  assert(!cuda::clip(false, false, true));
  assert(cuda::clip(true, false, true));
  assert(!cuda::clip(true, false, false));
  assert(cuda::clip(false, true, true));
  return true;
}

template <class T>
TEST_FUNC void check_clip(T value, T lo, T hi, T expected)
{
  DoNotOptimize(value);
  DoNotOptimize(lo);
  DoNotOptimize(hi);
  auto result   = cuda::clip(value, lo, hi);
  using Compare = cuda::std::conditional_t<(sizeof(T) < sizeof(float)), float, T>;
  assert(static_cast<Compare>(result) == static_cast<Compare>(expected));
}

template <class T>
TEST_FUNC void test_fp()
{
  static_assert(cuda::is_floating_point_v<T>);
  static_assert(cuda::std::is_same_v<decltype(cuda::clip(T{}, T{}, T{})), T>);
  static_assert(cuda::std::is_same_v<decltype(cuda::clip(cuda::std::declval<const T&>(), T{}, T{})), T>);
  static_assert(noexcept(cuda::clip(T{}, T{}, T{})));

  auto one  = T(1.0f);
  auto two  = T(2.0f);
  auto four = T(4.0f);
  auto half = T(0.5f);
  auto low  = cuda::std::numeric_limits<T>::lowest();
  auto high = cuda::std::numeric_limits<T>::max();
  check_clip(half, one, four, one);
  check_clip(one, one, four, one);
  check_clip(two, one, four, two);
  check_clip(four, one, four, four);
  check_clip(high, one, four, four);
  check_clip(half, two, two, two);
  check_clip(two, two, two, two);
  check_clip(four, two, two, two);
  check_clip(low, low, high, low);
  check_clip(high, low, high, high);
  check_clip(low, one, four, one);
  check_clip(high, one, four, four);
  check_clip(half, T(0.25f), one, half);

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    check_clip(T(-4.0f), T(-2.0f), T(-1.0f), T(-2.0f));
    check_clip(T(-2.0f), T(-4.0f), T(-1.0f), T(-2.0f));
    check_clip(one, T(-4.0f), T(-1.0f), T(-1.0f));

    auto zero     = T(0.0f);
    auto neg_zero = T(-0.0f);
    // The sign of equal zeros follows the underlying fmin/fmax implementation.
    check_clip(neg_zero, zero, one, zero);
    check_clip(zero, T(-1.0f), neg_zero, zero);
    assert(cuda::std::signbit(cuda::clip(T(-1.0f), neg_zero, one)));
    assert(!cuda::std::signbit(cuda::clip(T(-1.0f), zero, one)));
    assert(cuda::std::signbit(cuda::clip(one, T(-1.0f), neg_zero)));
    assert(!cuda::std::signbit(cuda::clip(one, T(-1.0f), zero)));
  }
  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    auto inf     = cuda::std::numeric_limits<T>::infinity();
    auto neg_inf = cuda::std::copysign(inf, T(-1.0f));
    check_clip(inf, one, four, four);
    check_clip(neg_inf, one, four, one);
    check_clip(inf, neg_inf, inf, inf);
    check_clip(neg_inf, neg_inf, inf, neg_inf);
  }
  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    check_clip(cuda::std::numeric_limits<T>::quiet_NaN(), one, four, one);
    check_clip(cuda::std::numeric_limits<T>::quiet_NaN(), two, two, two);
  }
  if constexpr (cuda::std::numeric_limits<T>::has_signaling_NaN)
  {
    check_clip(cuda::std::numeric_limits<T>::signaling_NaN(), one, four, one);
  }
}

int main(int, char**)
{
  test();
  static_assert(test());
  test_fp<float>();
  test_fp<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_fp<__float128>();
#endif // _CCCL_HAS_FLOAT128()
#if _CCCL_HAS_NVFP16()
  test_fp<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  static_assert(!can_clip<__nv_fp8_e4m3>);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  static_assert(!can_clip<__nv_fp8_e5m2>);
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  static_assert(!can_clip<__nv_fp8_e8m0>);
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  static_assert(!can_clip<__nv_fp6_e2m3>);
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  static_assert(!can_clip<__nv_fp6_e3m2>);
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  static_assert(!can_clip<__nv_fp4_e2m1>);
#endif // _CCCL_HAS_NVFP4_E2M1()
  return 0;
}
