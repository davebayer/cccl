//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numeric>

// template<class T>
// constexpr T sub_sat(T x, T y) noexcept;                     // freestanding

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/numeric>

#include "test_macros.h"

template <class I>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_sub_sat(I x, I y, I res, I zero_value)
{
  assert(cuda::std::sub_sat(static_cast<I>(zero_value + x), static_cast<I>(zero_value + y)) == res);
  return true;
}

template <class I>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_signed(I zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  ASSERT_SAME_TYPE(I, decltype(cuda::std::sub_sat(I{}, I{})));
  static_assert(noexcept(cuda::std::sub_sat(I{}, I{})), "");

  // Limit values (-1, 0, 1, min, max)

  test_sub_sat<I>(I{-1}, I{-1}, I{0}, zero_value);
  test_sub_sat<I>(I{-1}, I{0}, I{-1}, zero_value);
  test_sub_sat<I>(I{-1}, I{1}, I{-2}, zero_value);
  test_sub_sat<I>(I{-1}, minVal, I{-1} - minVal, zero_value);
  test_sub_sat<I>(I{-1}, maxVal, I{-1} - maxVal, zero_value);

  test_sub_sat<I>(I{0}, I{-1}, I{1}, zero_value);
  test_sub_sat<I>(I{0}, I{0}, I{0}, zero_value);
  test_sub_sat<I>(I{0}, I{1}, I{-1}, zero_value);
  test_sub_sat<I>(I{0}, minVal, maxVal, zero_value); // saturated
  test_sub_sat<I>(I{0}, maxVal, -maxVal, zero_value);

  test_sub_sat<I>(minVal, I{-1}, minVal - I{-1}, zero_value);
  test_sub_sat<I>(minVal, I{0}, minVal, zero_value);
  test_sub_sat<I>(minVal, I{1}, minVal, zero_value); // saturated
  test_sub_sat<I>(minVal, minVal, I{0}, zero_value);
  test_sub_sat<I>(minVal, maxVal, minVal, zero_value); // saturated

  test_sub_sat<I>(maxVal, I{-1}, maxVal, zero_value); // saturated
  test_sub_sat<I>(maxVal, I{0}, maxVal, zero_value);
  test_sub_sat<I>(maxVal, I{1}, maxVal - I{1}, zero_value);
  test_sub_sat<I>(maxVal, minVal, maxVal, zero_value); // saturated
  test_sub_sat<I>(maxVal, maxVal, I{0}, zero_value);

  // No saturation (no limit values)

  test_sub_sat<I>(I{27}, I{-28}, I{55}, zero_value);
  test_sub_sat<I>(I{27}, I{28}, I{-1}, zero_value);
  test_sub_sat<I>(I{-27}, I{28}, I{-55}, zero_value);
  test_sub_sat<I>(I{-27}, I{-28}, I{1}, zero_value);

  // Saturation (no limit values)

  {
    constexpr I lesserVal = minVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_sub_sat<I>(lesserVal, biggerVal, minVal, zero_value); // saturated
  }
  {
    constexpr I biggerVal = maxVal / I{2} + I{28};
    constexpr I lesserVal = minVal / I{2} + I{27};
    test_sub_sat<I>(biggerVal, lesserVal, maxVal, zero_value); // saturated
  }

  return true;
}

template <class I>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_unsigned(I zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  ASSERT_SAME_TYPE(I, decltype(cuda::std::sub_sat(I{}, I{})));
  static_assert(noexcept(cuda::std::sub_sat(I{}, I{})), "");

  // Limit values (0, 1, min, max)

  test_sub_sat<I>(I{0}, I{0}, I{0}, zero_value);
  test_sub_sat<I>(I{0}, I{1}, minVal, zero_value); // saturated
  test_sub_sat<I>(I{0}, minVal, minVal, zero_value);
  test_sub_sat<I>(I{0}, maxVal, minVal, zero_value); // saturated

  test_sub_sat<I>(I{1}, I{0}, I{1}, zero_value);
  test_sub_sat<I>(I{1}, I{1}, I{0}, zero_value);
  test_sub_sat<I>(I{1}, minVal, I{1}, zero_value);
  test_sub_sat<I>(I{1}, maxVal, minVal, zero_value); // saturated

  test_sub_sat<I>(minVal, I{0}, I{0}, zero_value);
  test_sub_sat<I>(minVal, I{1}, minVal, zero_value);
  test_sub_sat<I>(minVal, maxVal, minVal, zero_value);
  test_sub_sat<I>(minVal, maxVal, minVal, zero_value);

  test_sub_sat<I>(maxVal, I{0}, maxVal, zero_value);
  test_sub_sat<I>(maxVal, I{1}, maxVal - I{1}, zero_value);
  test_sub_sat<I>(maxVal, minVal, maxVal, zero_value);
  test_sub_sat<I>(maxVal, maxVal, I{0}, zero_value);

  // Saturation (no limit values)

  {
    constexpr I lesserVal = minVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_sub_sat<I>(lesserVal, biggerVal, minVal, zero_value); // saturated
  }

  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test(int zero_value)
{
  test_signed<signed char>(static_cast<signed char>(zero_value));
  test_signed<short int>(static_cast<short int>(zero_value));
  test_signed<int>(static_cast<int>(zero_value));
  test_signed<long int>(static_cast<long int>(zero_value));
  test_signed<long long int>(static_cast<long long int>(zero_value));
#ifndef TEST_HAS_NO_INT128_T
  test_signed<__int128_t>(static_cast<__int128_t>(zero_value));
#endif // TEST_HAS_NO_INT128_T

  test_unsigned<unsigned char>(static_cast<unsigned char>(zero_value));
  test_unsigned<unsigned short int>(static_cast<unsigned short int>(zero_value));
  test_unsigned<unsigned int>(static_cast<unsigned int>(zero_value));
  test_unsigned<unsigned long int>(static_cast<unsigned long int>(zero_value));
  test_unsigned<unsigned long long int>(static_cast<unsigned long long int>(zero_value));
#ifndef TEST_HAS_NO_INT128_T
  test_unsigned<__uint128_t>(static_cast<__uint128_t>(zero_value));
#endif // TEST_HAS_NO_INT128_T

  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
#if TEST_STD_VER >= 2014
  static_assert(test(0), "");
#endif // TEST_STD_VER >= 2014
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
#if TEST_STD_VER >= 2014
  static_assert(test(0), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
