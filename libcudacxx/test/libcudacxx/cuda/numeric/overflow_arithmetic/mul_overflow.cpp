//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_mul_overflow(T lhs, T rhs, T expected_result, bool overflow)
{
  // test bool mul_overflow(T lhs, T rhs, T& result) overload
  {
    T result{};
    const bool has_overflow = cuda::mul_overflow(lhs, rhs, result);

    // overflow result is well-defined only for unsigned types
    if (!overflow || cuda::std::is_unsigned_v<T>)
    {
      assert(result == expected_result);
    }
    assert(has_overflow == overflow);
  }

  // test overflow_result<T> mul_overflow(T lhs, T rhs) overload
  {
    const auto result = cuda::mul_overflow(lhs, rhs);

    // overflow result is well-defined only for unsigned types
    if (!overflow || cuda::std::is_unsigned_v<T>)
    {
      assert(result.value == expected_result);
    }
    assert(result.overflow == overflow);
  }
}

template <class T>
__host__ __device__ constexpr bool test_type()
{
  constexpr auto max = cuda::std::numeric_limits<T>::max();
  constexpr auto min = cuda::std::numeric_limits<T>::min();

  ASSERT_SAME_TYPE(decltype(cuda::mul_overflow(T{}, T{}, cuda::std::declval<T&>())), bool);
  static_assert(noexcept(cuda::mul_overflow(T{}, T{}, cuda::std::declval<T&>())));
  ASSERT_SAME_TYPE(decltype(cuda::mul_overflow(T{}, T{})), cuda::overflow_result<T>);
  static_assert(noexcept(cuda::mul_overflow(T{}, T{})));

  test_mul_overflow<T>(T{0}, T{0}, T{0}, false);
  test_mul_overflow<T>(min, T{1}, min, false);

  return true;
}

__host__ __device__ constexpr bool test()
{
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
#if !defined(TEST_HAS_NO_INT128_T)
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // !TEST_HAS_NO_INT128_T

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());

  return 0;
}
