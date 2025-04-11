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
#include <cuda/std/utility>

#include "test_macros.h"

template <class Lhs, class Rhs>
__host__ __device__ constexpr void test_sub_overflow(Lhs lhs, Rhs rhs, bool overflow)
{
  using Result = cuda::std::common_type_t<Lhs, Rhs>;

  // test overflow_result<Result> sub_overflow(Lhs lhs, Rhs rhs) overload
  {
    const auto result = cuda::sub_overflow(lhs, rhs);

    // overflow result is well-defined only for unsigned types
    if (!overflow || cuda::std::is_unsigned_v<Result>)
    {
      assert(result.value == static_cast<Result>(static_cast<Result>(lhs) - static_cast<Result>(rhs)));
    }
    assert(result.overflow == overflow);
  }

  // test bool sub_overflow(Lhs lhs, Rhs rhs, Result& result) overload
  {
    Result result{};
    const bool has_overflow = cuda::sub_overflow(lhs, rhs, result);

    // overflow result is well-defined only for unsigned types
    if (!overflow || cuda::std::is_unsigned_v<Result>)
    {
      assert(result == static_cast<Result>(static_cast<Result>(lhs) - static_cast<Result>(rhs)));
    }
    assert(has_overflow == overflow);
  }
}

template <class Lhs, class Rhs>
__host__ __device__ constexpr void test_type()
{
  using Result = cuda::std::common_type_t<Lhs, Rhs>;

  ASSERT_SAME_TYPE(decltype(cuda::sub_overflow(Lhs{}, Rhs{})), cuda::overflow_result<Result>);
  static_assert(noexcept(cuda::sub_overflow(Lhs{}, Rhs{})));

  ASSERT_SAME_TYPE(decltype(cuda::sub_overflow(Lhs{}, Rhs{}, cuda::std::declval<Result&>())), bool);
  static_assert(noexcept(cuda::sub_overflow(Lhs{}, Rhs{}, cuda::std::declval<Result&>())));

  // 1. Subtracting zeros should never overflow
  test_sub_overflow(Lhs{}, Rhs{}, false);

  // 2. Subtracting ones should never overflow
  test_sub_overflow(Lhs{1}, Rhs{1}, false);

  // 3. Subtracting zero and one should overflow if the destination type is unsigned
  test_sub_overflow(Lhs{}, Rhs{1}, cuda::std::is_unsigned_v<Result>);

  constexpr auto lhs_min    = cuda::std::numeric_limits<Lhs>::min();
  constexpr auto lhs_max    = cuda::std::numeric_limits<Lhs>::max();
  constexpr auto rhs_min    = cuda::std::numeric_limits<Rhs>::min();
  constexpr auto rhs_max    = cuda::std::numeric_limits<Rhs>::max();
  constexpr auto result_min = cuda::std::numeric_limits<Result>::min();
  constexpr auto result_max = cuda::std::numeric_limits<Result>::max();

  // 5. Subtracting max and zero
  test_sub_overflow(lhs_max, Rhs{}, cuda::std::cmp_less(-lhs_max, result_max));

  // 6. Subtracting zero and max
  test_sub_overflow(Lhs{}, rhs_max, cuda::std::cmp_greater(rhs_max, result_max));

  // 7. Subtracting max and minus one
  if constexpr (cuda::std::is_signed_v<Rhs>)
  {
    test_sub_overflow(lhs_max, Rhs{-1}, cuda::std::cmp_greater_equal(lhs_max, result_max));
  }

  // 8. Subtracting minus two and max
  if constexpr (cuda::std::is_signed_v<Lhs>)
  {
    test_sub_overflow(Lhs{-2}, rhs_max, cuda::std::cmp_less_equal(rhs_min, result_min));
  }

  // 9. Subtracting max and max
  test_sub_overflow(lhs_max, rhs_max, cuda::std::cmp_less(result_min + static_cast<Result>(lhs_max), rhs_max));

  // 10. Subtracting min and zero
  test_sub_overflow(lhs_min, Rhs{}, cuda::std::cmp_less(lhs_min, result_min));

  // 11. Subtracting zero and min
  test_sub_overflow(Lhs{}, rhs_min, cuda::std::cmp_less(rhs_min, result_min));

  // 12. Subtracting min and minus one
  if constexpr (cuda::std::is_signed_v<Rhs>)
  {
    test_sub_overflow(lhs_min, Rhs{-1}, cuda::std::cmp_less_equal(lhs_min, result_min));
  }

  // 13. Subtracting minus one and min
  if constexpr (cuda::std::is_signed_v<Lhs>)
  {
    test_sub_overflow(Lhs{-1}, rhs_min, cuda::std::cmp_less_equal(rhs_min, result_min));
  }

  // 14. Subtracting min and min
  test_sub_overflow(lhs_min, rhs_min, cuda::std::cmp_greater(result_min - static_cast<Result>(lhs_min), rhs_min));
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_type<T, signed char>();
  test_type<T, unsigned char>();
  test_type<T, short>();
  test_type<T, unsigned short>();
  test_type<T, int>();
  test_type<T, unsigned int>();
  test_type<T, long>();
  test_type<T, unsigned long>();
  test_type<T, long long>();
  test_type<T, unsigned long long>();
#if !defined(TEST_HAS_NO_INT128_T)
  test_type<T, __int128_t>();
  test_type<T, __uint128_t>();
#endif // !TEST_HAS_NO_INT128_T
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
