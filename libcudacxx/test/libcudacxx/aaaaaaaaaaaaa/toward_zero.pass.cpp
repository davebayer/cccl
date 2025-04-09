//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/numeric>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

inline constexpr auto round_kind = cuda::std::__fp_round_kind::__toward_zero;

template <size_t NBits>
using test_make_nbit_uint_t = cuda::std::__make_nbit_uint_t<cuda::std::bit_ceil(cuda::ceil_div(NBits, 8)) * 8>;

template <size_t ToNBits,
          class ToMant,
          size_t FromNBits,
          class FromMant,
          cuda::std::enable_if_t<(ToNBits >= FromNBits), int> = 0>
__host__ __device__ constexpr void test_fp_rounding_toward_zero(const FromMant& input)
{
  ToMant ref_result{};

  if constexpr (FromNBits != 0)
  {
    ref_result = static_cast<ToMant>(input) << (ToNBits - FromNBits);
  }

  // 1. Test positive input
  {
    const auto result = cuda::std::__fp_round<round_kind, ToNBits, ToMant, FromNBits>(input, false);
    assert(result.value == ref_result);
    assert(result.overflow == false);
  }

  // 2. Test negative input
  {
    const auto result = cuda::std::__fp_round<round_kind, ToNBits, ToMant, FromNBits>(input, true);
    assert(result.value == ref_result);
    assert(result.overflow == false);
  }
}

template <size_t ToNBits,
          class ToMant,
          size_t FromNBits,
          class FromMant,
          cuda::std::enable_if_t<(ToNBits < FromNBits), int> = 0>
__host__ __device__ constexpr void test_fp_rounding_toward_zero(const FromMant& input)
{
  ToMant ref_result{};

  if constexpr (ToNBits != 0)
  {
    ref_result = static_cast<ToMant>(input >> (FromNBits - ToNBits));
  }

  // 1. Test positive input
  {
    const auto result = cuda::std::__fp_round<round_kind, ToNBits, ToMant, FromNBits>(input, false);
    assert(result.value == ref_result);
    assert(result.overflow == false);
  }

  // 2. Test negative input
  {
    const auto result = cuda::std::__fp_round<round_kind, ToNBits, ToMant, FromNBits>(input, true);
    assert(result.value == ref_result);
    assert(result.overflow == false);
  }
}

template <size_t ToNBits, size_t FromNBits>
__host__ __device__ constexpr void test()
{
  using ToMant   = test_make_nbit_uint_t<ToNBits>;
  using FromMant = test_make_nbit_uint_t<FromNBits>;

  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::__fp_round<round_kind, ToNBits, ToMant, FromNBits>(FromMant{}, bool{})),
                         ::cuda::overflow_result<ToMant>>);
  static_assert(noexcept(cuda::std::__fp_round<round_kind, ToNBits, ToMant, FromNBits>(FromMant{}, bool{})));

  // 1. Test all zeros
  {
    const FromMant input{0};
    test_fp_rounding_toward_zero<ToNBits, ToMant, FromNBits>(input);
  }

  // 2. Test all ones
  if constexpr (FromNBits > 0)
  {
    const FromMant input = static_cast<FromMant>((FromMant{1} << (FromNBits - 1)) - 1);
    test_fp_rounding_toward_zero<ToNBits, ToMant, FromNBits>(input);
  }
}

template <size_t ToNBits>
__host__ __device__ constexpr void test()
{
  test<ToNBits, 0>(); // fp8_nv_e8m0
  test<ToNBits, 1>(); // fp4_nv_e2m1
  test<ToNBits, 2>(); // fp8_nv_e5m2, fp6_nv_e3m2
  test<ToNBits, 3>(); // fp8_nv_e4m3, fp6_nv_e2m3
  test<ToNBits, 7>(); // bfloat16
  test<ToNBits, 10>(); // binary16
  test<ToNBits, 23>(); // binary32
  test<ToNBits, 52>(); // binary64
  test<ToNBits, 63>(); // fp80_x86
#if _CCCL_HAS_INT128
  test<ToNBits, 112>(); // binary128
#endif // _CCCL_HAS_INT128

  test<ToNBits, 8>(); // 8-bit integers
  test<ToNBits, 16>(); // 16-bit integers
  test<ToNBits, 32>(); // 32-bit integers
  test<ToNBits, 64>(); // 64-bit integers
#if _CCCL_HAS_INT128
  test<ToNBits, 128>(); // 128-bit integers
#endif // _CCCL_HAS_INT128
}

__host__ __device__ constexpr bool test()
{
  test<0>(); // fp8_nv_e8m0
  test<1>(); // fp4_nv_e2m1
  test<2>(); // fp8_nv_e5m2, fp6_nv_e3m2
  test<3>(); // fp8_nv_e4m3, fp6_nv_e2m3
  test<7>(); // bfloat16
  test<10>(); // binary16
  test<23>(); // binary32
  test<52>(); // binary64
  test<63>(); // fp80_x86
#if _CCCL_HAS_INT128
  test<112>(); // binary128
#endif // _CCCL_HAS_INT128

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
