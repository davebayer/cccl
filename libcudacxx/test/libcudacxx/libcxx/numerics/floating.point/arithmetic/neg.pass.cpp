//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(23) // integer constant is too large

enum class UnsignedFormatTag
{
};

template <cuda::std::__fp_format Fmt>
struct RefNegValues
{
  cuda::std::__fp_storage_t<Fmt> neg_zero;
  cuda::std::__fp_storage_t<Fmt> neg_one;
  cuda::std::__fp_storage_t<Fmt> neg_inf;
  cuda::std::__fp_storage_t<Fmt> neg_nan;
};

template <cuda::std::__fp_format Fmt>
__host__ __device__ constexpr RefNegValues<Fmt> test_make_ref_values()
{
  using namespace test_integer_literals;

  RefNegValues<Fmt> ret{};

  if constexpr (Fmt == cuda::std::__fp_format::__binary16)
  {
    ret.neg_zero = 0x8000u;
    ret.neg_one  = 0xbc00u;
    ret.neg_inf  = 0xfc00u;
    ret.neg_nan  = 0xfe00u;
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__binary32)
  {
    ret.neg_zero = 0x80000000u;
    ret.neg_one  = 0xbf800000u;
    ret.neg_inf  = 0xff800000u;
    ret.neg_nan  = 0xffc00000u;
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__binary64)
  {
    ret.neg_zero = 0x8000000000000000ull;
    ret.neg_one  = 0xbff0000000000000ull;
    ret.neg_inf  = 0xfff0000000000000ull;
    ret.neg_nan  = 0xfff8000000000000ull;
  }
#if _CCCL_HAS_INT128()
  else if constexpr (Fmt == cuda::std::__fp_format::__binary128)
  {
    ret.neg_zero = 0x80000000000000000000000000000000_u128;
    ret.neg_one  = 0xbfff0000000000000000000000000000_u128;
    ret.neg_inf  = 0xffff0000000000000000000000000000_u128;
    ret.neg_nan  = 0xffff8000000000000000000000000000_u128;
  }
#endif // _CCCL_HAS_INT128()
  else if constexpr (Fmt == cuda::std::__fp_format::__bfloat16)
  {
    ret.neg_zero = 0x8000u;
    ret.neg_one  = 0xbf80u;
    ret.neg_inf  = 0xff80u;
    ret.neg_nan  = 0xffc0u;
  }
#if _CCCL_HAS_INT128()
  else if constexpr (Fmt == cuda::std::__fp_format::__fp80_x86)
  {
    ret.neg_zero = 0x80000000000000000000_u128;
    ret.neg_one  = 0xbfff8000000000000000_u128;
    ret.neg_inf  = 0xffff8000000000000000_u128;
    ret.neg_nan  = 0xffffc000000000000000_u128;
  }
#endif // _CCCL_HAS_INT128()
  else if constexpr (Fmt == cuda::std::__fp_format::__fp8_nv_e4m3)
  {
    ret.neg_zero = 0x80u;
    ret.neg_one  = 0xb8u;
    // no infinity
    ret.neg_nan = 0xffu;
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__fp8_nv_e5m2)
  {
    ret.neg_zero = 0x80u;
    ret.neg_one  = 0xbcu;
    ret.neg_inf  = 0xfcu;
    ret.neg_nan  = 0xfeu;
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__fp8_nv_e8m0)
  {
    // the format is unsigned
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__fp6_nv_e2m3)
  {
    ret.neg_zero = 0x20u;
    ret.neg_one  = 0x28u;
    // no infinity
    // no NaN
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__fp6_nv_e3m2)
  {
    ret.neg_zero = 0x20u;
    ret.neg_one  = 0x2cu;
    // no infinity
    // no NaN
  }
  else if constexpr (Fmt == cuda::std::__fp_format::__fp4_nv_e2m1)
  {
    ret.neg_zero = 0x8u;
    ret.neg_one  = 0xau;
    // no infinity
    // no NaN
  }

  return ret;
}

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_neg()
{
  constexpr auto ref_values = test_make_ref_values<Fmt>();

  // 1. Test negation of 0
  const auto neg_zero = cuda::std::__fp_neg<Fmt>(cuda::std::__fp_zero<Fmt>());
  assert(neg_zero == ref_values.neg_zero);
  const auto pos_zero = cuda::std::__fp_neg<Fmt>(neg_zero);
  assert(pos_zero == cuda::std::__fp_zero<Fmt>());

  // 2. Test negation of 1
  const auto neg_one = cuda::std::__fp_neg<Fmt>(cuda::std::__fp_one<Fmt>());
  assert(neg_one == ref_values.neg_one);
  const auto pos_one = cuda::std::__fp_neg<Fmt>(neg_one);
  assert(pos_one == cuda::std::__fp_one<Fmt>());

  // 3. Test negation of inf
  if constexpr (cuda::std::__fp_has_inf_v<Fmt>)
  {
    const auto neg_inf = cuda::std::__fp_neg<Fmt>(cuda::std::__fp_inf<Fmt>());
    assert(neg_inf == ref_values.neg_inf);
    const auto pos_inf = cuda::std::__fp_neg<Fmt>(neg_inf);
    assert(pos_inf == cuda::std::__fp_inf<Fmt>());
  }

  // 4. Test negation of nan
  if constexpr (cuda::std::__fp_has_nan_v<Fmt>)
  {
    const auto neg_nan = cuda::std::__fp_neg<Fmt>(cuda::std::__fp_nan<Fmt>());
    assert(neg_nan == ref_values.neg_nan);
    const auto pos_nan = cuda::std::__fp_neg<Fmt>(neg_nan);
    assert(pos_nan == cuda::std::__fp_nan<Fmt>());
  }
}

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_neg(UnsignedFormatTag)
{
  static_assert(!cuda::std::__fp_is_signed_v<Fmt>);
}

template <class T>
__host__ __device__ void test_fp_neg()
{
  constexpr auto fmt        = cuda::std::__fp_format_of_v<T>;
  constexpr auto ref_values = test_make_ref_values<fmt>();

  // 1. Test negation of 0
  const auto neg_zero = cuda::std::__fp_neg(cuda::std::__fp_zero<T>());
  assert(cuda::std::__fp_get_storage(neg_zero) == ref_values.neg_zero);
  const auto pos_zero = cuda::std::__fp_neg(neg_zero);
  assert(cuda::std::__fp_get_storage(pos_zero) == cuda::std::__fp_zero<fmt>());

  // 2. Test negation of 1
  const auto neg_one = cuda::std::__fp_neg(cuda::std::__fp_one<T>());
  assert(cuda::std::__fp_get_storage(neg_one) == ref_values.neg_one);
  const auto pos_one = cuda::std::__fp_neg(neg_one);
  assert(cuda::std::__fp_get_storage(pos_one) == cuda::std::__fp_one<fmt>());

  // 3. Test negation of inf
  if constexpr (cuda::std::__fp_has_inf_v<fmt>)
  {
    const auto neg_inf = cuda::std::__fp_neg(cuda::std::__fp_inf<T>());
    assert(cuda::std::__fp_get_storage(neg_inf) == ref_values.neg_inf);
    const auto pos_inf = cuda::std::__fp_neg(neg_inf);
    assert(cuda::std::__fp_get_storage(pos_inf) == cuda::std::__fp_inf<fmt>());
  }

  // 4. Test negation of nan
  if constexpr (cuda::std::__fp_has_nan_v<fmt>)
  {
    const auto neg_nan = cuda::std::__fp_neg(cuda::std::__fp_nan<T>());
    assert(cuda::std::__fp_get_storage(neg_nan) == ref_values.neg_nan);
    const auto pos_nan = cuda::std::__fp_neg(neg_nan);
    assert(cuda::std::__fp_get_storage(pos_nan) == cuda::std::__fp_nan<fmt>());
  }

  // 5. Test that __fp_neg is constexpr
  static_assert(((void) cuda::std::__fp_neg(cuda::std::__fp_one<T>()), true));
}

template <class T>
__host__ __device__ void test_fp_neg(UnsignedFormatTag)
{
  static_assert(!cuda::std::__fp_is_signed_v<cuda::std::__fp_format_of_v<T>>);
}

__host__ __device__ void test()
{
  // 1. Test formats
  test_fp_neg<cuda::std::__fp_format::__binary16>();
  test_fp_neg<cuda::std::__fp_format::__binary32>();
  test_fp_neg<cuda::std::__fp_format::__binary64>();
#if _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__fp_format::__binary128>();
#endif // _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__fp_format::__bfloat16>();
#if _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__fp_format::__fp80_x86>();
#endif // _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__fp_format::__fp8_nv_e4m3>();
  test_fp_neg<cuda::std::__fp_format::__fp8_nv_e5m2>();
  test_fp_neg<cuda::std::__fp_format::__fp8_nv_e8m0>(UnsignedFormatTag{});
  test_fp_neg<cuda::std::__fp_format::__fp6_nv_e2m3>();
  test_fp_neg<cuda::std::__fp_format::__fp6_nv_e3m2>();
  test_fp_neg<cuda::std::__fp_format::__fp4_nv_e2m1>();

  // 2. Test standard floating point types
  test_fp_neg<float>();
  test_fp_neg<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_neg<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  // 3. Test extended nvidia floating point types
#if _CCCL_HAS_NVFP16()
  test_fp_neg<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_neg<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_neg<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_neg<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_neg<__nv_fp8_e8m0>(UnsignedFormatTag{});
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_neg<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_neg<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_neg<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  // 4. Test extended compiler floating point types
#if _CCCL_HAS_FLOAT128()
  test_fp_neg<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  // 5. Test extended cccl floating point types
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary16>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary32>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary64>>();
#if _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary128>>();
#endif // _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__bfloat16>>();
#if _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp80_x86>>();
#endif // _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e4m3>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e5m2>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e8m0>>(UnsignedFormatTag{});
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e2m3>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e3m2>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp4_nv_e2m1>>();
}

int main(int, char**)
{
  test();
  return 0;
}
