//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_REPEAT_H
#define _CUDA_STD___BIT_BIT_REPEAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bit_trunc.h>
#include <cuda/__bit/bitmask.h>
#include <cuda/std/__bit/shl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_repeat_impl_generic(_Tp __v, int __n) noexcept(false)
{
  constexpr auto __width = __num_bits_v<_Tp>;
  if (__n >= __width)
  {
    return __v;
  }

  _Tp __ret = static_cast<_Tp>(__v & ::cuda::bitmask<_Tp>(0, __n));

  for (int __i = __n; __i < __width; __i *= 2)
  {
    __ret = (__ret << __i) | __ret;
  }
  return __ret;
}

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __bit_repeat_impl_device(const _Tp __v, const int __n) noexcept(false)
{
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    constexpr auto __width = unsigned{__num_bits_v<_Tp>};
    const auto __bounded_n = ::min(static_cast<unsigned>(__n), __width);

    uint32_t __ret = ::cuda::bit_trunc(__v, static_cast<int>(__bounded_n));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned __i = 1; __i < __width; __i *= 2)
    {
      __ret |= ::cuda::std::shl(__ret, __bounded_n * __i);
    }
    return static_cast<_Tp>(__ret);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    constexpr auto __width = unsigned{__num_bits_v<_Tp>};
    const auto __bounded_n = ::min(static_cast<unsigned>(__n), __width);

    auto __lo       = static_cast<uint32_t>(__v);
    const auto __hi = static_cast<uint32_t>(__v >> 32);

    // Clear all bits above __bounded_n in __lo.
    __lo = ::cuda::bit_trunc(__lo, static_cast<int>(__bounded_n));

    // Fill the low word; shl safely handles shifts >= 32.
    __lo |= ::cuda::std::shl(__lo, __bounded_n);
    __lo |= ::cuda::std::shl(__lo, __bounded_n * 2);
    __lo |= ::cuda::std::shl(__lo, __bounded_n * 4);
    __lo |= ::cuda::std::shl(__lo, __bounded_n * 8);
    __lo |= ::cuda::std::shl(__lo, __bounded_n * 16);

    // Scale widths <= 32 by a power of two to a period in [17, 32].
    // For larger widths, the unsigned overshift yields zero for this unused candidate.
    const auto __period = ::cuda::std::shl(__bounded_n, static_cast<uint32_t>(::__clz(__bounded_n - 1) - 27));

    // Continue the pattern from bit 32; clamping also handles a 32-bit period.
    const auto __tail     = ::__funnelshift_lc(__lo, 0u, __period);
    const auto __small_hi = ::__funnelshift_lc(__lo, __tail, __period);

    // For widths > 32, keep the original high pattern bits and start the next copy of __lo.
    const auto __shift    = 64 - __bounded_n;
    const auto __large_hi = ::__funnelshift_r(::cuda::std::shl(__hi, __shift), __lo, __shift);
    return (uint64_t{__bounded_n > 32 ? __large_hi : __small_hi} << 32) | __lo;
  }
  else if constexpr (sizeof(_Tp) == 16)
  {
    const auto __pattern_width = static_cast<unsigned>(__n);
    auto __word0               = ::cuda::bit_trunc(static_cast<uint32_t>(__v), __pattern_width);
    auto __word1               = static_cast<uint32_t>(__v >> 32);
    const auto __word2         = static_cast<uint32_t>(__v >> 64);
    const auto __word3         = static_cast<uint32_t>(__v >> 96);

    // Bound shift counts so doubling them cannot wrap for large pattern widths.
    const auto __repeat_width = ::min(__pattern_width, 32u);
    __word0 |= ::cuda::std::shl(__word0, __repeat_width);
    __word0 |= ::cuda::std::shl(__word0, __repeat_width * 2);
    __word0 |= ::cuda::std::shl(__word0, __repeat_width * 4);
    __word0 |= ::cuda::std::shl(__word0, __repeat_width * 8);
    __word0 |= ::cuda::std::shl(__word0, __repeat_width * 16);

    // For widths < 64, share a period in [32, 63] between the low and high halves.
    const auto __period      = ::cuda::std::shl(__pattern_width, static_cast<unsigned>(::__clz(__pattern_width) - 26));
    const auto __half_period = __period >> 1;
    const auto __tail        = ::__funnelshift_lc(__word0, 0u, __half_period);
    const auto __small_word1 = ::__funnelshift_lc(__word0, __tail, __half_period);
    const auto __shift32     = __pattern_width - 32;
    const auto __large_word1 = ::cuda::bit_trunc(__word1, __shift32) | ::cuda::std::shl(__word0, __shift32);
    __word1                  = __pattern_width >= 32 ? __large_word1 : __small_word1;

    // Wrapped counts continue the period without subtracting 32.
    const auto __small_word2 = ::__funnelshift_l(__word0, __word1, __period);
    const auto __small_word3 = ::__funnelshift_l(__word1, __small_word2, __period);

    // Preserve high input bits and insert the repeated prefix after them.
    const auto __shift64     = __pattern_width - 64;
    const auto __large_word2 = ::cuda::bit_trunc(__word2, __shift64) | ::cuda::std::shl(__word0, __shift64);
    const auto __shift96     = ::max(__n - 96, 0);
    const auto __large_word3 =
      ::cuda::bit_trunc(__word3, __shift96)
      | ::cuda::std::shl(::__funnelshift_lc(__word0, __word1, __shift64), static_cast<unsigned>(__shift96));
    const auto __result_word2 = __pattern_width >= 64 ? __large_word2 : __small_word2;
    const auto __result_word3 = __pattern_width >= 64 ? __large_word3 : __small_word3;
    return (_Tp{__result_word3} << 96) | (_Tp{__result_word2} << 64) | (_Tp{__word1} << 32) | __word0;
  }
  else
  {
    return static_cast<_Tp>(::cuda::std::__bit_repeat_impl_generic(__v, __n));
  }
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_repeat(const _Tp __v, const int __n) noexcept(false)
{
  // If __n <= 0, the function must fail during constant evaluation.
  _CCCL_IF_CONSTEVAL_DEFAULT
  {
    _CCCL_VERIFY(__n > 0, "__n must be greater than 0");
  }
  else
  {
    _CCCL_ASSERT(__n > 0, "__n must be greater than 0");
  }

  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE, ({ return ::cuda::std::__bit_repeat_impl_device(__v, __n); }))
  }
  return ::cuda::std::__bit_repeat_impl_generic(__v, __n);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_REPEAT_H
