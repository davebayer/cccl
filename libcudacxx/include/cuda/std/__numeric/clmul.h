//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_CLMUL_H
#define _CUDA_STD___NUMERIC_CLMUL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __clmul_impl_generic(_Tp __x, _Tp __y) noexcept
{
  for (int __i = 0; __i < __num_bits_v<_Tp>;)
  {
  }
}

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __clmul_impl_sm80(_Tp __x, _Tp __y) noexcept
{
#  if __cccl_ptx_isa >= 930
  if constexpr (sizeof(_Tp) <= sizeof(uint64_t))
  {
    uint64_t __ret;
    asm("clmad.lo.u64 %0, %1, %2, 0;" : "=l"(__ret) : "l"(uint64_t{__x}), "l"(uint64_t{__y}));
    return static_cast<_Tp>(__ret);
  }
#    if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __x_lo = static_cast<uint64_t>(__x);
    const auto __x_hi = static_cast<uint64_t>(__x >> 64);
    const auto __y_lo = static_cast<uint64_t>(__y);
    const auto __y_hi = static_cast<uint64_t>(__y >> 64);

    uint64_t __ret_lo;
    uint64_t __ret_hi;

    asm("clmad.lo.u64 %0, %1, %2, 0;" : "=l"(__ret_lo) : "l"(__x_lo), "l"(__y_lo));
    asm("clmad.hi.u64 %0, %1, %2, 0;" : "=l"(__ret_hi) : "l"(__x_lo), "l"(__y_lo));
    asm("clmad.lo.u64 %0, %1, %2, %0;" : "+l"(__ret_hi) : "l"(__x_lo), "l"(__y_hi));
    asm("clmad.lo.u64 %0, %1, %2, %0;" : "+l"(__ret_hi) : "l"(__x_hi), "l"(__y_lo));
    return (__uint128_t{__ret_hi} << 64) | __ret_lo;
  }
#    endif // _CCCL_HAS_INT128()
  else
#  endif // __cccl_ptx_isa >= 930
  {
    return ::cuda::std::__clmul_impl_generic(__x, __y);
  }
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp clmul(_Tp __x, _Tp __y) noexcept
{
#if !_CCCL_TILE_COMPILATION()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, ({ return ::cuda::std::__clmul_impl_sm80(__x, __y); }))
  }
#endif // !_CCCL_TILE_COMPILATION()

  return ::cuda::std::__clmul_impl_generic(__x, __y);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_CLMUL_H
