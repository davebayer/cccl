//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_CLIP_H
#define _CUDA___CMATH_CLIP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
inline constexpr bool __is_clip_type_v =
  ::cuda::std::is_arithmetic_v<_Tp> || ::cuda::std::__is_ext_compiler_fp_v<_Tp>
#if _CCCL_HAS_NVFP16()
  || ::cuda::std::is_same_v<_Tp, ::__half>
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  || ::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>
#endif // _CCCL_HAS_NVBF16()
  ;

//! @brief Clamps a numeric value to the inclusive range, returning by value.
//! @param __v The input value.
//! @param __lo The lower bound.
//! @param __hi The upper bound.
//! @pre __lo <= __hi. Floating-point bounds must not be NaN.
//! @return For integers, __lo if __v < __lo, __hi if __hi < __v, otherwise __v.
//! For floating-point types, fmin(__hi, fmax(__lo, __v)); a NaN input returns __lo.
//! @note Unlike cuda::std::clamp, this function takes and returns values, not references.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::__is_clip_type_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp clip(const _Tp __v, const _Tp __lo, const _Tp __hi) noexcept
{
  if constexpr (::cuda::std::__is_fp_v<_Tp>)
  {
    return ::cuda::std::fmin(__hi, ::cuda::std::fmax(__lo, __v));
  }
  else
  {
    // We need to use a temporary here to make sure nvcc compiles this to min and max PTX instructions.
    const auto __tmp = (__hi < __v) ? __hi : __v;
    return (__tmp < __lo) ? __lo : __tmp;
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_CLIP_H
