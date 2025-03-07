//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ISGREATER_H
#define _LIBCUDACXX___CMATH_ISGREATER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/fpclassify.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__internal/nvfp_types.h>
#include <cuda/std/__stdfloat/conversion_rank.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __isgreater_impl(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (_CCCL_TRAIT(is_floating_point, _Tp))
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return ::isgreater(__lhs, __rhs);),
                      (return !_CUDA_VSTD::isnan(__x) && !_CUDA_VSTD::isnan(__y) && __x > __y;))
  }
  else if constexpr (_CCCL_TRAIT(is_same_v, _Tp, __half) || _CCCL_TRAIT(is_same_v, _Tp, __nv_bfloat16))
  {
    return ::__hgt(__lhs, __rhs);
  }
  else
  {
    return _CUDA_VSTD::__isgreater_impl(
      _CUDA_VSTD::__cccl_fp_cvt_to<float>(__lhs), _CUDA_VSTD::__cccl_fp_cvt_to<float>(__rhs));
  }
}

_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES((_CCCL_TRAIT(is_arithmetic, _Lhs) || _CCCL_TRAIT(__is_extended_floating_point, _Lhs))
                 _CCCL_AND(_CCCL_TRAIT(is_arithmetic, _Rhs) || _CCCL_TRAIT(__is_extended_floating_point, _Rhs)))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isgreater(_Lhs __lhs, _Rhs __rhs) noexcept
{
  if constexpr (_CCCL_TRAIT(is_integral, _Lhs))
  {
    return _CUDA_VSTD::isgreater(static_cast<double>(__lhs), __rhs);
  }
  else if constexpr (_CCCL_TRAIT(is_integral, _Rhs))
  {
    return _CUDA_VSTD::isgreater(__lhs, static_cast<double>(__rhs));
  }
  else
  {
    constexpr auto __order = _CUDA_VSTD::__fp_make_conv_rank_order<_Lhs, _Rhs>();

    static_assert(__order != __fp_conv_rank_order::__unordered, "");

    if constexpr (__order == __fp_conv_rank_order::__greater)
    {
      return _CUDA_VSTD::isgreater(__lhs, _CUDA_VSTD::__cccl_fp_cvt_to<_Lhs>(__rhs));
    }
    else if constexpr (__order == __fp_conv_rank_order::__less)
    {
      return _CUDA_VSTD::isgreater(_CUDA_VSTD::__cccl_fp_cvt_to<_Rhs>(__lhs), __rhs);
    }
    else
    {
      return _CUDA_VSTD::__isgreater_impl(__lhs, __rhs);
    }
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ISGREATER_H
