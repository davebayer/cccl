//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_DIV_OVERFLOW_H
#define _CUDA___NUMERIC_DIV_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Divides two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr overflow_result<_CUDA_VSTD::common_type_t<_Lhs, _Rhs>>
div_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  using _Common = _CUDA_VSTD::common_type_t<_Lhs, _Rhs>;

  // Safely cast to common type (common type may be unsigned, e. g. for int and unsigned int -> unsigned int)
  const auto __l = ::cuda::overflow_cast<_Common>(__lhs);
  const auto __r = ::cuda::overflow_cast<_Common>(__rhs);

  overflow_result<_Common> __result{};
  __result.overflow = __l.overflow || __r.overflow;

  // Overflow detection
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Common))
  {
    if (__l == _CUDA_VSTD::numeric_limits<_Common>::min() && __r == _Common{-1})
    {
      __result.overflow = true;
    }
  }

  // Compute the result value if no overflow occurred or the common type is unsigned
  if (!__result.overflow || _CCCL_TRAIT(_CUDA_VSTD::is_unsigned, _Common))
  {
    __result.value = __l.value / __r.value;
  }

  return __result;
}

//! @brief Divides two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
//! @param __result The result of the division
_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
div_overflow(const _Lhs __lhs, const _Rhs __rhs, _CUDA_VSTD::common_type_t<_Lhs, _Rhs>& __result) noexcept
{
  const auto __res = ::cuda::div_overflow(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___NUMERIC_DIV_OVERFLOW_H
