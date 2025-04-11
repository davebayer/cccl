//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_SUB_OVERFLOW_H
#define _CUDA___NUMERIC_SUB_OVERFLOW_H

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
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

struct __sub_overflow
{
  template <class _Tp>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp>
  __impl_generic(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};

    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      using _Up         = _CUDA_VSTD::make_unsigned_t<_Tp>;
      __result.value    = static_cast<_Tp>(static_cast<_Up>(__lhs) - static_cast<_Up>(__rhs));
      __result.overflow = (__result.value < __lhs) == !(__rhs > _Tp{0});
    }
    else
    {
      __result.value    = static_cast<_Tp>(__lhs - __rhs);
      __result.overflow = __result.value > __lhs;
    }

    return __result;
  }

#if !_CCCL_COMPILER(NVRTC)
  template <class _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST static overflow_result<_Tp> __impl_host(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};

#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        __result.overflow = _sub_overflow_i8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = _sub_overflow_i16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = _sub_overflow_i32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = _sub_overflow_i64(0, __lhs, __rhs, &__result.value);
      }
      else
      {
        __result = __impl_generic(__lhs, __rhs);
      }
    }
    else
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        __result.overflow = _subborrow_u8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = _subborrow_u16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = _subborrow_u32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = _subborrow_u64(0, __lhs, __rhs, &__result.value);
      }
      else
      {
        __result = __impl_generic(__lhs, __rhs);
      }
    }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
    __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^

    return __result;
  }
#endif // !_CCCL_COMPILER(NVRTC)

  template <class _Tp>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp> __impl(_Tp __lhs, _Tp __rhs) noexcept
  {
#if defined(_CCCL_BUILTIN_SUB_OVERFLOW)
    overflow_result<_Tp> __result{};
    __result.overflow = _CCCL_BUILTIN_SUB_OVERFLOW(__lhs, __rhs, &__result.value);
    return __result;
#else // ^^^ _CCCL_BUILTIN_SUB_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_SUB_OVERFLOW vvv
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST, (return __impl_host(__lhs, __rhs);))
    }
    return __impl_generic(__lhs, __rhs);
#endif // !_CCCL_BUILTIN_SUB_OVERFLOW
  }
};

//! @brief Subtracts two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr overflow_result<_CUDA_VSTD::common_type_t<_Lhs, _Rhs>>
sub_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  using _Common  = _CUDA_VSTD::common_type_t<_Lhs, _Rhs>;
  using _UCommon = _CUDA_VSTD::make_unsigned_t<_Common>;

  // Safely cast to common type (common type may be unsigned, e. g. for int and unsigned int -> unsigned int)
  const auto __l = ::cuda::overflow_cast<_Common>(__lhs);
  const auto __r = ::cuda::overflow_cast<_Common>(__rhs);

  // Check for cast overflow
  if (__l.overflow || __r.overflow)
  {
    const auto __ul = static_cast<_UCommon>(__l.value);
    const auto __ur = static_cast<_UCommon>(__r.value);
    return {static_cast<_Common>(__ul - __ur), true};
  }

  return ::cuda::__sub_overflow::__impl(__l.value, __r.value);
}

//! @brief Subtracts two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
//! @param __result The result of the subtraction
_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
sub_overflow(const _Lhs __lhs, const _Rhs __rhs, _CUDA_VSTD::common_type_t<_Lhs, _Rhs>& __result) noexcept
{
  const auto __res = ::cuda::sub_overflow(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___NUMERIC_SUB_OVERFLOW_H
