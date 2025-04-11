//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_ADD_OVERFLOW_H
#define _CUDA___NUMERIC_ADD_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/uabs.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// same:
//   - s + s -> s
//   - u + u -> u
//   - s + u -> upcast(s), no check
//   - u + s -> upcast(s), no check
//
// different:
//   - s1 + s2 -> max(s1, s2)
//   - u1 + u2 -> max(u1, u2)
//   - s + u
//     - s > u -> s
//     - s < u -> upcast(make_signed(u)), no check
//   - u + s
//     - u > s -> upcasted(make_signed(u)), no check
//     - u < s -> s

template <class _Lhs, class _Rhs>
auto __add_overflow_type()
{
  constexpr bool __lhs_signed = _CCCL_TRAIT(_CUDA_VSTD::is_signed, _Lhs);
  constexpr bool __rhs_signed = _CCCL_TRAIT(_CUDA_VSTD::is_signed, _Rhs);

  if constexpr (sizeof(_Lhs) == sizeof(_Rhs))
  {
    if constexpr (__lhs_signed == __rhs_signed)
    {
      return _Lhs{};
    }
    else if constexpr (sizeof(_Lhs) < sizeof(_CUDA_VSTD::__intmax_t))
    {
      return _CUDA_VSTD::__int_t<sizeof(_Lhs) * 2>{};
    }
    else
    {
      return _CUDA_VSTD::__intmax_t{};
    }
  }
  else
  {
    if constexpr (__lhs_signed == __rhs_signed)
    {
      if constexpr (sizeof(_Lhs) > sizeof(_Rhs))
      {
        return _Lhs{};
      }
      else
      {
        return _Rhs{};
      }
    }
    else if constexpr (__lhs_signed)
    {
      return _Rhs{};
    }
    else
    {
      return _Lhs{};
    }
  }
}

struct __add_overflow
{
  template <class _Tp>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp>
  __impl_generic(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};

    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      using _Up         = _CUDA_VSTD::make_unsigned_t<_Tp>;
      __result.value    = static_cast<_Tp>(static_cast<_Up>(__lhs) + static_cast<_Up>(__rhs));
      __result.overflow = (__result.value < __lhs) == !(__rhs < static_cast<_Tp>(0));
    }
    else
    {
      __result.value    = static_cast<_Tp>(__lhs + __rhs);
      __result.overflow = __result.value < __lhs;
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
        __result.overflow = _add_overflow_i8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = _add_overflow_i16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = _add_overflow_i32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = _add_overflow_i64(0, __lhs, __rhs, &__result.value);
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
        __result.overflow = _addcarry_u8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = _addcarry_u16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = _addcarry_u32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = _addcarry_u64(0, __lhs, __rhs, &__result.value);
      }
      else
      {
        __result = __impl_generic(__lhs, __rhs);
      }
    }
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC, >=, 19, 37) ||
        // !_CCCL_ARCH(X86_64) vvv
    __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^

    return __result;
  }
#endif // !_CCCL_COMPILER(NVRTC)

  template <class _Tp>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp> __impl(_Tp __lhs, _Tp __rhs) noexcept
  {
#if defined(_CCCL_BUILTIN_ADD_OVERFLOW)
    overflow_result<_Tp> __result{};
    __result.overflow = _CCCL_BUILTIN_ADD_OVERFLOW(__lhs, __rhs, &__result.value);
    return __result;
#else // ^^^ _CCCL_BUILTIN_ADD_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_ADD_OVERFLOW vvv
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST, (return __impl_host(__lhs, __rhs);))
    }
    return __impl_generic(__lhs, __rhs);
#endif // !_CCCL_BUILTIN_ADD_OVERFLOW
  }
};

//! @brief Adds two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
_CCCL_TEMPLATE(
  class _Result = void,
  class _Lhs,
  class _Rhs,
  class _ActResult =
    _CUDA_VSTD::conditional_t<_CCCL_TRAIT(_CUDA_VSTD::is_void, _Result), _CUDA_VSTD::common_type_t<_Lhs, _Rhs>, _Result>)
_CCCL_REQUIRES((_CCCL_TRAIT(_CUDA_VSTD::is_void, _Result) || _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Result))
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                   _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr overflow_result<_ActResult>
add_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  using _Tp = _CUDA_VSTD::__make_nbit_uint_t<_CUDA_VSTD::max({sizeof(_ActResult), sizeof(_Lhs), sizeof(_Rhs)})>;

  const auto __lhs_sign = _CUDA_VSTD::cmp_less(__lhs, _Lhs{0});
  const auto __rhs_sign = _CUDA_VSTD::cmp_less(__rhs, _Rhs{0});
  const auto __lhs_val  = static_cast<_Tp>(_CUDA_VSTD::uabs(__lhs));
  const auto __rhs_val  = static_cast<_Tp>(_CUDA_VSTD::uabs(__rhs));

  ::cuda::overflow_result<_ActResult> __result{};

  if (__lhs_sign == __rhs_sign)
  {
    const auto __add_res  = __lhs_val + __rhs_val;
    const auto __cast_res = ::cuda::overflow_cast<_ActResult>(__res);

    __result.value    = __cast_res.value;
    __result.overflow = __cast_res.overflow || (__add_res < __lhs_val);

    if (__lhs_sign)
    {
      __result.value = -__result.value;
    }
  }
  else if (__lhs_sign)
  {
    const auto __sub_res  = __rhs_val - __lhs_val;
    const auto __cast_res = ::cuda::overflow_cast<_ActResult>(__res);

    __result.value    = __cast_res.value;
    __result.overflow = __cast_res.overflow || (__sub_res > __rhs_val);
  }
  else
  {
  }

  return __result;
}

//! @brief Adds two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
//! @param __result The result of the addition
_CCCL_TEMPLATE(class _Result, class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Result)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                   _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
add_overflow(_Result& __result, const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  const auto __res = ::cuda::add_overflow<_Result>(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___NUMERIC_ADD_OVERFLOW_H
