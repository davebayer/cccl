// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_SATURATION_ARITHMETIC_H
#define _LIBCUDACXX___NUMERIC_SATURATION_ARITHMETIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/int_arithmetic.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/climits>
#include <cuda/std/limits>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

class __add_sat
{
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp
  __fix_overflow(_Tp __x, _Tp, _Tp __result, bool __overflow) noexcept
  {
    if (__overflow)
    {
      _CCCL_IF_CONSTEXPR (_CCCL_TRAIT(is_unsigned, _Tp))
      {
        __result = _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
      else
      {
        __result = (__x > 0) ? _CUDA_VSTD::numeric_limits<_Tp>::max() : _CUDA_VSTD::numeric_limits<_Tp>::min();
      }
    }

    return __result;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    using _Up = make_unsigned_t<_Tp>;

    _Tp __result = static_cast<_Tp>(static_cast<_Up>(__x) + static_cast<_Up>(__y));

    return __fix_overflow(__x, __y, __result, (__result < __x) == !(__y < static_cast<_Tp>(0)));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result = __x + __y;

    return __fix_overflow(__x, __y, __result, (__result < __x));
  }

public:
#if defined(_CCCL_BUILTIN_ADD_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_ADD_OVERFLOW(__x, __y, &__result);

    return __fix_overflow(__x, __y, __result, __overflow);
  }
#endif // _CCCL_BUILTIN_ADD_OVERFLOW

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static _Tp __impl_host(_Tp __x, _Tp __y) noexcept
  {
    return __impl_generic(__x, __y);
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int8_t __impl_host(int8_t __x, int8_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_i8(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int8_t __result;
    bool __overflow = _add_overflow_i8(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int16_t __impl_host(int16_t __x, int16_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_i16(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int16_t __result;
    bool __overflow = _add_overflow_i16(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int32_t __impl_host(int32_t __x, int32_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_i32(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int32_t __result;
    bool __overflow = _add_overflow_i32(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int64_t __impl_host(int64_t __x, int64_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_i64(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _M_X64
    int64_t __result;
    bool __overflow = _add_overflow_i64(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint8_t __impl_host(uint8_t __x, uint8_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_u8(__x, __y);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    uint8_t __result;
    bool __overflow = _addcarry_u8(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint16_t __impl_host(uint16_t __x, uint16_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_u16(__x, __y);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    uint16_t __result;
    bool __overflow = _addcarry_u16(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint32_t __impl_host(uint32_t __x, uint32_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_u32(__x, __y);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    uint32_t __result;
    bool __overflow = _addcarry_u32(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint64_t __impl_host(uint64_t __x, uint64_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_add_u64(__x, __y);
#elif _CCCL_COMPILER(MSVC) && _M_X64
    uint64_t __result;
    bool __overflow = _addcarry_u64(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

#if _CCCL_HAS_CUDA_COMPILER
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    return __impl_generic(__x, __y);
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static int32_t __impl_device(int32_t __x, int32_t __y) noexcept
  {
    int32_t __result{};
    asm("add.sat.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
    return __result;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    const _Tp __bneg_x = ~__x;
    return __x + _CUDA_VSTD::min<_Tp>(__y, __bneg_x);
  }
#endif // _CCCL_HAS_CUDA_COMPILER

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_constexpr(_Tp __x, _Tp __y) noexcept
  {
    if (__y > 0 && __x > _CUDA_VSTD::numeric_limits<_Tp>::max() - __y)
    {
      return _CUDA_VSTD::numeric_limits<_Tp>::max();
    }
    else if (__y < 0 && __x < _CUDA_VSTD::numeric_limits<_Tp>::min() - __y)
    {
      return _CUDA_VSTD::numeric_limits<_Tp>::min();
    }
    return __x + __y;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_constexpr(_Tp __x, _Tp __y) noexcept
  {
    return (__x > _CUDA_VSTD::numeric_limits<_Tp>::max() - __y) ? _CUDA_VSTD::numeric_limits<_Tp>::max() : __x + __y;
  }
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp add_sat(_Tp __x, _Tp __y) noexcept
{
#if defined(_CCCL_BUILTIN_ADD_OVERFLOW)
  return __add_sat::__impl_builtin(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ADD_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_ADD_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __add_sat::__impl_host(__x, __y);), (return __add_sat::__impl_device(__x, __y);))
  }
  return __add_sat::__impl_constexpr(__x, __y);
#endif // !_CCCL_BUILTIN_ADD_OVERFLOW
}

class __sub_sat
{
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp
  __fix_overflow(_Tp __x, _Tp, _Tp __result, bool __overflow) noexcept
  {
    if (__overflow)
    {
      _CCCL_IF_CONSTEXPR (_CCCL_TRAIT(is_unsigned, _Tp))
      {
        __result = _CUDA_VSTD::numeric_limits<_Tp>::min();
      }
      else
      {
        __result = (_CCCL_TRAIT(is_signed, _Tp) && __x >= 0)
                   ? _CUDA_VSTD::numeric_limits<_Tp>::max()
                   : _CUDA_VSTD::numeric_limits<_Tp>::min();
      }
    }

    return __result;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    using _Up = make_unsigned_t<_Tp>;

    _Tp __result = static_cast<_Tp>(static_cast<_Up>(__x) - static_cast<_Up>(__y));

    return __fix_overflow(__x, __y, __result, (__result < __x) == !(__y > static_cast<_Tp>(0)));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result = __x - __y;

    return __fix_overflow(__x, __y, __result, (__result > __x));
  }

public:
#if defined(_CCCL_BUILTIN_SUB_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_SUB_OVERFLOW(__x, __y, &__result);

    return __fix_overflow(__x, __y, __result, __overflow);
  }
#endif // _CCCL_BUILTIN_SUB_OVERFLOW

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static _Tp __impl_host(_Tp __x, _Tp __y) noexcept
  {
    return __impl_generic(__x, __y);
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int8_t __impl_host(int8_t __x, int8_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_i8(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int8_t __result;
    bool __overflow = _sub_overflow_i8(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int16_t __impl_host(int16_t __x, int16_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_i16(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int16_t __result;
    bool __overflow = _sub_overflow_i16(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int32_t __impl_host(int32_t __x, int32_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_i32(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int32_t __result;
    bool __overflow = _sub_overflow_i32(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int64_t __impl_host(int64_t __x, int64_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_i64(__x, __y);
#elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _M_X64
    int64_t __result;
    bool __overflow = _sub_overflow_i64(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint8_t __impl_host(uint8_t __x, uint8_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_u8(__x, __y);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    uint8_t __result;
    bool __overflow = _subborrow_u8(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint16_t __impl_host(uint16_t __x, uint16_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_u16(__x, __y);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    uint16_t __result;
    bool __overflow = _subborrow_u16(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint32_t __impl_host(uint32_t __x, uint32_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_u32(__x, __y);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    uint32_t __result;
    bool __overflow = _subborrow_u32(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint64_t __impl_host(uint64_t __x, uint64_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 41) && _M_X64
    return _sat_sub_u64(__x, __y);
#elif _CCCL_COMPILER(MSVC) && _M_X64
    uint64_t __result;
    bool __overflow = _subborrow_u64(0, __x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

#if _CCCL_HAS_CUDA_COMPILER
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    return __impl_generic(__x, __y);
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static int32_t __impl_device(int32_t __x, int32_t __y) noexcept
  {
    int32_t __result{};
    asm("sub.sat.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
    return __result;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    return _CUDA_VSTD::max<_Tp>(__x, __y) - __y;
  }
#endif // _CCCL_HAS_CUDA_COMPILER

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_constexpr(_Tp __x, _Tp __y) noexcept
  {
    if (__y < 0 && __x > _CUDA_VSTD::numeric_limits<_Tp>::max() + __y)
    {
      return _CUDA_VSTD::numeric_limits<_Tp>::max();
    }
    else if (__y > 0 && __x < _CUDA_VSTD::numeric_limits<_Tp>::min() + __y)
    {
      return _CUDA_VSTD::numeric_limits<_Tp>::min();
    }
    return __x - __y;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_constexpr(_Tp __x, _Tp __y) noexcept
  {
    return (__y > __x) ? _CUDA_VSTD::numeric_limits<_Tp>::min() : __x - __y;
  }
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp sub_sat(_Tp __x, _Tp __y) noexcept
{
#if defined(_CCCL_BUILTIN_SUB_OVERFLOW)
  return __sub_sat::__impl_builtin(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SUB_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_SUB_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __sub_sat::__impl_host(__x, __y);), (return __sub_sat::__impl_device(__x, __y);))
  }
  return __sub_sat::__impl_constexpr(__x, __y);
#endif // !_CCCL_BUILTIN_SUB_OVERFLOW
}

class __mul_sat
{
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp
  __fix_overflow(_Tp __x, _Tp __y, _Tp __result, bool __overflow) noexcept
  {
    if (__overflow)
    {
      _CCCL_IF_CONSTEXPR (_CCCL_TRAIT(is_unsigned, _Tp))
      {
        __result = _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
      else
      {
        __result = (__x > 0 && __y > 0) || (__x < 0 && __y < 0)
                   ? _CUDA_VSTD::numeric_limits<_Tp>::max()
                   : _CUDA_VSTD::numeric_limits<_Tp>::min();
      }
    }

    return __result;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    // TODO
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    // TODO
  }

public:
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__x, __y, &__result);

    return __fix_overflow(__x, __y, __result, __overflow);
  }
#endif // _CCCL_BUILTIN_MUL_OVERFLOW

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static _Tp __impl_host(_Tp __x, _Tp __y) noexcept
  {
    return __impl_generic(__x, __y);
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int8_t __impl_host(int8_t __x, int8_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int16_t __result;
    bool __overflow = _mul_full_overflow_i8(__x, __y, &__result);
    return __fix_overflow(__x, __y, static_cast<int8_t>(__result), __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int16_t __impl_host(int16_t __x, int16_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int16_t __result;
    bool __overflow = _mul_overflow_i16(__x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int32_t __impl_host(int32_t __x, int32_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    int32_t __result;
    bool __overflow = _mul_overflow_i32(__x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    const int64_t __result = __emul(__x, __y);
    return __fix_overflow(
      __x,
      __y,
      static_cast<int32_t>(__result),
      __result > _CUDA_VSTD::numeric_limits<int32_t>::max() || __result < _CUDA_VSTD::numeric_limits<int32_t>::min());
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static int64_t __impl_host(int64_t __x, int64_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && _M_X64
    int64_t __result;
    bool __overflow = _mul_overflow_i64(__x, __y, &__result);
    return __fix_overflow(__x, __y, __result, __overflow);
#elif _CCCL_COMPILER(MSVC) && _M_X64
    int64_t __hi = __mulh(__x, __y);
    int64_t __lo = __x * __y;
    return __fix_overflow(__x, __y, __lo, hi != 0 && hi != -1);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint8_t __impl_host(uint8_t __x, uint8_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    uint16_t __result;
    bool __overflow = _mul_full_overflow_u8(__x, __y, &__result);
    return __fix_overflow(__x, __y, static_cast<uint8_t>(__result), __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint16_t __impl_host(uint16_t __x, uint16_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    uint16_t __lo, __hi;
    bool __overflow = _mul_full_overflow_u16(__x, __y, &__lo, &__hi);
    return __fix_overflow(__x, __y, __lo, __overflow);
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint32_t __impl_host(uint32_t __x, uint32_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && (_M_IX86 || _M_X64)
    uint32_t __lo, __hi;
    bool __overflow = _mul_full_overflow_u32(__x, __y, &__lo, &__hi);
    return __fix_overflow(__x, __y, __lo, __overflow);
#elif _CCCL_COMPILER(MSVC) && (_M_IX86 || _M_X64)
    const uint64_t __result = __emulu(__x, __y);
    return __fix_overflow(
      __x, __y, static_cast<uint32_t>(__result), __result > _CUDA_VSTD::numeric_limits<uint32_t>::max());
#else
    return __impl_generic(__x, __y);
#endif
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static uint64_t __impl_host(uint64_t __x, uint64_t __y) noexcept
  {
#if _CCCL_COMPILER(MSVC, >=, 19, 37) && _M_X64
    uint64_t __lo, __hi;
    bool __overflow = _mul_full_overflow_u64(__x, __y, &__lo, &__hi);
    return __fix_overflow(__x, __y, __lo, __overflow);
#elif _CCCL_COMPILER(MSVC) && _M_X64
    return __fix_overflow(__x, __y, __x * __y, __umulh(__x, __y) != 0);
#else
    return __impl_generic(__x, __y);
#endif
  }

#if _CCCL_HAS_CUDA_COMPILER
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    return __impl_generic(__x, __y);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp) _CCCL_AND(sizeof(_Tp) <= 8))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    // TODO
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    // TODO
  }
#endif // _CCCL_HAS_CUDA_COMPILER

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_signed, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_constexpr(_Tp __x, _Tp __y) noexcept
  {
    // TODO
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_unsigned, _Tp))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl_constexpr(_Tp __x, _Tp __y) noexcept
  {
    // TODO
  }
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp mul_sat(_Tp __x, _Tp __y) noexcept
{
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  return __mul_sat::__impl_builtin(__x, __y);
#else // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_MUL_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __mul_sat::__impl_host(__x, __y);), (return __mul_sat::__impl_device(__x, __y);))
  }
  return __mul_sat::__impl_constexpr(__x, __y);
#endif // !_CCCL_BUILTIN_MUL_OVERFLOW
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NUMERIC_SATURATION_ARITHMETIC_H
