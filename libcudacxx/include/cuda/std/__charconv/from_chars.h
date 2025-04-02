//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHARCONV_FROM_CHARS_H
#define _LIBCUDACXX___CHARCONV_FROM_CHARS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__charconv/from_chars_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int __from_chars_char_to_value(char __c, int __base) noexcept
{
  int __ret{};

  if (__c >= '0' && __c <= '9' && __c)
  {
    __ret = __c - '0';
  }
  else if (__c >= 'a' && __c <= 'z' && __c)
  {
    __ret = __c - 'a' + 10;
  }
  else if (__c >= 'A' && __c <= 'Z' && __c)
  {
    __ret = __c - 'A' + 10;
  }
  else
  {
    return -1;
  }

  return (__ret < __base) ? __ret : -1;
}

template <size_t __base, class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void __from_chars_int_pow2_base(const char* __last, _Tp __value) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(__base), "base must be a power of 2");

  constexpr int __base_log2 = ::cuda::ilog2(__base);

  do
  {
    const auto __digit = static_cast<int>(__value) & (__base - 1);
    *--__last          = _CUDA_VSTD::__from_chars_value_to_char(__digit, __base);
    __value >>= __base_log2;
  } while (__value != 0);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr from_chars_result
__from_chars_int_generic(const char* __it, ptrdiff_t __n, _Tp& __value, int __base) noexcept
{
  for (; __n-- > 0; ++__it)
  {
    const int __digit = _CUDA_VSTD::__from_chars_char_to_value(*__it, __base);

    if (__digit < 0)
    {
      return {__it, errc::invalid_argument};
    }
  }
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__cccl_is_integer, _Tp) || _CCCL_TRAIT(is_same, _Tp, char))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr from_chars_result
from_chars(const char* __first, const char* __last, _Tp& __value, int __base = 10) noexcept
{
  _CCCL_ASSERT(__base >= 2 && __base < 36, "base must be in the range [2, 35]");

  using _Up = make_unsigned_t<_Tp>;

  ptrdiff_t __n = __last - __first;

  if (__n <= 0)
  {
    return {__first, errc::invalid_argument};
  }

  [[maybe_unused]] bool __sign = false;
  const char* __it             = __first;

  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    if (*__it == '-')
    {
      __sign = true;
      ++__it;
      --__n;
    }
  }

  if (__n == 0)
  {
    return {__first, errc::invalid_argument};
  }

  from_chars_result __result{};
  _Up __value_unsigned{};

  for (; __n-- > 0; ++__it)
  {
    if (*__it != '0')
    {
      break;
    }
  }

  switch (__base)
  {
    case 2:
      __result = _CUDA_VSTD::__from_chars_int_pow2_base<2>(__it, __n, __value_unsigned);
      break;
    case 4:
      __result = _CUDA_VSTD::__from_chars_int_pow2_base<4>(__it, __n, __value_unsigned);
      break;
    case 8:
      __result = _CUDA_VSTD::__from_chars_int_pow2_base<8>(__it, __n, __value_unsigned);
      break;
    case 10:
      __result = _CUDA_VSTD::__from_chars_int_generic(__it, __n, __value_unsigned, 10); // todo: optimize
      break;
    case 16:
      __result = _CUDA_VSTD::__from_chars_int_pow2_base<16>(__it, __n, __value_unsigned);
      break;
    case 32:
      __result = _CUDA_VSTD::__from_chars_int_pow2_base<32>(__it, __n, __value_unsigned);
      break;
    default:
      __result = _CUDA_VSTD::__from_chars_int_generic(__it, __n, __value_unsigned, __base);
      break;
  }

  if (__result.ec != errc{})
  {
    return {__first, __result.ec};
  }

  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    if (__sign)
    {
      if (__value_unsigned <= _Up(numeric_limits<_Tp>::max()))
      {
        __value = -__value_unsigned;
      }
      else if (__value_unsigned == _Up(numeric_limits<_Tp>::max()) + 1)
      {
        __value = numeric_limits<_Tp>::min();
      }
      else
      {
        return {__first, errc::value_too_large};
      }
    }
    else
    {
      if (__value_unsigned > _Up(numeric_limits<_Tp>::max()))
      {
        return {__first, errc::value_too_large};
      }
      __value = static_cast<_Tp>(__value_unsigned);
    }
  }

  return {__it, errc{}};
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHARCONV_FROM_CHARS_H
