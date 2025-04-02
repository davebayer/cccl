//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHARCONV_TO_CHARS_H
#define _LIBCUDACXX___CHARCONV_TO_CHARS_H

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
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__charconv/to_chars_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char __to_chars_value_to_char(int __v, int __base) noexcept
{
  _CCCL_ASSERT(__v >= 0 && __v < __base, "value must be in the range [0, base)"); // todo: remove after testing
  const int __offset = (__base < 10 || __v < 10) ? '0' : ('a' - 10);
  return static_cast<char>(__offset + __v);
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t __to_chars_int_width(_Tp __v, int __base) noexcept
{
  int __ndigits{};

  if (_CUDA_VSTD::has_single_bit(__v))
  {
    __ndigits =
      ::cuda::ceil_div((numeric_limits<_Tp>::digits - _CUDA_VSTD::countl_zero(__v | 1)), ::cuda::ilog2(__base));
  }
  else
  {
    // todo
  }

  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    if (__v < _Tp{0})
    {
      ++__ndigits;

      // special case for -min for signed base 2
      if (__base == 2 && __v == numeric_limits<_Tp>::min())
      {
        __ndigits += 1;
      }
    }
  }

  return __ndigits;
}

template <size_t __base, class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void __to_chars_int_pow2_base(char* __last, _Tp __value) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(__base), "base must be a power of 2");

  constexpr int __base_log2 = ::cuda::ilog2(__base);

  do
  {
    const auto __digit = static_cast<int>(__value) & (__base - 1);
    *--__last          = _CUDA_VSTD::__to_chars_value_to_char(__digit, __base);
    __value >>= __base_log2;
  } while (__value != 0);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void __to_chars_int_generic(char* __last, _Tp __value, int __base) noexcept
{
  do
  {
    const int __c = __value % __base;
    *--__last     = "0123456789abcdefghijklmnopqrstuvwxyz"[__c];
    __value /= __base;
  } while (__value != 0);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__cccl_is_integer, _Tp) || _CCCL_TRAIT(is_same, _Tp, char))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr to_chars_result
to_chars(char* __first, char* __last, _Tp __value, int __base = 10) noexcept
{
  _CCCL_ASSERT(__base >= 2 && __base < 36, "base must be in the range [2, 35]");

  const ptrdiff_t __cap = __last - __first;
  const int __n         = _CUDA_VSTD::__to_chars_int_width(__value, __base);

  if (__n > __cap)
  {
    return {__first, errc::value_too_large};
  }

  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    if (__value < _Tp{0})
    {
      *__first++ = '-';
    }
  }

  const auto __abs_value = make_unsigned_t<_Tp>((__value < 0) ? -__value : __value); // todo: replace with cuda::uabs
  char* __new_last       = __first + __n;

  switch (__base)
  {
    case 2:
      _CUDA_VSTD::__to_chars_int_pow2_base<2>(__new_last, __abs_value);
      break;
    case 4:
      _CUDA_VSTD::__to_chars_int_pow2_base<4>(__new_last, __abs_value);
      break;
    case 8:
      _CUDA_VSTD::__to_chars_int_pow2_base<8>(__new_last, __abs_value);
      break;
    case 10:
      _CUDA_VSTD::__to_chars_int_generic(__new_last, __abs_value, 10); // todo: optimize
      break;
    case 16:
      _CUDA_VSTD::__to_chars_int_pow2_base<16>(__new_last, __abs_value);
      break;
    case 32:
      _CUDA_VSTD::__to_chars_int_pow2_base<32>(__new_last, __abs_value);
      break;
    default:
      _CUDA_VSTD::__to_chars_int_generic(__new_last, __abs_value, __base);
      break;
  }
  return {__new_last, errc{}};
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr to_chars_result to_chars(char*, char*, bool, int = 10) noexcept = delete;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHARCONV_TO_CHARS_H
