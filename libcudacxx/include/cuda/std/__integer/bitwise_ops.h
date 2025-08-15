//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTEGER_BITWISE_OPS_H
#define _LIBCUDACXX___INTEGER_BITWISE_OPS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/integer.h>
#include <cuda/std/__integer/properties.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __int_bitwise_not(_Tp __v) noexcept
{
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    _Tp __ret{};
    for (size_t __i = 0; __i < _Tp::__nwords; ++__i)
    {
      __ret.__storage[__i] = ~__v.__storage[__i];
    }
    return __ret;
  }
  else
  {
    return ~__v;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __int_bitwise_and(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    _Tp __ret{};
    for (size_t __i = 0; __i < _Tp::__nwords; ++__i)
    {
      __ret.__storage[__i] = __lhs.__storage[__i] & __rhs.__storage[__i];
    }
    return __ret;
  }
  else
  {
    return __lhs & __rhs;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __int_bitwise_or(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    _Tp __ret{};
    for (size_t __i = 0; __i < _Tp::__nwords; ++__i)
    {
      __ret.__storage[__i] = __lhs.__storage[__i] | __rhs.__storage[__i];
    }
    return __ret;
  }
  else
  {
    return __lhs | __rhs;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __int_bitwise_xor(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    _Tp __ret{};
    for (size_t __i = 0; __i < _Tp::__nwords; ++__i)
    {
      __ret.__storage[__i] = __lhs.__storage[__i] ^ __rhs.__storage[__i];
    }
    return __ret;
  }
  else
  {
    return __lhs ^ __rhs;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __int_bitwise_shift_left(_Tp __v, size_t __shift) noexcept
{
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    _Tp __ret{};
    const auto __word_stride  = __shift / _Tp::__word_nbits;
    const auto __inword_shift = __shift % _Tp::__word_nbits;
    for (ptrdiff_t __i = _Tp::__nwords - 1; __i >= 0; --__i)
    {
      if (__i >= __word_stride)
      {
        __ret.__storage[__i] =
          __v.__storage[__i - __word_stride] << __inword_shift
          | (__i - __word_stride > 0 ? __v.__storage[__i - __word_stride - 1] : 0)
              >> (_Tp::__word_nbits - __inword_shift);
      }
      else
      {
        __ret.__storage[__i] = 0;
      }
    }
    return __ret;
  }
  else
  {
    return __v << __shift;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __int_bitwise_shift_right(_Tp __v, size_t __shift) noexcept
{
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    _Tp __ret{};
    const auto __word_stride  = __shift / _Tp::__word_nbits;
    const auto __inword_shift = __shift % _Tp::__word_nbits;
    const auto __fill         = static_cast<typename _Tp::__word_type>(_Tp::__is_signed && __int_get_msb(__v));
    for (size_t __i = 0; __i < _Tp::__nwords; ++__i)
    {
      if (__i + __word_stride < _Tp::__nwords)
      {
        __ret.__storage[__i] =
          __v.__storage[__i + __word_stride] >> __inword_shift
          | (__i + __word_stride + 1 < _Tp::__nwords ? __v.__storage[__i + __word_stride + 1] : 0)
              << (_Tp::__word_nbits - __inword_shift);
      }
      else
      {
        __ret.__storage[__i] = __fill;
      }
    }
    return __ret;
  }
  else
  {
    return __v >> __shift;
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___INTEGER_BITWISE_OPS_H
