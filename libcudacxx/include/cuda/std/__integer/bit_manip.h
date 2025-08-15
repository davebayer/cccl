//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTEGER_BIT_MANIP_H
#define _LIBCUDACXX___INTEGER_BIT_MANIP_H

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
[[nodiscard]] _CCCL_API constexpr bool __int_get_bit(_Tp __v, size_t __i) noexcept
{
  _CCCL_ASSERT(__i < __int_nbits_v<_Tp>, "index out of bounds");
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    const size_t __word_idx = __i / _Tp::__word_nbits;
    return (__v.__storage[__word_idx] >> (__i % _Tp::__word_nbits)) & 1;
  }
  else
  {
    return (__v >> __i) & 1;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr void __int_set_bit(_Tp& __v, size_t __i, bool __bit_val = true) noexcept
{
  _CCCL_ASSERT(__i < __int_nbits_v<_Tp>, "index out of bounds");
  if constexpr (__is_cccl_int_v<_Tp>)
  {
    const size_t __word_idx = __i / _Tp::__word_nbits;
    __v.__storage[__word_idx] |= (static_cast<typename _Tp::__word_type>(__bit_val) << (__i % _Tp::__word_nbits));
  }
  else
  {
    __v |= (static_cast<_Tp>(__bit_val) << __i);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool __int_get_msb(_Tp __v) noexcept
{
  return _CUDA_STD::__int_get_bit(__v, __int_nbits_v - 1);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool __int_set_msb(_Tp& __v, bool __bit_val = true) noexcept
{
  return _CUDA_STD::__int_set_bit(__v, __int_nbits_v - 1, __bit_val);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___INTEGER_BIT_MANIP_H
