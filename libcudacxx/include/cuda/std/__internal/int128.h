// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_INT128_H
#define _CUDA_STD___INTERNAL_INT128_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

class _CCCL_ALIGNAS(16) __cccl_uint128
{
  using __word_type = unsigned long long;

  __word_type __lo_;
  __word_type __hi_;

public:
  _CCCL_HIDE_FROM_ABI constexpr __cccl_uint128() noexcept = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_integral_v<_Tp> _CCCL_AND(!is_same_v<_Tp, __cccl_uint128>))
  _CCCL_API constexpr __cccl_uint128(_Tp __v) noexcept
      : __lo_{static_cast<__word_type>(__v)}
      , __hi_{}
  {}

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_floating_point_v<_Tp>)
  _CCCL_API constexpr __cccl_uint128(_Tp __v) noexcept
      : __cccl_uint128{}
  {
    // todo
  }

  _CCCL_HIDE_FROM_ABI constexpr __cccl_uint128(const __cccl_uint128&) noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr __cccl_uint128& operator=(const __cccl_uint128&) noexcept = default;

  // conversion operators

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_integral_v<_Tp> _CCCL_AND(!is_same_v<_Tp, __cccl_uint128>))
  _CCCL_API constexpr operator _Tp() noexcept
  {
    return static_cast<_Tp>(__lo_);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_floating_point_v<_Tp>)
  _CCCL_API constexpr operator _Tp() noexcept
  {
    // todo
    return {};
  }

  // bit assignment operators

  _CCCL_API constexpr __cccl_uint128& operator&=(const __cccl_uint128& __other) noexcept
  {
    __lo_ &= __other.__lo_;
    __hi_ &= __other.__hi_;
    return *this;
  }
  _CCCL_API constexpr __cccl_uint128& operator|=(const __cccl_uint128& __other) noexcept
  {
    __lo_ |= __other.__lo_;
    __hi_ |= __other.__hi_;
    return *this;
  }
  _CCCL_API constexpr __cccl_uint128& operator^=(const __cccl_uint128& __other) noexcept
  {
    __lo_ ^= __other.__lo_;
    __hi_ ^= __other.__hi_;
    return *this;
  }
  _CCCL_API constexpr __cccl_uint128& operator<<=(int __n) noexcept
  {
    if (__n == 0)
    {
    }
    else if (__n < 64)
    {
      __hi_ = (__hi_ << __n) | ((__lo_ & ((~0ull) << (64 - __n))) >> (64 - __n));
      __lo_ <<= __n;
    }
    else if (__n < 128)
    {
      __hi_ = __lo_ << (__n - 64);
      __lo_ = 0;
    }
    else
    {
      __lo_ = 0;
      __hi_ = 0;
    }
    return *this;
  }
  _CCCL_API constexpr __cccl_uint128& operator>>=(int __n) noexcept
  {
    if (__n == 0)
    {
    }
    else if (__n < 64)
    {
      __lo_ = (__lo_ >> __n) | ((__hi_ & ((~0ull) >> __n)) << __n);
      __hi_ >>= __n;
    }
    else if (__n < 128)
    {
      __lo_ = __hi_ >> (__n - 64);
      __hi_ = 0;
    }
    else
    {
      __hi_ = 0;
      __lo_ = 0;
    }
    return *this;
  }

  // arithmetic assignment operators

  _CCCL_API constexpr __cccl_uint128& operator+=(const __cccl_uint128& __other) noexcept
  {
    const auto __old_lo = __lo_;
    __lo_ += __other.__lo_;
    __hi_ += __other.__hi_ + (__lo_ < __old_lo);
    return *this;
  }
  _CCCL_API constexpr __cccl_uint128& operator-=(const __cccl_uint128& __other) noexcept
  {
    const auto __old_lo = __lo_;
    __lo_ -= __other.__lo_;
    __hi_ -= __other.__hi_ - (__lo_ > __old_lo);
    return *this;
  }
  _CCCL_API constexpr __cccl_uint128& operator*=(const __cccl_uint128& __other) noexcept
  {
    // todo
  }
  _CCCL_API constexpr __cccl_uint128& operator/=(const __cccl_uint128& __other) noexcept
  {
    // todo
  }
  _CCCL_API constexpr __cccl_uint128& operator%=(const __cccl_uint128& __other) noexcept
  {
    // todo
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___INTERNAL_INT128_H
