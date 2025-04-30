//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_DAY_H
#define _LIBCUDACXX___CHRONO_DAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
// #include <cuda/std/__compare/ordering.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

class day
{
private:
  unsigned char __d_;

public:
  _CCCL_HIDE_FROM_ABI constexpr day() noexcept = default;
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr day(unsigned __val) noexcept
      : __d_(static_cast<unsigned char>(__val))
  {}
  _LIBCUDACXX_HIDE_FROM_ABI constexpr day& operator++() noexcept
  {
    ++__d_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr day operator++(int) noexcept
  {
    day __tmp = *this;
    ++(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr day& operator--() noexcept
  {
    --__d_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr day operator--(int) noexcept
  {
    day __tmp = *this;
    --(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator unsigned() const noexcept
  {
    return __d_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool ok() const noexcept
  {
    return __d_ >= 1 && __d_ <= 31;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator==(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) == static_cast<unsigned>(__rhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator!=(const day& __lhs, const day& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator<(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) < static_cast<unsigned>(__rhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator>(const day& __lhs, const day& __rhs) noexcept
  {
    return __rhs < __lhs;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator<=(const day& __lhs, const day& __rhs) noexcept
  {
    return !(__rhs < __lhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator>=(const day& __lhs, const day& __rhs) noexcept
  {
    return !(__lhs < __rhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr day operator+(const day& __lhs, const days& __rhs) noexcept
  {
    return day(static_cast<unsigned>(__lhs) + __rhs.count());
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr day operator+(const days& __lhs, const day& __rhs) noexcept
  {
    return __rhs + __lhs;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr day operator-(const day& __lhs, const days& __rhs) noexcept
  {
    return __lhs + -__rhs;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr days operator-(const day& __lhs, const day& __rhs) noexcept
  {
    return days(static_cast<int>(static_cast<unsigned>(__lhs)) - static_cast<int>(static_cast<unsigned>(__rhs)));
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr day& operator+=(const days& __dd) noexcept
  {
    *this = *this + __dd;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr day& operator-=(const days& __dd) noexcept
  {
    *this = *this - __dd;
    return *this;
  }
};

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_DAY_H
