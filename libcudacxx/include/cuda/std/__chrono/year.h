//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_YEAR_H
#define _LIBCUDACXX___CHRONO_YEAR_H

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
#include <cuda/std/limits>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

class year
{
private:
  short __y_;

public:
  _CCCL_HIDE_FROM_ABI constexpr year() noexcept = default;
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr year(int __val) noexcept
      : __y_(static_cast<short>(__val))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr year& operator++() noexcept
  {
    ++__y_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year operator++(int) noexcept
  {
    year __tmp = *this;
    ++(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year& operator--() noexcept
  {
    --__y_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year operator--(int) noexcept
  {
    year __tmp = *this;
    --(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year& operator+=(const years& __dy) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year& operator-=(const years& __dy) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year operator+() const noexcept
  {
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr year operator-() const noexcept
  {
    return year{-__y_};
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_leap() const noexcept
  {
    return __y_ % 4 == 0 && (__y_ % 100 != 0 || __y_ % 400 == 0);
  }
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator int() const noexcept
  {
    return __y_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool ok() const noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr year min() noexcept
  {
    return year{-32767};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr year max() noexcept
  {
    return year{32767};
  }
};

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator==(const year& __lhs, const year& __rhs) noexcept
{
  return static_cast<int>(__lhs) == static_cast<int>(__rhs);
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
_LIBCUDACXX_HIDE_FROM_ABI constexpr strong_ordering operator<=>(const year& __lhs, const year& __rhs) noexcept
{
  return static_cast<int>(__lhs) <=> static_cast<int>(__rhs);
}
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_LIBCUDACXX_HIDE_FROM_ABI constexpr year operator+(const year& __lhs, const years& __rhs) noexcept
{
  return year(static_cast<int>(__lhs) + __rhs.count());
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr year operator+(const years& __lhs, const year& __rhs) noexcept
{
  return __rhs + __lhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr year operator-(const year& __lhs, const years& __rhs) noexcept
{
  return __lhs + -__rhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr years operator-(const year& __lhs, const year& __rhs) noexcept
{
  return years{static_cast<int>(__lhs) - static_cast<int>(__rhs)};
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr year& year::operator+=(const years& __dy) noexcept
{
  *this = *this + __dy;
  return *this;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr year& year::operator-=(const years& __dy) noexcept
{
  *this = *this - __dy;
  return *this;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool year::ok() const noexcept
{
  static_assert(static_cast<int>(std::numeric_limits<decltype(__y_)>::max()) == static_cast<int>(max()));
  return static_cast<int>(min()) <= __y_;
}

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___CHRONO_YEAR_H
