//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_MONTH_H
#define _LIBCUDACXX___CHRONO_MONTH_H

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

class month
{
private:
  unsigned char __m_;

public:
  _CCCL_HIDE_FROM_ABI constexpr month() noexcept = default;
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr month(unsigned __val) noexcept
      : __m_(static_cast<unsigned char>(__val))
  {}
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month& operator++() noexcept
  {
    *this += months{1};
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month operator++(int) noexcept
  {
    month __tmp = *this;
    ++(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month& operator--() noexcept
  {
    *this -= months{1};
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month operator--(int) noexcept
  {
    month __tmp = *this;
    --(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month& operator+=(const months& __m1) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month& operator-=(const months& __m1) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator unsigned() const noexcept
  {
    return __m_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool ok() const noexcept
  {
    return __m_ >= 1 && __m_ <= 12;
  }
};

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator==(const month& __lhs, const month& __rhs) noexcept
{
  return static_cast<unsigned>(__lhs) == static_cast<unsigned>(__rhs);
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
_LIBCUDACXX_HIDE_FROM_ABI constexpr strong_ordering operator<=>(const month& __lhs, const month& __rhs) noexcept
{
  return static_cast<unsigned>(__lhs) <=> static_cast<unsigned>(__rhs);
}
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_LIBCUDACXX_HIDE_FROM_ABI constexpr month operator+(const month& __lhs, const months& __rhs) noexcept
{
  auto const __mu = static_cast<long long>(static_cast<unsigned>(__lhs)) + (__rhs.count() - 1);
  auto const __yr = (__mu >= 0 ? __mu : __mu - 11) / 12;
  return month{static_cast<unsigned>(__mu - __yr * 12 + 1)};
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr month operator+(const months& __lhs, const month& __rhs) noexcept
{
  return __rhs + __lhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr month operator-(const month& __lhs, const months& __rhs) noexcept
{
  return __lhs + -__rhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr months operator-(const month& __lhs, const month& __rhs) noexcept
{
  auto const __dm = static_cast<unsigned>(__lhs) - static_cast<unsigned>(__rhs);
  return months(__dm <= 11 ? __dm : __dm + 12);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr month& month::operator+=(const months& __dm) noexcept
{
  *this = *this + __dm;
  return *this;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr month& month::operator-=(const months& __dm) noexcept
{
  *this = *this - __dm;
  return *this;
}

_CCCL_GLOBAL_CONSTANT month January{1};
_CCCL_GLOBAL_CONSTANT month February{2};
_CCCL_GLOBAL_CONSTANT month March{3};
_CCCL_GLOBAL_CONSTANT month April{4};
_CCCL_GLOBAL_CONSTANT month May{5};
_CCCL_GLOBAL_CONSTANT month June{6};
_CCCL_GLOBAL_CONSTANT month July{7};
_CCCL_GLOBAL_CONSTANT month August{8};
_CCCL_GLOBAL_CONSTANT month September{9};
_CCCL_GLOBAL_CONSTANT month October{10};
_CCCL_GLOBAL_CONSTANT month November{11};
_CCCL_GLOBAL_CONSTANT month December{12};

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_MONTH_H
