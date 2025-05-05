//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_MONTH_WEEKDAY_H
#define _LIBCUDACXX___CHRONO_MONTH_WEEKDAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/month.h>
#include <cuda/std/__chrono/weekday.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

class month_weekday
{
private:
  chrono::month __m_;
  chrono::weekday_indexed __wdi_;

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday(const chrono::month& __mval,
                                                    const chrono::weekday_indexed& __wdival) noexcept
      : __m_{__mval}
      , __wdi_{__wdival}
  {}
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::month month() const noexcept
  {
    return __m_;
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::weekday_indexed weekday_indexed() const noexcept
  {
    return __wdi_;
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool ok() const noexcept
  {
    return __m_.ok() && __wdi_.ok();
  }
};

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator==(const month_weekday& __lhs, const month_weekday& __rhs) noexcept
{
  return __lhs.month() == __rhs.month() && __lhs.weekday_indexed() == __rhs.weekday_indexed();
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday
operator/(const month& __lhs, const weekday_indexed& __rhs) noexcept
{
  return month_weekday{__lhs, __rhs};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday
operator/(int __lhs, const weekday_indexed& __rhs) noexcept
{
  return month_weekday{month(__lhs), __rhs};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday
operator/(const weekday_indexed& __lhs, const month& __rhs) noexcept
{
  return month_weekday{__rhs, __lhs};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday
operator/(const weekday_indexed& __lhs, int __rhs) noexcept
{
  return month_weekday{month(__rhs), __lhs};
}

class month_weekday_last
{
  chrono::month __m_;
  chrono::weekday_last __wdl_;

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday_last(
    const chrono::month& __mval, const chrono::weekday_last& __wdlval) noexcept
      : __m_{__mval}
      , __wdl_{__wdlval}
  {}
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::month month() const noexcept
  {
    return __m_;
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::weekday_last weekday_last() const noexcept
  {
    return __wdl_;
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool ok() const noexcept
  {
    return __m_.ok() && __wdl_.ok();
  }
};

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator==(const month_weekday_last& __lhs, const month_weekday_last& __rhs) noexcept
{
  return __lhs.month() == __rhs.month() && __lhs.weekday_last() == __rhs.weekday_last();
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday_last
operator/(const month& __lhs, const weekday_last& __rhs) noexcept
{
  return month_weekday_last{__lhs, __rhs};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday_last
operator/(int __lhs, const weekday_last& __rhs) noexcept
{
  return month_weekday_last{month(__lhs), __rhs};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday_last
operator/(const weekday_last& __lhs, const month& __rhs) noexcept
{
  return month_weekday_last{__rhs, __lhs};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr month_weekday_last
operator/(const weekday_last& __lhs, int __rhs) noexcept
{
  return month_weekday_last{month(__rhs), __lhs};
}
} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_MONTH_WEEKDAY_H
