//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_TIME_POINT_H
#define _LIBCUDACXX___CHRONO_TIME_POINT_H

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
// #include <cuda/std/__compare/three_way_comparable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

template <class _Clock, class _Duration = typename _Clock::duration>
class time_point
{
  static_assert(__is_duration_v<_Duration>, "Second template parameter of time_point must be a std::chrono::duration");

public:
  using clock    = _Clock;
  using duration = _Duration;
  using rep      = typename duration::rep;
  using period   = typename duration::period;

private:
  duration __d_;

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr time_point()
      : __d_(duration::zero())
  {}
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit time_point(const duration& __d)
      : __d_(__d)
  {}

  // conversions
  _CCCL_TEMPLATE(class _Duration2)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _Duration2, duration))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr time_point(const time_point<clock, _Duration2>& __t)
      : __d_(__t.time_since_epoch())
  {}

  // observer

  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration time_since_epoch() const
  {
    return __d_;
  }

  // arithmetic

  _LIBCUDACXX_HIDE_FROM_ABI constexpr time_point& operator+=(const duration& __d)
  {
    __d_ += __d;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr time_point& operator-=(const duration& __d)
  {
    __d_ -= __d;
    return *this;
  }

  // special values

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr time_point min() noexcept
  {
    return time_point(duration::min());
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr time_point max() noexcept
  {
    return time_point(duration::max());
  }
};

} // namespace chrono

template <class _Clock, class _Duration1, class _Duration2>
struct common_type<chrono::time_point<_Clock, _Duration1>, chrono::time_point<_Clock, _Duration2>>
{
  using type = chrono::time_point<_Clock, common_type_t<_Duration1, _Duration2>>;
};

namespace chrono
{

template <class _ToDuration, class _Clock, class _Duration>
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, _ToDuration>
time_point_cast(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>(chrono::duration_cast<_ToDuration>(__t.time_since_epoch()));
}

_CCCL_TEMPLATE(class _ToDuration, class _Clock, class _Duration)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, _ToDuration> floor(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>{chrono::floor<_ToDuration>(__t.time_since_epoch())};
}

_CCCL_TEMPLATE(class _ToDuration, class _Clock, class _Duration)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, _ToDuration> ceil(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>{chrono::ceil<_ToDuration>(__t.time_since_epoch())};
}

_CCCL_TEMPLATE(class _ToDuration, class _Clock, class _Duration)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, _ToDuration> round(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>{chrono::round<_ToDuration>(__t.time_since_epoch())};
}

_CCCL_TEMPLATE(class _Rep, class _Period)
_CCCL_REQUIRES(numeric_limits<_Rep>::is_signed)
_LIBCUDACXX_HIDE_FROM_ABI constexpr duration<_Rep, _Period> abs(duration<_Rep, _Period> __d)
{
  return __d >= __d.zero() ? +__d : -__d;
}

// time_point ==

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator==(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return __lhs.time_since_epoch() == __rhs.time_since_epoch();
}

// time_point !=

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator!=(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return !(__lhs == __rhs);
}

// time_point <

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator<(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return __lhs.time_since_epoch() < __rhs.time_since_epoch();
}

// time_point >

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return __rhs < __lhs;
}

// time_point <=

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator<=(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return !(__rhs < __lhs);
}

// time_point >=

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>=(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return !(__lhs < __rhs);
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class _Clock, class _Duration1, three_way_comparable_with<_Duration1> _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto
operator<=>(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return __lhs.time_since_epoch() <=> __rhs.time_since_epoch();
}

#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

// time_point operator+(time_point x, duration y);

template <class _Clock, class _Duration1, class _Rep2, class _Period2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, common_type_t<_Duration1, duration<_Rep2, _Period2>>>
operator+(const time_point<_Clock, _Duration1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
  using _Tr = time_point<_Clock, common_type_t<_Duration1, duration<_Rep2, _Period2>>>;
  return _Tr(__lhs.time_since_epoch() + __rhs);
}

// time_point operator+(duration x, time_point y);

template <class _Rep1, class _Period1, class _Clock, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, common_type_t<duration<_Rep1, _Period1>, _Duration2>>
operator+(const duration<_Rep1, _Period1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return __rhs + __lhs;
}

// time_point operator-(time_point x, duration y);

template <class _Clock, class _Duration1, class _Rep2, class _Period2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr time_point<_Clock, common_type_t<_Duration1, duration<_Rep2, _Period2>>>
operator-(const time_point<_Clock, _Duration1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
  using _Ret = time_point<_Clock, common_type_t<_Duration1, duration<_Rep2, _Period2>>>;
  return _Ret(__lhs.time_since_epoch() - __rhs);
}

// duration operator-(time_point x, time_point y);

template <class _Clock, class _Duration1, class _Duration2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<_Duration1, _Duration2>
operator-(const time_point<_Clock, _Duration1>& __lhs, const time_point<_Clock, _Duration2>& __rhs)
{
  return __lhs.time_since_epoch() - __rhs.time_since_epoch();
}

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_TIME_POINT_H
