//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_DURATION_H
#define _LIBCUDACXX___CHRONO_DURATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// #include <cuda/std/__compare/ordering.h>
// #include <cuda/std/__compare/three_way_comparable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/limits>
#include <cuda/std/ratio>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

template <class _Rep, class _Period = ratio<1>>
class duration;

template <class _Tp>
inline constexpr bool __is_duration_v = false;

template <class _Rep, class _Period>
inline constexpr bool __is_duration_v<duration<_Rep, _Period>> = true;

template <class _Rep, class _Period>
inline constexpr bool __is_duration_v<const duration<_Rep, _Period>> = true;

template <class _Rep, class _Period>
inline constexpr bool __is_duration_v<volatile duration<_Rep, _Period>> = true;

template <class _Rep, class _Period>
inline constexpr bool __is_duration_v<const volatile duration<_Rep, _Period>> = true;

} // namespace chrono

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
struct common_type<chrono::duration<_Rep1, _Period1>, chrono::duration<_Rep2, _Period2>>
{
  using type = chrono::duration<common_type_t<_Rep1, _Rep2>, __ratio_gcd<_Period1, _Period2>>;
};

namespace chrono
{

// duration_cast

template <class _FromDuration,
          class _ToDuration,
          class _Period = typename ratio_divide<typename _FromDuration::period, typename _ToDuration::period>::type,
          bool          = _Period::num == 1,
          bool          = _Period::den == 1>
struct __duration_cast;

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, true, true>
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration operator()(const _FromDuration& __fd) const
  {
    return _ToDuration(static_cast<typename _ToDuration::rep>(__fd.count()));
  }
};

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, true, false>
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration operator()(const _FromDuration& __fd) const
  {
    using _Ct = common_type_t<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>;
    return _ToDuration(
      static_cast<typename _ToDuration::rep>(static_cast<_Ct>(__fd.count()) / static_cast<_Ct>(_Period::den)));
  }
};

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, false, true>
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration operator()(const _FromDuration& __fd) const
  {
    using _Ct = common_type_t<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>;
    return _ToDuration(
      static_cast<typename _ToDuration::rep>(static_cast<_Ct>(__fd.count()) * static_cast<_Ct>(_Period::num)));
  }
};

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, false, false>
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration operator()(const _FromDuration& __fd) const
  {
    using _Ct = common_type_t<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>;
    return _ToDuration(static_cast<typename _ToDuration::rep>(
      static_cast<_Ct>(__fd.count()) * static_cast<_Ct>(_Period::num) / static_cast<_Ct>(_Period::den)));
  }
};

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
_LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration duration_cast(const duration<_Rep, _Period>& __fd)
{
  return __duration_cast<duration<_Rep, _Period>, _ToDuration>()(__fd);
}

template <class _Rep>
struct treat_as_floating_point : is_floating_point<_Rep>
{};

template <class _Rep>
inline constexpr bool treat_as_floating_point_v = treat_as_floating_point<_Rep>::value;

template <class _Rep>
struct duration_values
{
public:
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Rep zero() noexcept
  {
    return _Rep(0);
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Rep max() noexcept
  {
    return numeric_limits<_Rep>::max();
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Rep min() noexcept
  {
    return numeric_limits<_Rep>::lowest();
  }
};

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration floor(const duration<_Rep, _Period>& __d)
{
  _ToDuration __t = chrono::duration_cast<_ToDuration>(__d);
  if (__t > __d)
  {
    __t = __t - _ToDuration{1};
  }
  return __t;
}

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration ceil(const duration<_Rep, _Period>& __d)
{
  _ToDuration __t = chrono::duration_cast<_ToDuration>(__d);
  if (__t < __d)
  {
    __t = __t + _ToDuration{1};
  }
  return __t;
}

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_duration_v<_ToDuration>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _ToDuration round(const duration<_Rep, _Period>& __d)
{
  _ToDuration __lower = chrono::floor<_ToDuration>(__d);
  _ToDuration __upper = __lower + _ToDuration{1};
  auto __lower_diff   = __d - __lower;
  auto __upper_diff   = __upper - __d;
  if (__lower_diff < __upper_diff)
  {
    return __lower;
  }
  if (__lower_diff > __upper_diff)
  {
    return __upper;
  }
  return __lower.count() & 1 ? __upper : __lower;
}

// duration

template <class _Rep, class _Period>
class duration
{
  static_assert(!__is_duration_v<_Rep>, "A duration representation can not be a duration");
  static_assert(__is_ratio_v<_Period>, "Second template parameter of duration must be a std::ratio");
  static_assert(_Period::num > 0, "duration period must be positive");

  template <class _R1, class _R2>
  struct __no_overflow
  {
  private:
    static const intmax_t __gcd_n1_n2 = __static_gcd<_R1::num, _R2::num>;
    static const intmax_t __gcd_d1_d2 = __static_gcd<_R1::den, _R2::den>;
    static const intmax_t __n1        = _R1::num / __gcd_n1_n2;
    static const intmax_t __d1        = _R1::den / __gcd_d1_d2;
    static const intmax_t __n2        = _R2::num / __gcd_n1_n2;
    static const intmax_t __d2        = _R2::den / __gcd_d1_d2;
    static const intmax_t max         = -((intmax_t(1) << (sizeof(intmax_t) * CHAR_BIT - 1)) + 1);

    template <intmax_t _Xp, intmax_t _Yp, bool __overflow>
    struct __mul // __overflow == false
    {
      static const intmax_t value = _Xp * _Yp;
    };

    template <intmax_t _Xp, intmax_t _Yp>
    struct __mul<_Xp, _Yp, true>
    {
      static const intmax_t value = 1;
    };

  public:
    static const bool value = (__n1 <= max / __d2) && (__n2 <= max / __d1);
    using type              = ratio<__mul<__n1, __d2, !value>::value, __mul<__n2, __d1, !value>::value>;
  };

public:
  using rep    = _Rep;
  using period = typename _Period::type;

private:
  rep __rep_;

public:
  _CCCL_HIDE_FROM_ABI constexpr duration() noexcept = default;

  _CCCL_TEMPLATE(class _Rep2)
  _CCCL_REQUIRES(
    is_convertible_v<const _Rep2&, rep> _CCCL_AND(treat_as_floating_point_v<rep> || !treat_as_floating_point_v<_Rep2>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit duration(const _Rep2& __r)
      : __rep_(__r)
  {}

  // conversions
  _CCCL_TEMPLATE(class _Rep2, class _Period2)
  _CCCL_REQUIRES(__no_overflow<_Period2, period>::value _CCCL_AND(
    treat_as_floating_point_v<rep>
    || (__no_overflow<_Period2, period>::type::den == 1 && !treat_as_floating_point_v<_Rep2>) ))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration(const duration<_Rep2, _Period2>& __d)
      : __rep_(chrono::duration_cast<duration>(__d).count())
  {}

  // observer

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr rep count() const
  {
    return __rep_;
  }

  // arithmetic

  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<duration> operator+() const
  {
    return common_type_t<duration>(*this);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<duration> operator-() const
  {
    return common_type_t<duration>(-__rep_);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator++()
  {
    ++__rep_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration operator++(int)
  {
    return duration(__rep_++);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator--()
  {
    --__rep_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration operator--(int)
  {
    return duration(__rep_--);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator+=(const duration& __d)
  {
    __rep_ += __d.count();
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator-=(const duration& __d)
  {
    __rep_ -= __d.count();
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator*=(const rep& __rhs)
  {
    __rep_ *= __rhs;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator/=(const rep& __rhs)
  {
    __rep_ /= __rhs;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator%=(const rep& __rhs)
  {
    __rep_ %= __rhs;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr duration& operator%=(const duration& __rhs)
  {
    __rep_ %= __rhs.count();
    return *this;
  }

  // Comparisons

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    if constexpr (is_same_v<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>)
    {
      return __lhs.count() == __rhs.count();
    }
    else
    {
      using _Ct = common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>;
      return _Ct(__lhs).count() == _Ct(__rhs).count();
    }
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return !(__lhs == __rhs);
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator<(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    if constexpr (is_same_v<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>)
    {
      return __lhs.count() < __rhs.count();
    }
    else
    {
      using _Ct = common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>;
      return __duration_lt<_Ct, _Ct>()(__lhs, __rhs);
    }
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator>(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return __rhs < __lhs;
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator<=(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return !(__rhs < __lhs);
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator>=(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return !(__lhs < __rhs);
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>
  operator+(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Cd = common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>;
    return _Cd(_Cd(__lhs).count() + _Cd(__rhs).count());
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>
  operator-(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Cd = common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>;
    return _Cd(_Cd(__lhs).count() - _Cd(__rhs).count());
  }

  _CCCL_TEMPLATE(class _Rep1, class _Period, class _Rep2)
  _CCCL_REQUIRES(is_convertible_v<const _Rep2&, common_type_t<_Rep1, _Rep2>>)
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr duration<common_type_t<_Rep1, _Rep2>, _Period>
  operator*(const duration<_Rep1, _Period>& __d, const _Rep2& __s)
  {
    using _Cr = common_type_t<_Rep1, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() * static_cast<_Cr>(__s));
  }

  _CCCL_TEMPLATE(class _Rep1, class _Period, class _Rep2)
  _CCCL_REQUIRES(is_convertible_v<const _Rep1&, common_type_t<_Rep1, _Rep2>>)
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr duration<common_type_t<_Rep1, _Rep2>, _Period>
  operator*(const _Rep1& __s, const duration<_Rep2, _Period>& __d)
  {
    return __d * __s;
  }

  _CCCL_TEMPLATE(class _Rep1, class _Period, class _Rep2)
  _CCCL_REQUIRES((!__is_duration_v<_Rep2>) _CCCL_AND is_convertible_v<const _Rep2&, common_type_t<_Rep1, _Rep2>>)
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr duration<common_type_t<_Rep1, _Rep2>, _Period>
  operator/(const duration<_Rep1, _Period>& __d, const _Rep2& __s)
  {
    using _Cr = common_type_t<_Rep1, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() / static_cast<_Cr>(__s));
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<_Rep1, _Rep2>
  operator/(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Ct = common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>;
    return _Ct(__lhs).count() / _Ct(__rhs).count();
  }

  _CCCL_TEMPLATE(class _Rep1, class _Period, class _Rep2)
  _CCCL_REQUIRES((!__is_duration_v<_Rep2>) _CCCL_AND is_convertible_v<const _Rep2&, common_type_t<_Rep1, _Rep2>>)
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr duration<common_type_t<_Rep1, _Rep2>, _Period>
  operator%(const duration<_Rep1, _Period>& __d, const _Rep2& __s)
  {
    using _Cr = common_type_t<_Rep1, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() % static_cast<_Cr>(__s));
  }

  template <class _Rep1, class _Period1, class _Rep2, class _Period2>
  _CCCL_NODISCARD_FRIEND
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>
  operator%(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Cr = common_type_t<_Rep1, _Rep2>;
    using _Cd = common_type_t<duration<_Rep1, _Period1>, duration<_Rep2, _Period2>>;
    return _Cd(static_cast<_Cr>(_Cd(__lhs).count()) % static_cast<_Cr>(_Cd(__rhs).count()));
  }

  // special values

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr duration zero() noexcept
  {
    return duration(duration_values<rep>::zero());
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr duration min() noexcept
  {
    return duration(duration_values<rep>::min());
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr duration max() noexcept
  {
    return duration(duration_values<rep>::max());
  }
};

using nanoseconds  = duration<long long, nano>;
using microseconds = duration<long long, micro>;
using milliseconds = duration<long long, milli>;
using seconds      = duration<long long>;
using minutes      = duration<long, ratio<60>>;
using hours        = duration<long, ratio<3600>>;
using days         = duration<int, ratio_multiply<ratio<24>, hours::period>>;
using weeks        = duration<int, ratio_multiply<ratio<7>, days::period>>;
using years        = duration<int, ratio_multiply<ratio<146097, 400>, days::period>>;
using months       = duration<int, ratio_divide<years::period, ratio<12>>>;

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___CHRONO_DURATION_H
