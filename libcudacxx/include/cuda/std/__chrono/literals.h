//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_LITERALS_H
#define _LIBCUDACXX___CHRONO_LITERALS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/day.h>
#include <cuda/std/__chrono/year.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

inline namespace literals
{
inline namespace chrono_literals
{

using __literal_float_t = double;

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::hours operator""h(unsigned long long __h)
{
  return chrono::hours(static_cast<chrono::hours::rep>(__h));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::duration<__literal_float_t, ratio<3600, 1>> operator""h(long double __h)
{
  return chrono::duration<__literal_float_t, ratio<3600, 1>>(__h);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::minutes operator""min(unsigned long long __m)
{
  return chrono::minutes(static_cast<chrono::minutes::rep>(__m));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::duration<__literal_float_t, ratio<60, 1>> operator""min(long double __m)
{
  return chrono::duration<__literal_float_t, ratio<60, 1>>(__m);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::seconds operator""s(unsigned long long __s)
{
  return chrono::seconds(static_cast<chrono::seconds::rep>(__s));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::duration<__literal_float_t> operator""s(long double __s)
{
  return chrono::duration<__literal_float_t>(__s);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::milliseconds operator""ms(unsigned long long __ms)
{
  return chrono::milliseconds(static_cast<chrono::milliseconds::rep>(__ms));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::duration<__literal_float_t, milli> operator""ms(long double __ms)
{
  return chrono::duration<__literal_float_t, milli>(__ms);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::microseconds operator""us(unsigned long long __us)
{
  return chrono::microseconds(static_cast<chrono::microseconds::rep>(__us));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::duration<__literal_float_t, micro> operator""us(long double __us)
{
  return chrono::duration<__literal_float_t, micro>(__us);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::nanoseconds operator""ns(unsigned long long __ns)
{
  return chrono::nanoseconds(static_cast<chrono::nanoseconds::rep>(__ns));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::duration<__literal_float_t, nano> operator""ns(long double __ns)
{
  return chrono::duration<__literal_float_t, nano>(__ns);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::day operator""d(unsigned long long __d) noexcept
{
  return chrono::day(static_cast<unsigned>(__d));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chrono::year operator""y(unsigned long long __y) noexcept
{
  return chrono::year(static_cast<int>(__y));
}
} // namespace chrono_literals
} // namespace literals

namespace chrono
{ // hoist the literals into namespace std::chrono
using namespace literals::chrono_literals;
} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_LITERALS_H
