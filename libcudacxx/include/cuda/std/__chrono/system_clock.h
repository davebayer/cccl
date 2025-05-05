//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_SYSTEM_CLOCK_H
#define _LIBCUDACXX___CHRONO_SYSTEM_CLOCK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/ctime>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

class _CCCL_TYPE_VISIBILITY_DEFAULT system_clock
{
public:
  using duration                  = microseconds;
  using rep                       = duration::rep;
  using period                    = duration::period;
  using time_point                = chrono::time_point<system_clock>;
  static constexpr bool is_steady = false;

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static time_point now() noexcept {}
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static time_t to_time_t(const time_point& __t) noexcept {}
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static time_point from_time_t(time_t __t) noexcept {}
};

template <class _Duration>
using sys_time    = time_point<system_clock, _Duration>;
using sys_seconds = sys_time<seconds>;
using sys_days    = sys_time<days>;

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_SYSTEM_CLOCK_H
