//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_STEADY_CLOCK_H
#define _LIBCUDACXX___CHRONO_STEADY_CLOCK_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono
{

#if _LIBCUDACXX_HAS_MONOTONIC_CLOCK()
class _CCCL_TYPE_VISIBILITY_DEFAULT steady_clock
{
public:
  using duration                  = nanoseconds;
  using rep                       = duration::rep;
  using period                    = duration::period;
  using time_point                = chrono::time_point<steady_clock, duration>;
  static constexpr bool is_steady = true;

  static time_point now() noexcept;
};
#endif // _LIBCUDACXX_HAS_MONOTONIC_CLOCK()

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_STEADY_CLOCK_H
