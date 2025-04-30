//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHRONO_CALENDAR_H
#define _LIBCUDACXX___CHRONO_CALENDAR_H

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

struct local_t
{};

template <class _Duration>
using local_time    = time_point<local_t, _Duration>;
using local_seconds = local_time<seconds>;
using local_days    = local_time<days>;

struct last_spec
{
  _CCCL_HIDE_FROM_ABI explicit constexpr last_spec() noexcept = default;
};

_CCCL_GLOBAL_CONSTANT last_spec last{};

} // namespace chrono

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CHRONO_CALENDAR_H
