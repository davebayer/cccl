//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___OPTIONAL_NULLOPT_H
#define _LIBCUDACXX___OPTIONAL_NULLOPT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT nullopt_t
{
  struct __secret_tag
  {
    _CCCL_HIDE_FROM_ABI explicit __secret_tag() = default;
  };
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit nullopt_t(__secret_tag, __secret_tag) noexcept {}
};

_CCCL_GLOBAL_CONSTANT nullopt_t nullopt{nullopt_t::__secret_tag{}, nullopt_t::__secret_tag{}};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___OPTIONAL_NULLOPT_H
