//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_OPERATOR_NOTHROW_H
#define _LIBCUDACXX___NEW_OPERATOR_NOTHROW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT nothrow_t
{
  explicit nothrow_t() = default;
};

_CCCL_GLOBAL_CONSTANT nothrow_t nothrow{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_OPERATOR_NOTHROW_H
