//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_NOTHROW
#define _LIBCUDACXX___NEW_NOTHROW

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <new>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct nothrow_t
{
  _CCCL_HIDE_FROM_ABI explicit nothrow_t() = default;
};

_CCCL_GLOBAL_CONSTANT nothrow_t nothrow{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_NOTHROW
