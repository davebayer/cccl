//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_FWD_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_FWD_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
struct __reduce;

template <class _Group, class _Tp, class _RedOp, bool _Deterministic, class = void>
struct __reduce_scratch_select;

template <bool _Dummy = false>
_CCCL_DEVICE_API auto __reduce_impl(...)
{
  static_assert(_Dummy, "cudax::coop::reduce is not supported for the group");
}
// _CCCL_GLOBAL_CONSTANT __reduce_fn reduce{};
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_FWD_CUH
