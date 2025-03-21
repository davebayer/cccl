//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_OPERATOR_NEW_H
#define _LIBCUDACXX___NEW_OPERATOR_NEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstdlib/aligned_alloc.h>
#include <cuda/std/__cstdlib/malloc.h>
#include <cuda/std/__new/align_val_t.h>
#include <cuda/std/__new/bad_alloc.h>
#include <cuda/std/__new/nothrow.h>
#include <cuda/std/cstddef>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <new>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __is_overaligned_for_new(size_t __align) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return __align > __STDCPP_DEFAULT_NEW_ALIGNMENT__;), (return __align > 16;))
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __is_overaligned_for_new(align_val_t __align) noexcept
{
  return __is_overaligned_for_new(static_cast<size_t>(__align));
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* __cccl_operator_new(size_t __size)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return ::operator new(__size);), (if (void* __ptr = _CUDA_VSTD::malloc(__size)) {
                      return __ptr;
                    } _CUDA_VSTD::__throw_bad_alloc();))
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* __cccl_operator_new(size_t __size, align_val_t __align)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::operator new(__size, __align);),
                    (if (void* __ptr = _CUDA_VSTD::aligned_alloc(__size, static_cast<size_t>(__align))) {
                      return __ptr;
                    } _CUDA_VSTD::__throw_bad_alloc();))
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* __cccl_operator_new(size_t __size, const nothrow_t&) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return ::operator new(__size, ::std::nothrow);), (return _CUDA_VSTD::malloc(__size);))
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void*
__cccl_operator_new(size_t __size, align_val_t __align, const nothrow_t&) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::operator new(__size, __align, ::std::nothrow);),
                    (return _CUDA_VSTD::aligned_alloc(__size, static_cast<size_t>(__align));))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_OPERATOR_NEW_H
