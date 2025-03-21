//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_OPERATOR_DELETE_H
#define _LIBCUDACXX___NEW_OPERATOR_DELETE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstdlib/malloc.h>
#include <cuda/std/__new/align_val_t.h>
#include <cuda/std/__new/nothrow.h>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <new>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void __cccl_operator_delete(void* __ptr) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr);), (_CUDA_VSTD::free(__ptr);))
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void
__cccl_operator_delete(void* __ptr, [[maybe_unused]] align_val_t __align) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr, __align);), (_CUDA_VSTD::free(__ptr);))
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void __cccl_operator_delete(void* __ptr, [[maybe_unused]] size_t __size) noexcept
{
#if _LIBCUDACXX_HAS_SIZED_DEALLOCATION()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr, __size);), (_CUDA_VSTD::free(__ptr);))
#else // ^^^ _LIBCUDACXX_HAS_SIZED_DEALLOCATION() ^^^ / vvv !_LIBCUDACXX_HAS_SIZED_DEALLOCATION() vvv
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr);), (_CUDA_VSTD::free(__ptr);))
#endif // _LIBCUDACXX_HAS_SIZED_DEALLOCATION()
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void
__cccl_operator_delete(void* __ptr, [[maybe_unused]] size_t __size, [[maybe_unused]] align_val_t __align) noexcept
{
#if _LIBCUDACXX_HAS_SIZED_DEALLOCATION()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr, __size, __align);), (_CUDA_VSTD::free(__ptr);))
#else // ^^^ _LIBCUDACXX_HAS_SIZED_DEALLOCATION() ^^^ / vvv !_LIBCUDACXX_HAS_SIZED_DEALLOCATION() vvv
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr, __align);), (_CUDA_VSTD::free(__ptr);))
#endif // _LIBCUDACXX_HAS_SIZED_DEALLOCATION()
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void __cccl_operator_delete(void* __ptr, [[maybe_unused]] const nothrow_t&) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr, ::std::nothrow);), (_CUDA_VSTD::free(__ptr);))
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void
__cccl_operator_delete(void* __ptr, [[maybe_unused]] align_val_t __align, [[maybe_unused]] const nothrow_t&) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::operator delete(__ptr, __align, ::std::nothrow);), (_CUDA_VSTD::free(__ptr);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_OPERATOR_DELETE_H
