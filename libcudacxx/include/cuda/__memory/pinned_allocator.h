//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_PINNED_ALLOCATOR_H
#define _CUDA___MEMORY_PINNED_ALLOCATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/std/__utility/to_underlying.h>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp>
struct pinned_allocator
{
  using value_type = _Tp;

  _CCCL_HIDE_FROM_ABI constexpr pinned_allocator() noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr pinned_allocator(const pinned_allocator&) noexcept = default;

  template <class _Up>
  _CCCL_HIDE_FROM_ABI constexpr pinned_allocator(const pinned_allocator<_Up>&) noexcept
  {}

  _CCCL_HIDE_FROM_ABI constexpr pinned_allocator& operator=(const pinned_allocator&) noexcept = default;

  template <class _Up>
  _CCCL_HIDE_FROM_ABI constexpr pinned_allocator& operator=(const pinned_allocator<_Up>&) noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _Tp* allocate(_CUDA_VSTD::size_t __n)
  {
    if (__n == 0)
    {
      return nullptr;
    }
    if (__n > _CUDA_VSTD::numeric_limits<_CUDA_VSTD::size_t>::max() / sizeof(_Tp))
    {
      _CUDA_VSTD::__throw_bad_array_new_length();
    }

    void* __ptr{};
    const auto __error = ::cudaHostAlloc(&__ptr, __n * sizeof(_Tp), cudaHostAllocPortable | cudaHostAllocMapped);
    (void) ::cudaGetLastError(); // clear cuda error state

    if (__error == cudaErrorMemoryAllocation)
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }
    else if (__error != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(__error, "failed to allocate pinned memory");
    }
    return static_cast<_Tp*>(__ptr);
  }

  void deallocate(_Tp* __ptr, _CUDA_VSTD::size_t) noexcept
  {
    if (__ptr != nullptr)
    {
      (void) ::cudaFreeHost(__ptr); // we *must* ignore the return value
      (void) ::cudaGetLastError(); // clear cuda error state
    }
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr bool operator==(const pinned_allocator&) const noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr bool operator!=(const pinned_allocator&) const noexcept
  {
    return false;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___MEMORY_PINNED_ALLOCATOR_H
