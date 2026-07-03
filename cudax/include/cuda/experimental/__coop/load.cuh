//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_LOAD_CUH
#define _CUDA_EXPERIMENTAL___COOP_LOAD_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy_n.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
_CCCL_TEMPLATE(class _Group, class _Range, class _Tp = ::cuda::std::ranges::range_value_t<_Range>, class _OutIt)
_CCCL_REQUIRES(::cuda::std::ranges::input_range<_Range>)
/*discard*/ _CCCL_DEVICE_API ::cuda::std::size_t
load_n(const _Group& __group, _Range&& __range, ::cuda::std::size_t __n, _OutIt __out_it)
{
  const auto __offset = ::cuda::gpu_thread.rank(__group) * __n;
  if constexpr (::cuda::std::ranges::sized_range<_Range>)
  {
    [[maybe_unused]] const auto __range_size = ::cuda::std::ranges::size(__range);
    _CCCL_ASSERT(__offset < __range_size, "out-of-bounds read detected in cuda::coop::load_n_to");
  }

  auto __it = __range.begin();
  ::cuda::std::advance(__it, __offset);
  ::cuda::std::copy_n(::cuda::std::move(__it), __n, ::cuda::std::move(__out_it));
  return __n;
}

_CCCL_TEMPLATE(class _Group, class _Range, class _Tp = ::cuda::std::ranges::range_value_t<_Range>, class _OutIt)
_CCCL_REQUIRES(::cuda::std::ranges::input_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
/*discard*/ _CCCL_DEVICE_API ::cuda::std::size_t
load_at_most_n(const _Group& __group, _Range&& __range, ::cuda::std::size_t __n, _OutIt __out_it)
{
  const auto __offset     = ::cuda::gpu_thread.rank(__group) * __n;
  const auto __range_size = ::cuda::std::ranges::size(__range);
  if (__offset > __range_size)
  {
    return 0;
  }

  if (__offset + __n > __range_size)
  {
    __n = __range_size - __offset;
  }

  auto __it = __range.begin();
  ::cuda::std::advance(__it, __offset);
  ::cuda::std::copy_n(::cuda::std::move(__it), __n, ::cuda::std::move(__out_it));
  return __n;
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_LOAD_CUH
