//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_STATIC_DISPATCH_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_STATIC_DISPATCH_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/coop/__group/concepts.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
struct __static_dispatch_impl_functor
{
  template <class _Group, class _Callable, ::cuda::std::size_t _Rank>
  _CCCL_DEVICE_API void operator()(
    const _Group& __group, _Callable&& __callable, ::cuda::std::integral_constant<::cuda::std::size_t, _Rank>) noexcept
  {
    using _MappingResult = typename _Group::__mapping_result_type;
    using _ViewMappingResult =
      __mapping_result<_MappingResult::static_group_count(),
                       _MappingResult::static_unit_count(_Rank),
                       _MappingResult::is_always_exhaustive(),
                       _MappingResult::is_always_contiguous()>;

    if (__group.__mapping_result().group_rank() == static_cast<::cuda::std::uint32_t>(_Rank))
    {
      ::cuda::std::forward<_Callable>(__callable)(group_view{__group});
    }
  }
};

template <class _Group, class _Callable, ::cuda::std::size_t... _Ranks>
_CCCL_DEVICE_API void
__static_dispatch_impl(const _Group& __group, _Callable&& __callable, ::cuda::std::index_sequence<_Ranks...>) noexcept
{
  (__static_dispatch_impl_functor{}(__group,
                                    ::cuda::std::forward<_Callable>(__callable),
                                    ::cuda::std::integral_constant<::cuda::std::size_t, _Ranks>{}),
   ...);
}

_CCCL_TEMPLATE(class _Group, class _Callable)
_CCCL_REQUIRES(group<_Group>)
_CCCL_DEVICE_API void static_dispatch(const _Group& __group, _Callable&& __callable) noexcept
{
  using _MappingResult = typename _Group::__mapping_result_type;
  if constexpr (_MappingResult::static_group_count() != ::cuda::std::dynamic_extent)
  {
    ::cuda::experimental::__static_dispatch_impl(
      __group,
      ::cuda::std::forward<_Callable>(__callable),
      ::cuda::std::make_index_sequence<_MappingResult::static_group_count()>{});
  }
  else
  {
    ::cuda::std::forward<_Callable>(__callable)(group_view{__group});
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_STATIC_DISPATCH_CUH
