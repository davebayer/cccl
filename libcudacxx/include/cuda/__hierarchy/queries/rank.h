//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_QUERIES_RANK_H
#define _CUDA___HIERARCHY_QUERIES_RANK_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/queries/extents.h>
#  include <cuda/__hierarchy/queries/index.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/conditional.h>
#  include <cuda/std/cstdint>

#  if _CCCL_CUDA_COMPILATION()
#    include <cuda/__ptx/instructions/get_sreg.h> // IWYU pragma: keep
#  endif // _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_CUDA_COMPILATION()

// cudafe++ makes the queries (that are device only) return void when compiling for host, which causes host compilers
// to warn about applying [[nodiscard]] to a function that returns void.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(nodiscard_doesnt_apply)
#    if _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
#    endif // _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)

template <class _Tp, class _Exts, class _Index>
[[nodiscard]] _CCCL_DEVICE_API
_Tp __hierarchy_linear_rank(_Tp __rank, const _Exts& __exts, const hierarchy_query_result<_Index>& __idx) noexcept
{
  // A wide enclosing rank is scaled only once. The block's count and local rank fit in _Index.
  _Index __local = sizeof(_Tp) > sizeof(_Index) ? _Index{0} : static_cast<_Index>(__rank);
  for (::cuda::std::size_t __i = _Exts::rank(); __i > 0; --__i)
  {
    __local = (__local * __exts.extent(__i - 1)) + __idx[__i - 1];
  }
  if constexpr (sizeof(_Tp) > sizeof(_Index))
  {
    _Index __count = 1;
    for (::cuda::std::size_t __i = 0; __i < _Exts::rank(); ++__i)
    {
      __count *= __exts.extent(__i);
    }
    return static_cast<_Tp>((__rank * __count) + __local);
  }
  else
  {
    return static_cast<_Tp>(__local);
  }
}

// Blocks and clusters have the same x-fastest ordering in a grid. Their y and z dimensions are at most 65535.
template <class _Unit>
struct __rank_in_grid
{
private:
  template <class _Tp, class _GridExts = ::cuda::std::dims<3, ::cuda::std::uint32_t>>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp
  __linearize(const hierarchy_query_result<::cuda::std::uint32_t>& __idx) noexcept
  {
    const auto __live = __extents_query_native<_Unit, grid_level>::template __call<::cuda::std::uint32_t>();
    ::cuda::std::uint32_t __dims[2]{__live.extent(0), __live.extent(1)};
    for (::cuda::std::size_t __i = 0; __i < 2; ++__i)
    {
      const auto __dim = __i < _GridExts::rank() ? _GridExts::static_extent(__i) : 1;
      // A physical grid axis of one also contains exactly one cluster for any preferred cluster shape.
      if (__dim == 1 || (::cuda::std::is_same_v<_Unit, block_level> && __dim != ::cuda::std::dynamic_extent))
      {
        __dims[__i] = static_cast<::cuda::std::uint32_t>(__dim);
      }
    }

    const ::cuda::std::uint32_t __yz = (__idx.z * __dims[1]) + __idx.y;
    using _Product                   = ::cuda::std::
      conditional_t<(sizeof(_Tp) > sizeof(::cuda::std::uint32_t)), ::cuda::std::uint64_t, ::cuda::std::uint32_t>;
    return static_cast<_Tp>((static_cast<_Product>(__yz) * __dims[0]) + __idx.x);
  }

public:
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    return __linearize<_Tp>(__index_query_native<_Unit, grid_level>::template __call<::cuda::std::uint32_t>());
  }

  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    const auto __idx = __index_query<_Unit, grid_level>::template __call<::cuda::std::uint32_t>(__hier);
    if constexpr (__has_bottom_unit_or_level_v<block_level, _Hierarchy> && _Hierarchy::template has_level<grid_level>())
    {
      // The grid's shape in blocks is fixed at launch. Only use the descriptor for its static dimensions.
      using _GridExts =
        decltype(__extents_query<block_level, grid_level>::template __call<::cuda::std::size_t>(__hier));
      return __linearize<_Tp, _GridExts>(__idx);
    }
    else
    {
      return __linearize<_Tp>(__idx);
    }
  }
};

// native hierarchy queries

template <class _Unit, class _Level>
struct __rank_query_native
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    using _NextLevel  = typename _Unit::__next_native_level;
    const auto __exts = __extents_query_native<_Unit, _NextLevel>::template __call<::cuda::std::uint32_t>();
    const auto __idx  = __index_query_native<_Unit, _NextLevel>::template __call<::cuda::std::uint32_t>();
    if constexpr (::cuda::std::is_same_v<_NextLevel, _Level>)
    {
      return static_cast<_Tp>(::cuda::__hierarchy_linear_rank(::cuda::std::uint32_t{0}, __exts, __idx));
    }
    else
    {
      using _Outer = ::cuda::std::conditional_t<::cuda::std::is_same_v<_Level, grid_level>, _Tp, ::cuda::std::uint32_t>;
      const auto __outer = __rank_query_native<_NextLevel, _Level>::template __call<_Outer>();
      return static_cast<_Tp>(::cuda::__hierarchy_linear_rank(__outer, __exts, __idx));
    }
  }
};

template <>
struct __rank_query_native<thread_level, warp_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    return static_cast<_Tp>(::cuda::ptx::get_sreg_laneid());
  }
};

template <>
struct __rank_query_native<block_level, cluster_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    ::cuda::std::uint32_t __rank = 0; // NOLINT(misc-const-correctness)
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__rank = ::__clusterRelativeBlockRank();))
    return static_cast<_Tp>(__rank);
  }
};

template <>
struct __rank_query_native<block_level, grid_level> : __rank_in_grid<block_level>
{};

template <>
struct __rank_query_native<cluster_level, grid_level> : __rank_in_grid<cluster_level>
{};

// hierarchy queries

template <class _Tp, class _Unit, class _NextLevel, class _Level, class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API _Tp __rank_query_generic(const _Hierarchy& __hier) noexcept
{
  // Thread and warp coordinates within a running block fit in 32 bits even for a wide grid rank.
  constexpr bool __within_block =
    ::cuda::std::is_same_v<_NextLevel, block_level>
    && (::cuda::std::is_same_v<_Unit, thread_level> || ::cuda::std::is_same_v<_Unit, warp_level>);
  using _Index = ::cuda::std::
    conditional_t<__within_block || (sizeof(_Tp) <= sizeof(::cuda::std::uint32_t)), ::cuda::std::uint32_t, _Tp>;
  const auto __exts = __extents_query<_Unit, _NextLevel>::template __call<_Index>(__hier);
  const auto __idx  = __index_query<_Unit, _NextLevel>::template __call<_Index>(__hier);
  if constexpr (::cuda::std::is_same_v<_NextLevel, _Level>)
  {
    return static_cast<_Tp>(::cuda::__hierarchy_linear_rank(_Index{0}, __exts, __idx));
  }
  else
  {
    const auto __outer = __rank_query<_NextLevel, _Level>::template __call<_Tp>(__hier);
    return ::cuda::__hierarchy_linear_rank(__outer, __exts, __idx);
  }
}

template <class _Unit, class _Level>
struct __rank_query
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    using _NextLevel = __next_hierarchy_level_t<_Unit, _Hierarchy>;
    return ::cuda::__rank_query_generic<_Tp, _Unit, _NextLevel, _Level>(__hier);
  }
};

template <>
struct __rank_query<thread_level, warp_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    return __index_query<thread_level, warp_level>::template __call<_Tp>(__hier).x;
  }
};

template <class _Level>
struct __rank_query<warp_level, _Level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    return ::cuda::__rank_query_generic<_Tp, warp_level, block_level, _Level>(__hier);
  }
};

template <>
struct __rank_query<block_level, cluster_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy&) noexcept
  {
    return __rank_query_native<block_level, cluster_level>::template __call<_Tp>();
  }
};

template <>
struct __rank_query<block_level, grid_level> : __rank_in_grid<block_level>
{};

template <>
struct __rank_query<cluster_level, grid_level> : __rank_in_grid<cluster_level>
{};

_CCCL_DIAG_POP
#  endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_QUERIES_RANK_H
