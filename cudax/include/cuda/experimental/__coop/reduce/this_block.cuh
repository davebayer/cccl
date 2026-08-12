//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_BLOCK_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_BLOCK_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>

#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/optional>
#include <cuda/std/span>

#include <cuda/experimental/__coop/reduce/entry.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <bool _Broadcasted, class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
struct __reduce_impl<_Broadcasted,
                     _Group,
                     _Tp,
                     _Np,
                     _RedFn,
                     ::cuda::std::enable_if_t<::cuda::std::is_same_v<typename _Group::unit_type, block_level>&& ::cuda::
                                                std::is_same_v<typename _Group::level_type, block_level>>>
{
  using _BlockExts = decltype(gpu_thread.extents(block, ::cuda::std::declval<typename _Group::hierarchy_type>()));
  static_assert(_BlockExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the block level to have all static extents.");

  using _CubBlockReduce =
    ::cub::BlockReduce<_Tp,
                       static_cast<int>(_BlockExts::static_extent(0)),
                       ::cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                       static_cast<int>(_BlockExts::static_extent(1)),
                       static_cast<int>(_BlockExts::static_extent(2))>;

  union _SmemScratch
  {
    typename _CubBlockReduce::TempStorage __cub_block_reduce_;
    _Tp __bcast_;
  };
  using _GmemScratch = __empty_gmem_scratch;

  [[nodiscard]] _CCCL_DEVICE_API auto
  operator()(const _Group& __group, ::cuda::std::span<_Tp, _Np> __thread_data, _RedFn& __red_fn)
  {
    const auto __result = _CubBlockReduce{__smem_scratch_.__cub_block_reduce_}.Reduce(
      *reinterpret_cast<_Tp(*)[_Np]>(__thread_data.data()), __red_fn);
    if constexpr (_Broadcasted)
    {
      if (gpu_thread.is_root_rank(__group))
      {
        __smem_scratch_.__bcast_ = __result;
      }
      __group.sync_aligned();
      return __smem_scratch_.__bcast_;
    }
    else
    {
      return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
    }
  }

  _SmemScratch& __smem_scratch_;
  _GmemScratch& __gmem_scratch_;
};
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_BLOCK_CUH
