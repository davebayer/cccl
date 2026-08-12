//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_GRID_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_GRID_CUH

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
#include <cuda/experimental/group.cuh>

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
                     ::cuda::std::enable_if_t<::cuda::std::is_same_v<typename _Group::unit_type, grid_level>&& ::cuda::
                                                std::is_same_v<typename _Group::level_type, grid_level>>>
{
  using _GridExts = decltype(cluster.extents(grid, __group.hierarchy()));
  static_assert(_GridExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the grid level to have all static extents.");

  constexpr auto __nclusters_in_grid =
    _GridExts::static_extent(0) * _GridExts::static_extent(1) * _GridExts::static_extent(2);

  using _ClusterReduce = __reduce_impl<false, this_cluster, _Tp, _Np, _RedFn>;
  using _BlockReduce   = __reduce_impl<false, this_cluster, _Tp, _Np, _RedFn>;

  struct _SmemScratch
  {
    union
    {
      typename _CubBlockReduce::TempStorage __cub_block_reduce_;
      typename _CubWarpReduce::TempStorage __cub_warp_reduce_;
    };
    _Tp __partials_[__nblocks_in_cluster];
    _Tp __bcast_;
  };

  struct _GmemScratch
  {
    _Tp __cluster_partials_[__nclusters_in_grid];
  };

  [[nodiscard]] _CCCL_DEVICE_API auto
  operator()(const _Group& __group, ::cuda::std::span<_Tp, _Np> __thread_data, _RedFn& __red_fn)
  {
    const auto __partial = _CubBlockReduce{__smem_scratch_.__cub_block_reduce_}.Reduce(
      *reinterpret_cast<_Tp(*)[_Np]>(__thread_data.data()), __red_fn);
    _Tp __result{};
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      ({
        const auto __root_scratch = static_cast<_SmemScratch*>(::__cluster_map_shared_rank(&__smem_scratch_, 0));
        auto& __partials_root     = __root_scratch->__partials_;

        if (gpu_thread.is_root_rank(this_block{__group.hierarchy()}))
        {
          __partials_root[block.rank(__group)] = __partial;
        }
        __group.sync_aligned();

        if (warp.is_root_rank(__group))
        {
          this_warp __warp{__group.hierarchy()};
          const auto __value = (gpu_thread.rank(__warp) < __nblocks_in_cluster)
                               ? __smem_scratch_.__partials_[gpu_thread.rank(__warp)]
                               : ::cuda::identity_element<_RedFn, _Tp>();
          __result           = _CubWarpReduce{__smem_scratch_.__cub_warp_reduce_}.Reduce(__value, __red_fn);
        }

        if constexpr (_Broadcasted)
        {
          if (gpu_thread.is_root_rank(__group))
          {
            __smem_scratch_.__bcast_ = __result;
          }
          __group.sync_aligned();
          __result = __root_scratch->__bcast_;

          // Wait until all threads are done reading the result.
          __group.sync_aligned();
        }
      }),
      ({ _CCCL_VERIFY(false, "not implemented yet"); }))

    if constexpr (_Broadcasted)
    {
      return __result;
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

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_GRID_CUH
