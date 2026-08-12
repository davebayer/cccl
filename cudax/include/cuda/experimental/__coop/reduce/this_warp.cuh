//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_WARP_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_WARP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/warp/warp_reduce.cuh>

#include <cuda/__warp/warp_shuffle.h>
#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
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
                     ::cuda::std::enable_if_t<::cuda::std::is_same_v<typename _Group::unit_type, warp_level>&& ::cuda::
                                                std::is_same_v<typename _Group::level_type, warp_level>>>
{
  using _CubWarpReduce = ::cub::WarpReduce<_Tp>;

  using _SmemScratch = typename _CubWarpReduce::TempStorage;
  using _GmemScratch = __empty_gmem_scratch;

  [[nodiscard]] _CCCL_DEVICE_API auto
  operator()(const _Group& __group, ::cuda::std::span<_Tp, _Np> __thread_data, _RedFn& __red_fn)
  {
    const auto __result = _CubWarpReduce{__smem_scratch_}.Reduce(__thread_data, __red_fn);
    if constexpr (_Broadcasted)
    {
      return ::cuda::device::warp_shuffle_idx(__result, 0).data;
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

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_WARP_CUH
