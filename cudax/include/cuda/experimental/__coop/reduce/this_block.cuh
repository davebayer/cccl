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

#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/optional>

#include <cuda/experimental/__coop/reduce/fwd.cuh>
#include <cuda/experimental/__coop/scratch.cuh>
#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Tp,
          class _Hierarchy,
          bool _Deterministic,
          class _BlockExts = decltype(gpu_thread.extents(block, _Hierarchy{}))>
using __block_reduce = ::cub::BlockReduce<
  _Tp,
  static_cast<int>(_BlockExts::static_extent(0)),
  (_Deterministic) ? ::cub::BLOCK_REDUCE_WARP_REDUCTIONS : ::cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
  static_cast<int>(_BlockExts::static_extent(1)),
  static_cast<int>(_BlockExts::static_extent(2))>;

template <class _Hierarchy, class _Tp, class _RedOp, bool _Deterministic>
struct __reduce_scratch_select<this_warp<_Hierarchy>, _Tp, _RedOp, _Deterministic>
{
  using _BlockReduce = __block_reduce<_Tp, _Hierarchy, _Deterministic>;

  union __smem_type
  {
    typename _BlockReduce::TempStorage __block_reduce_;
    _Tp __bcast_;
  };

  using __gmem_type _CCCL_NODEBUG_ALIAS = __gmem_empty_scratch;
};

template <bool _Broadcasted,
          class _Hierarchy,
          class _Tp,
          ::cuda::std::size_t _Np,
          class _RedFn,
          class _SmemScratch,
          class _GmemScratch>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>,
  const this_block<_Hierarchy>& __group,
  _Tp (&__thread_data)[_Np],
  _RedFn __red_fn,
  _SmemScratch& __smem_scratch,
  _GmemScratch&)
{
  using _BlockReduce = __block_reduce<_Tp, _Hierarchy, _Deterministic>;

  const auto __result = _BlockReduce{__smem_scratch.__block_reduce_}.Reduce(__thread_data, __red_fn);
  if constexpr (_Broadcasted)
  {
    if (gpu_thread.is_root_rank(__group))
    {
      __smem_scratch.__bcast_ = __result;
    }
    __group.sync_aligned();
    return __smem_scratch.__bcast_;
  }
  else
  {
    return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_BLOCK_CUH
