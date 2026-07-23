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
template <class _Hierarchy, class _Tp, class _RedOp, bool _Deterministic>
struct __reduce_scratch_select<this_warp<_Hierarchy>, _Tp, _RedOp, _Deterministic>
{
  using __smem_type _CCCL_NODEBUG_ALIAS = typename ::cub::WarpReduce<_Tp>::TempStorage;
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
  const this_warp<_Hierarchy>& __group,
  _Tp (&__thread_data)[_Np],
  _RedFn __red_fn,
  _SmemScratch& __smem_scratch,
  _GmemScratch&)
{
  const auto __result = ::cub::WarpReduce<_Tp>{__smem_scratch}.Reduce(__thread_data, __red_fn);
  if constexpr (_Broadcasted)
  {
    return ::cuda::device::warp_shuffle_idx(__result, 0).data;
  }
  else
  {
    return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_WARP_CUH
