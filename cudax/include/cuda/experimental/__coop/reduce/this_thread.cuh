//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_THREAD_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_THREAD_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_reduce.cuh>

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
struct __reduce_scratch_select<this_thread<_Hierarchy>, _Tp, _RedOp, _Deterministic>
{
  using __smem_type _CCCL_NODEBUG_ALIAS = __smem_empty_scratch;
  using __gmem_type _CCCL_NODEBUG_ALIAS = __gmem_empty_scratch;
};

template <bool _ScratchQuery,
          bool _Broadcasted,
          class _Hierarchy,
          class _Tp,
          ::cuda::std::size_t _Np,
          class _RedFn,
          class _SmemScratch,
          class _GmemScratch>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>,
  const this_thread<_Hierarchy>&,
  _Tp (&__thread_data)[_Np],
  _RedFn __red_fn,
  _SmemScratch&,
  _GmemScratch&)
{
  const auto __result = ::cub::ThreadReduce(__thread_data, __red_fn);
  if constexpr (_Broadcasted)
  {
    return __result;
  }
  else
  {
    return ::cuda::std::optional{__result};
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_THIS_THREAD_CUH
