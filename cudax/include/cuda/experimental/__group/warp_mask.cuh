//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_WARP_MASK_CUH
#define _CUDA_EXPERIMENTAL___GROUP_WARP_MASK_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/hierarchy>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
using __warp_mask_t = ::cuda::std::uint32_t;

template <class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API __warp_mask_t __warp_mask_this(const _Hierarchy& __hier) noexcept
{
  return 1u << warp.rank(block, __hier);
}

template <class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API __warp_mask_t __warp_mask_all(const _Hierarchy& __hier) noexcept
{
  return ::cuda::bitmask<__warp_mask_t>(0, warp.count(block, __hier));
}

template <bool _IsContiguous, class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API __warp_mask_t __make_warp_mask_for_n(
  __warp_mask_t __prev_warp_mask,
  ::cuda::std::uint32_t __n,
  ::cuda::std::uint32_t __rank,
  const _Hierarchy& __hier) noexcept
{
  const auto __this_mask = ::cuda::experimental::__warp_mask_this(__hier);
  const auto __all_mask  = ::cuda::experimental::__warp_mask_all(__hier);

  if constexpr (_IsContiguous)
  {
    auto __warp_mask    = __prev_warp_mask;
    const auto __wrank  = warp.rank(block, __hier);
    const auto __wcount = warp.count(block, __hier);

    if (__wrank > __rank)
    {
      __warp_mask &= __all_mask << (__wrank - __rank);
    }
    if (__wrank + (__n - __rank) < __wcount)
    {
      __warp_mask &= __all_mask >> (__wcount - __wrank - (__n - __rank));
    }
    return __warp_mask;
  }
  else
  {
    auto __warp_mask = ::cuda::device::lane_mask::this_lane();

    const auto __less_mask = __prev_warp_mask & ::cuda::device::lane_mask::all_less();
    const auto __nless     = ::cuda::std::popcount(__less_mask.value());
    if (__nless >= __rank)
    {
      const auto __last_to_remove = ::cuda::bit_fns(__less_mask.value(), __nless - __rank);
      __warp_mask |= ::cuda::device::lane_mask{__less_mask.value() & (~0u << (__last_to_remove + 1))};
    }
    else
    {
      __warp_mask |= __less_mask;
    }

    const auto __greater_mask = __prev_warp_mask & ::cuda::device::lane_mask::all_greater();
    const auto __ngreater     = ::cuda::std::popcount(__greater_mask.value());
    if (__rank + __ngreater >= __n)
    {
      const auto __first_to_remove = ::cuda::bit_fns(__greater_mask.value(), __n - __rank - 1);
      __warp_mask |= ::cuda::device::lane_mask{__greater_mask.value() & ((1u << __first_to_remove) - 1u)};
    }
    else
    {
      __warp_mask |= __greater_mask;
    }
    return __warp_mask;
  }
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_WARP_MASK_CUH
