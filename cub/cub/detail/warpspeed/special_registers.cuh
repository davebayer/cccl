// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/make_warp_uniform.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// Commonly used special registers that we should cache in registers or uniform
// registers.
struct SpecialRegisters
{
  const ::cuda::std::uint32_t clusterCtaRank;
  const ::cuda::std::uint32_t blockIdxX;
  const ::cuda::std::uint32_t threadIdxX;
  const ::cuda::std::uint32_t warpIdx;
  const ::cuda::std::uint32_t laneIdx;
};

[[nodiscard]] _CCCL_DEVICE_API inline SpecialRegisters getSpecialRegisters()
{
  ::cuda::std::uint32_t clusterCtaRank = ::cuda::ptx::get_sreg_cluster_ctarank();
  ::cuda::std::uint32_t threadIdxX     = threadIdx.x;
#if defined(_CUB_WARPSPEED_USE_CUDAX_GROUPS)
  // group_as establishes warp uniformity when constructing the partition.
  ::cuda::std::uint32_t warpIdx = threadIdxX / 32;
#else
  ::cuda::std::uint32_t warpIdx = _CUB::detail::warpspeed::makeWarpUniform(threadIdxX / 32);
#endif
  return {clusterCtaRank, blockIdx.x, threadIdxX, warpIdx, ::cuda::ptx::get_sreg_laneid()};
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
