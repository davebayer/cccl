// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/coop/group>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
namespace coop = ::cuda::experimental::coop;

using squad_hierarchy_t   = decltype(::cuda::make_hierarchy(::cuda::block_dims(1u)));
using squad_block_t       = coop::this_block<squad_hierarchy_t>;
using warp_group_t        = coop::this_warp<squad_hierarchy_t>;
using squad_span_t        = ::cuda::std::span<const ::cuda::std::uint32_t>;
using warp_thread_group_t = coop::group_view_t<::cuda::thread_level, warp_group_t>;

[[nodiscard]] _CCCL_DEVICE_API inline warp_group_t make_warp_group()
{
  return warp_group_t{squad_hierarchy_t{::cuda::block_dims(blockDim.x)}};
}

[[nodiscard]] _CCCL_DEVICE_API inline warp_thread_group_t make_warp_thread_group()
{
  const warp_group_t warp = _CUB::detail::warpspeed::make_warp_group();
  return warp_thread_group_t{::cuda::gpu_thread, warp};
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
