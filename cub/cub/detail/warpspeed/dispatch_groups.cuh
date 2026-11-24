// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/coop/__group/generic_group.cuh>
#include <cuda/experimental/coop/__group/group_view.cuh>
#include <cuda/experimental/coop/__group/mapping/group_as.cuh>
#include <cuda/experimental/coop/__group/synchronizer/interwarp_synchronizer.cuh>
#include <cuda/experimental/coop/__group/tags.cuh>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
template <::cuda::std::size_t... Ranks>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::index_sequence<(Ranks + 1)...>
  role_barrier_ids(::cuda::std::index_sequence<Ranks...>) noexcept
{
  // ID zero is used for synchronization of the whole block.
  return {};
}

template <class Counts>
struct warp_partition
{
  template <class Block, class Input>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const ::cuda::warp_level& warp, const Block& block, const Input& input) const
  {
    // All lanes enter dispatch together. A warp-uniform register lets the compiler branch uniformly
    // on the role and avoids reconvergence around collectives throughout the role's kernel body.
    auto rank = input.unit_rank();
    NV_IF_TARGET(NV_PROVIDES_SM_90, (rank = __reduce_min_sync(~0u, rank);))
    const Input uniform{input.group_count(), input.group_rank(), input.unit_count(), rank, input.lane_mask()};
    return ::cuda::experimental::coop::group_as<Counts>{Counts{}}.map(warp, block, uniform);
  }
};

template <class Block, ::cuda::std::size_t... WarpCounts, class Function>
_CCCL_DEVICE_API void
dispatch_groups(const Block& block, ::cuda::std::index_sequence<WarpCounts...> counts, Function&& fn)
{
  namespace coop = ::cuda::experimental::coop;
  using barrier_ids =
    decltype(_CUB::detail::warpspeed::role_barrier_ids(::cuda::std::make_index_sequence<sizeof...(WarpCounts)>{}));
  const coop::generic_group roles{
    ::cuda::warp, block, warp_partition<decltype(counts)>{}, coop::interwarp_synchronizer<barrier_ids>{barrier_ids{}}};
  roles.dispatch(coop::by_group_rank, ::cuda::std::forward<Function>(fn));
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
