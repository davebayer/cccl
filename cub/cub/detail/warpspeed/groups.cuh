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
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__ranges/iota_view.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/group>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// The layout describes only the static partition. Runtime membership, ranks and
// synchronization belong to the cooperative group returned by make_group().
template <int... WarpCounts>
struct group_layout
{
  static_assert(sizeof...(WarpCounts) > 0, "A layout must contain at least one group.");
  static_assert(sizeof...(WarpCounts) < 16, "Barrier zero is reserved for block synchronization.");
  static_assert(((WarpCounts > 0) && ...), "Every group must contain at least one warp.");

  static constexpr int total_threads = 32 * (0 + ... + WarpCounts);
  static_assert(total_threads <= 1024, "The layout must fit in one block.");

  template <::cuda::std::size_t Index>
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr int thread_count() noexcept
  {
    static_assert(Index < sizeof...(WarpCounts), "Group index is out of bounds.");
    constexpr int counts[]{WarpCounts...};
    return 32 * counts[Index];
  }

  template <typename Group>
  [[nodiscard]] _CCCL_DEVICE_API static auto make_static_group(const Group& group, ::cuda::std::uint32_t role)
  {
    using barrier_ids_t = ::cuda::std::ranges::iota_view<::cuda::std::uint32_t, ::cuda::std::uint32_t>;
    // Identity mapping retains this view's static size, so the native barrier count is an immediate.
    return ::cuda::experimental::coop::make_group<::cuda::experimental::coop::generic_group>(
      ::cuda::warp,
      group,
      ::cuda::experimental::coop::identity_mapping{},
      ::cuda::experimental::coop::interwarp_synchronizer<barrier_ids_t>{barrier_ids_t{role + 1, role + 2}});
  }

  template <typename Block>
  [[nodiscard]] _CCCL_DEVICE_API static auto make_group(const Block& block)
  {
    using counts_t      = ::cuda::std::index_sequence<static_cast<::cuda::std::size_t>(WarpCounts)...>;
    using barrier_ids_t = ::cuda::std::ranges::iota_view<::cuda::std::uint32_t, ::cuda::std::uint32_t>;
    // A counting range keeps dynamic ID selection in registers instead of indexing a local array.
    return ::cuda::experimental::coop::make_group<::cuda::experimental::coop::generic_group>(
      ::cuda::warp,
      block,
      ::cuda::experimental::coop::group_as<counts_t>{counts_t{}},
      ::cuda::experimental::coop::interwarp_synchronizer<barrier_ids_t>{
        barrier_ids_t{1u, static_cast<::cuda::std::uint32_t>(sizeof...(WarpCounts) + 1)}});
  }
};
} // namespace detail::warpspeed

CUB_NAMESPACE_END
