// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS) && !defined(_CUB_WARPSPEED_USE_CUDAX_GROUPS)
#  error "_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS requires _CUB_WARPSPEED_USE_CUDAX_GROUPS"
#endif // static groups require the primary group experiment

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/special_registers.cuh>
#include <cub/detail/warpspeed/squad/squad_desc.cuh>

#if defined(_CUB_WARPSPEED_USE_CUDAX_GROUPS)
#  include <cub/detail/warpspeed/squad/group.cuh>
#endif // _CUB_WARPSPEED_USE_CUDAX_GROUPS

#include <cuda/__ptx/instructions/elect_sync.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// Squad - device squad instance
//
// A squad is a collection of warps that work together in a warp-specialized
// kernel. A warp-specialized kernel has multiple squads that perform part of
// the computation.
//
// The Squad class is a device runtime instance of a squad. It provides
// functionality to determine the rank of the current thread or warp in the
// squad, and to sync all threads in the squad.
#if defined(_CUB_WARPSPEED_USE_CUDAX_GROUPS)
template <typename Group>
struct group_squad : SquadDesc
{
  SpecialRegisters mSpecialRegisters;
  bool mIsWarpLeader;
  bool mIsLeaderWarp;
  Group mGroup;

  _CCCL_DEVICE_API group_squad(SquadDesc descriptor, SpecialRegisters registers, Group group)
      : SquadDesc(descriptor)
      , mSpecialRegisters(registers)
      , mIsWarpLeader(::cuda::ptx::elect_sync(~0u))
      , mIsLeaderWarp(::cuda::warp.is_root_rank(group))
      , mGroup(group)
  {}

  [[nodiscard]] _CCCL_DEVICE_API int warpCount() const
  {
    return ::cuda::warp.count_as<int>(mGroup);
  }

  [[nodiscard]] _CCCL_DEVICE_API int threadCount() const
  {
    return ::cuda::gpu_thread.count_as<int>(mGroup);
  }

  [[nodiscard]] _CCCL_DEVICE_API int warpRank() const
  {
    return ::cuda::warp.rank_as<int>(mGroup);
  }

  [[nodiscard]] _CCCL_DEVICE_API int threadRank() const
  {
    return ::cuda::gpu_thread.rank_as<int>(mGroup);
  }

  [[nodiscard]] _CCCL_DEVICE_API int laneRank() const
  {
    const warp_group_t warp{mGroup.hierarchy()};
    return ::cuda::gpu_thread.rank_as<int>(warp);
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThread() const
  {
    return mIsWarpLeader && mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderWarp() const
  {
    return mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThreadOfWarp() const
  {
    return mIsWarpLeader;
  }

  _CCCL_DEVICE_API void syncThreads() const
  {
    mGroup.sync();
  }
};
#else // _CUB_WARPSPEED_USE_CUDAX_GROUPS
struct Squad : SquadDesc
{
  SpecialRegisters mSpecialRegisters;
  bool mIsWarpLeader = false;
  bool mIsLeaderWarp = false;

  _CCCL_DEVICE_API Squad(SquadDesc squadStatic, SpecialRegisters specialRegisters)
      : SquadDesc(squadStatic)
      , mSpecialRegisters(specialRegisters)
      , mIsWarpLeader(::cuda::ptx::elect_sync(~0u))
      , mIsLeaderWarp(warpRank() == 0)
  {}

  [[nodiscard]] _CCCL_DEVICE_API int warpRank() const
  {
    return static_cast<int>(mSpecialRegisters.warpIdx % this->warpCount());
  }

  [[nodiscard]] _CCCL_DEVICE_API int threadRank() const
  {
    return static_cast<int>(mSpecialRegisters.threadIdxX % this->threadCount());
  }

  [[nodiscard]] _CCCL_DEVICE_API int laneRank() const
  {
    return static_cast<int>(mSpecialRegisters.laneIdx);
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThread() const
  {
    return mIsWarpLeader && mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderWarp() const
  {
    return mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThreadOfWarp() const
  {
    return mIsWarpLeader;
  }

  _CCCL_DEVICE_API void syncThreads() const
  {
    // barrier 0 is reserved for __syncthreads(). We use barrier ids 1, ...
    const int barrierIdx = this->mSquadIdx + 1;

    __barrier_sync_count(barrierIdx, this->threadCount());
  }
};
#endif // _CUB_WARPSPEED_USE_CUDAX_GROUPS

#if defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
template <typename Group>
struct static_squad : SquadDesc
{
  using group_type = Group;
  static_assert(Group::static_unit_count() != ::cuda::std::dynamic_extent);

  SpecialRegisters mSpecialRegisters;
  bool mIsWarpLeader;
  bool mIsLeaderWarp;
  Group mGroup;

  _CCCL_DEVICE_API static_squad(SquadDesc descriptor, SpecialRegisters registers, Group group)
      : SquadDesc(descriptor)
      , mSpecialRegisters(registers)
      , mIsWarpLeader(::cuda::ptx::elect_sync(~0u))
      , mIsLeaderWarp(::cuda::warp.is_root_rank(group))
      , mGroup(group)
  {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr int warpCount() noexcept
  {
    return static_cast<int>(Group::static_unit_count());
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr int threadCount() noexcept
  {
    return 32 * warpCount();
  }

  [[nodiscard]] _CCCL_DEVICE_API int warpRank() const
  {
    return ::cuda::warp.rank_as<int>(mGroup);
  }

  [[nodiscard]] _CCCL_DEVICE_API int threadRank() const
  {
    return ::cuda::gpu_thread.rank_as<int>(mGroup);
  }

  [[nodiscard]] _CCCL_DEVICE_API int laneRank() const
  {
    const warp_group_t warp{mGroup.hierarchy()};
    return ::cuda::gpu_thread.rank_as<int>(warp);
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThread() const
  {
    return mIsWarpLeader && mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderWarp() const
  {
    return mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThreadOfWarp() const
  {
    return mIsWarpLeader;
  }

  _CCCL_DEVICE_API void syncThreads() const
  {
    mGroup.sync();
  }
};

template <::cuda::std::size_t End, ::cuda::std::size_t... Counts>
_CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t squad_static_prefix(::cuda::std::index_sequence<Counts...>)
{
  constexpr ::cuda::std::size_t values[]{Counts...};
  ::cuda::std::size_t prefix = 0;
  for (::cuda::std::size_t i = 0; i < End; ++i)
  {
    prefix += values[i];
  }
  return prefix;
}

template <::cuda::std::size_t First,
          ::cuda::std::size_t Last,
          int NumSquads,
          ::cuda::std::size_t... Counts,
          typename ParentGroup,
          typename Group,
          typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void squad_dispatch_static_leaf(
  SpecialRegisters registers,
  const SquadDesc* squads,
  const ParentGroup& parent,
  const Group& group,
  ::cuda::std::index_sequence<Counts...> counts,
  F f)
{
  // Static squads cover this one-dimensional block contiguously. Select the
  // descriptor before specializing its count so the heavy callback avoids a
  // cross-product of count and descriptor dispatches.
  if constexpr (Last - First > 1)
  {
    constexpr auto mid = First + (Last - First) / 2;
    constexpr auto mid_warp =
      _CUB::detail::warpspeed::squad_static_prefix<mid>(::cuda::std::index_sequence<Counts...>{});
    if (registers.warpIdx < mid_warp)
    {
      _CUB::detail::warpspeed::squad_dispatch_static_leaf<First, mid, NumSquads>(
        registers, squads, parent, group, counts, f);
    }
    else
    {
      _CUB::detail::warpspeed::squad_dispatch_static_leaf<mid, Last, NumSquads>(
        registers, squads, parent, group, counts, f);
    }
  }
  else
  {
    // group_as defines the partition; the static pack expresses this leaf's
    // equivalent count and prefix directly in its view's mapping type.
    constexpr ::cuda::std::size_t static_counts[]{Counts...};
    constexpr auto count = static_counts[First];
    constexpr auto prefix =
      _CUB::detail::warpspeed::squad_static_prefix<First>(::cuda::std::index_sequence<Counts...>{});
    const auto warp_rank = registers.warpIdx - static_cast<::cuda::std::uint32_t>(prefix);
    _CCCL_ASSERT(group.rank(parent) == First, "the selected static squad must match the group mapping");
    _CCCL_ASSERT(warp_rank == ::cuda::warp.rank(group), "the cached squad rank must match the group mapping");
    using partition_view_t = coop::group_view_t<::cuda::warp_level, Group>;
    const auto leaf_group  = partition_view_t{group}.template with_static_unit_count<count>().with_unit_rank(warp_rank);
    const ::cuda::std::uint32_t barrier_id = static_cast<::cuda::std::uint32_t>(squads[First].mSquadIdx + 1);
    using barrier_span_t                   = ::cuda::std::span<const ::cuda::std::uint32_t, 1>;
    const auto synchronized_group          = coop::make_group<coop::generic_group>(
      ::cuda::warp,
      leaf_group,
      coop::identity_mapping{},
      coop::interwarp_synchronizer<barrier_span_t>{barrier_span_t{&barrier_id, 1}});
    using synchronized_group_t = ::cuda::std::remove_cv_t<decltype(synchronized_group)>;
    using view_t               = coop::group_view_t<::cuda::warp_level, synchronized_group_t>;
    f(static_squad<view_t>{squads[First], registers, view_t{synchronized_group}});
  }
}

template <int NumSquads, ::cuda::std::size_t... Counts, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void squad_dispatch_static_impl(
  SpecialRegisters registers, const SquadDesc* squads, ::cuda::std::index_sequence<Counts...> counts, F f)
{
  static_assert(NumSquads == sizeof...(Counts));
  static_assert(((Counts > 0) && ...), "each static squad must contain at least one warp");
  [[maybe_unused]] constexpr ::cuda::std::size_t static_counts[]{Counts...};
  for (int i = 0; i < NumSquads; ++i)
  {
    _CCCL_ASSERT(static_cast<::cuda::std::size_t>(squads[i].warpCount()) == static_counts[i],
                 "static squad counts must match the descriptors");
  }
  constexpr auto total_warps = (Counts + ...);
  static_assert(total_warps <= 32, "a static squad partition must fit within one block");
  _CCCL_ASSERT(blockDim.x == total_warps * 32 && blockDim.y == 1 && blockDim.z == 1,
               "static squad partition must cover a one-dimensional block");
  const squad_block_t block{squad_hierarchy_t{::cuda::block_dims(blockDim.x)}};

  // Preserve the cached warp-uniform rank. Recomputing threadIdx.x / 32 through
  // a thread-unit block group loses compiler uniformity, causing divergent
  // squad dispatch and reconvergence around the collective operations.
  using warp_view_t = coop::group_view_t<::cuda::warp_level, squad_block_t>;
  const auto parent =
    warp_view_t{::cuda::warp, block}.template with_static_unit_count<total_warps>().with_unit_rank(registers.warpIdx);
  const auto group = coop::make_group<coop::virtual_group>(
    ::cuda::warp, parent, coop::group_as<::cuda::std::index_sequence<Counts...>>{counts});
  _CUB::detail::warpspeed::squad_dispatch_static_leaf<0, NumSquads, NumSquads>(
    registers, squads, parent, group, counts, f);
}

template <int NumSquads, ::cuda::std::size_t... Counts, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void squad_dispatch_static(
  SpecialRegisters registers, const SquadDesc (&squads)[NumSquads], ::cuda::std::index_sequence<Counts...> counts, F f)
{
  _CUB::detail::warpspeed::squad_dispatch_static_impl<NumSquads>(registers, squads, counts, f);
}

template <::cuda::std::size_t NumSquads, ::cuda::std::size_t... Counts, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void squad_dispatch_static(
  SpecialRegisters registers,
  ::cuda::std::array<SquadDesc, NumSquads> squads,
  ::cuda::std::index_sequence<Counts...> counts,
  F f)
{
  _CUB::detail::warpspeed::squad_dispatch_static_impl<static_cast<int>(NumSquads)>(registers, squads.data(), counts, f);
}
#endif // defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)

#if defined(_CUB_WARPSPEED_USE_CUDAX_GROUPS)
template <int NumSquads, typename F, typename Group>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
squad_dispatch_group(SpecialRegisters sr, const SquadDesc* squads, F f, Group group, int warpIdxStart)
{
  static_assert(NumSquads > 0);
  if constexpr (NumSquads == 1)
  {
    const SquadDesc squad = squads[0];
    if (static_cast<unsigned>(warpIdxStart) <= sr.warpIdx
        && sr.warpIdx < static_cast<unsigned>(warpIdxStart + squad.warpCount()))
    {
      f(group_squad<Group>{squad, sr, group});
    }
  }
  else
  {
    constexpr int mid   = NumSquads / 2;
    int warpIdxStartMid = warpIdxStart;
    for (int gi = 0; gi < mid; ++gi)
    {
      warpIdxStartMid += squads[gi].warpCount();
    }
    if (sr.warpIdx < static_cast<unsigned>(warpIdxStartMid))
    {
      SquadDesc squadsLeft[mid];
      for (int gi = 0; gi < mid; ++gi)
      {
        squadsLeft[gi] = squads[gi];
      }
      _CUB::detail::warpspeed::squad_dispatch_group<mid>(sr, squadsLeft, f, group, warpIdxStart);
    }
    else
    {
      SquadDesc squadsRight[NumSquads - mid]{};
      for (int gi = 0; gi < NumSquads - mid; ++gi)
      {
        squadsRight[gi] = squads[mid + gi];
      }
      _CUB::detail::warpspeed::squad_dispatch_group<NumSquads - mid>(sr, squadsRight, f, group, warpIdxStartMid);
    }
  }
}
#endif // _CUB_WARPSPEED_USE_CUDAX_GROUPS
// squadDispatch
//
// squadDispatch is used at the start of the kernel. It takes an array of squad
// descriptors and determines which squad the current thread belongs to. The
// lambda `f: (Squad) -> void` is called with the squad currently active on this
// thread.
//
// Typically, the user will call the kernel body with the active squad.
//
// Implementation notes:
//
// Dispatch to squad based on warp index using a binary search. This balances
// the number of BRA instructions per squad and avoids NVVM inserting BRX
// instructions. BRX instructions require a jump table that is loaded from GCC,
// which incurs latency.
//
// The benefit of this function for fastScan is that adding a new squad doesn't
// require code changes in the dispatch. For low-latency inference, I hope that
// the avoidance of linear search and BRX instructions translates into latency
// reductions.
//
template <int NumSquads, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
squad_dispatch_impl(SpecialRegisters sr, const SquadDesc* squads, F f, int warpIdxStart)
{
  static_assert(NumSquads > 0);
#if defined(_CUB_WARPSPEED_USE_CUDAX_GROUPS)
  // Prefix and suffix groups preserve dispatches over a subrange of the block.
  // Empty groups contain no threads and therefore never use their barrier id.
  ::cuda::std::uint32_t warpCounts[NumSquads + 2]{};
  ::cuda::std::uint32_t barrierIds[NumSquads + 2]{};
  warpCounts[0]  = static_cast<::cuda::std::uint32_t>(warpIdxStart);
  int warpIdxEnd = warpIdxStart;
  for (int gi = 0; gi < NumSquads; ++gi)
  {
    warpCounts[gi + 1] = static_cast<::cuda::std::uint32_t>(squads[gi].warpCount());
    barrierIds[gi + 1] = static_cast<::cuda::std::uint32_t>(squads[gi].mSquadIdx + 1);
    warpIdxEnd += squads[gi].warpCount();
  }
  warpCounts[NumSquads + 1] = blockDim.x / 32 - static_cast<::cuda::std::uint32_t>(warpIdxEnd);
  const squad_block_t block{squad_hierarchy_t{::cuda::block_dims(blockDim.x)}};
  const auto group = coop::make_group<coop::generic_group>(
    ::cuda::warp,
    block,
    coop::group_as<squad_span_t>{squad_span_t{warpCounts}},
    coop::interwarp_synchronizer<squad_span_t>{squad_span_t{barrierIds}});
  using group_t = ::cuda::std::remove_cv_t<decltype(group)>;
  using view_t  = coop::group_view_t<::cuda::warp_level, group_t>;
  _CUB::detail::warpspeed::squad_dispatch_group<NumSquads>(sr, squads, f, view_t{group}, warpIdxStart);
#else // _CUB_WARPSPEED_USE_CUDAX_GROUPS
  if (NumSquads == 1)
  {
    // Leaf
    SquadDesc squad = squads[0];

    if (static_cast<unsigned>(warpIdxStart) <= sr.warpIdx
        && sr.warpIdx < static_cast<unsigned>(warpIdxStart + squad.warpCount()))
    {
      f(Squad(squad, sr));
    }
  }
  else
  {
    constexpr int mid = NumSquads / 2;
    // Left
    int warpIdxStartMid = warpIdxStart;
    for (int gi = 0; gi < mid; ++gi)
    {
      warpIdxStartMid += squads[gi].warpCount();
    }
    if (sr.warpIdx < static_cast<unsigned>(warpIdxStartMid))
    {
      if constexpr (0 < mid)
      {
        SquadDesc squadsLeft[mid];
        for (int gi = 0; gi < mid; ++gi)
        {
          squadsLeft[gi] = squads[gi];
        }
        _CUB::detail::warpspeed::squad_dispatch_impl<mid>(sr, squadsLeft, f, warpIdxStart);
      }
    }
    else
    {
      SquadDesc squadsRight[NumSquads - mid]{};
      for (int gi = 0; gi < NumSquads - mid; ++gi)
      {
        squadsRight[gi] = squads[mid + gi];
      }
      _CUB::detail::warpspeed::squad_dispatch_impl<NumSquads - mid>(sr, squadsRight, f, warpIdxStartMid);
    }
  }
#endif // _CUB_WARPSPEED_USE_CUDAX_GROUPS
}

template <int NumSquads, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
squadDispatch(SpecialRegisters sr, const SquadDesc (&squads)[NumSquads], F f, int warpIdxStart = 0)
{
  _CUB::detail::warpspeed::squad_dispatch_impl<NumSquads>(sr, squads, f, warpIdxStart);
}

template <::cuda::std::size_t NumSquads, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
squadDispatch(SpecialRegisters sr, ::cuda::std::array<SquadDesc, NumSquads> squads, F f, int warpIdxStart = 0)
{
  _CUB::detail::warpspeed::squad_dispatch_impl<static_cast<int>(NumSquads)>(sr, squads.data(), f, warpIdxStart);
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
