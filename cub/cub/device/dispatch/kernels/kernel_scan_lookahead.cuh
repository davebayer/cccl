// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/iket_support.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/detail/warpspeed/allocators/smem_allocator.cuh>
#include <cub/detail/warpspeed/dispatch_groups.cuh>
#include <cub/detail/warpspeed/load_store.cuh>
#include <cub/detail/warpspeed/look_ahead.cuh>
#include <cub/detail/warpspeed/resource/smem_phase.cuh>
#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/resource/smem_resource.cuh>
#include <cub/detail/warpspeed/resource/smem_stage.cuh>
#include <cub/detail/warpspeed/sync_handler.cuh>
#include <cub/detail/warpspeed/values.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__ptx/instructions/clusterlaunchcontrol.h> // IWYU pragma: keep
#include <cuda/hierarchy>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__algorithm/invoke_one.cuh>
#include <cuda/experimental/coop/__group/this_group.cuh>

#include <nv/target>

#ifdef _CCCL_ENABLE_IKET
#  include <cub/agent/agent_scan.cuh>
#endif // defined(_CCCL_ENABLE_IKET)

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
namespace __cub_detail  = _CUB::detail;
namespace __scan_detail = _CUB::detail::scan;
namespace coop          = ::cuda::experimental::coop;

_CCCL_IKET_CREATE_PUSH_POP_RANGE(Prologue);
_CCCL_IKET_CREATE_START_END_RANGE(GroupReduce);
_CCCL_IKET_CREATE_START_END_RANGE(GroupScan);
_CCCL_IKET_CREATE_START_END_RANGE(GroupLoadAndNextIdx);
_CCCL_IKET_CREATE_START_END_RANGE(GroupStore);
// _CCCL_IKET_CREATE_PUSH_POP_RANGE(Load); // Already declared in cub/agent/agent_scan.cuh, can't declare again
_CCCL_IKET_CREATE_PUSH_POP_RANGE(NextIdx);
_CCCL_IKET_CREATE_START_END_RANGE(GroupLookahead);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(GetNextIdx);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(ReduceThreadWarp);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(ReduceGroup);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(StoreTileAggregate);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(IncludeThreadWarpAggr);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(IncludeCtaAggr);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(ThreadScan);

template <typename InputT, typename OutputT, typename AccumT>
struct scanKernelParams
{
  const InputT* ptrIn;
  OutputT* ptrOut;
  warpspeed::tile_state_t<AccumT>* ptrTileStates;
  ::cuda::std::uint32_t* atomicCounter;
  ::cuda::std::size_t numElem;
  int numStages;
};

// holds all scan kernel resources
template <typename PolicySelector, typename InputT, typename OutputT, typename AccumT>
struct ScanResources
{
  static constexpr ScanLookaheadPolicy policy = _CUB::detail::current_policy<PolicySelector>().lookahead;

  // align to at least 16 bytes (InputT/OutputT may be aligned higher) so each stage starts correctly aligned
  struct alignas((::cuda::std::max) ({::cuda::std::size_t{16}, alignof(InputT), alignof(OutputT)})) in_out_t
  {
    // the tile_size size is a multiple of the warp size, and thus for sure a multiple of 16
    static_assert(policy.tile_size() % 16 == 0, "tile_size must be multiple of 16");

    // therefore, unaligned inputs need exactly 16 bytes extra for overcopying (tail padding = 16 - head padding)
    ::cuda::std::byte inout[(policy.tile_size() * sizeof(InputT)) + 16];
  };
  static_assert(alignof(in_out_t) >= alignof(InputT));
  static_assert(alignof(in_out_t) >= alignof(OutputT));
  using thread_and_warp_aggr_t = AccumT[policy.reduce_and_scan_warps * (warp_threads + 1)];

  _CUB::detail::warpspeed::SmemResource<in_out_t> smemInOut; // will also be used to stage the output (as
                                                             // OutputT) for the bulk copy
  _CUB::detail::warpspeed::SmemResource<uint4> smemNextBlockIdx;
  _CUB::detail::warpspeed::SmemResource<AccumT> smemAggrExclusiveCta;
  _CUB::detail::warpspeed::SmemResource<thread_and_warp_aggr_t> smemThreadAndWarpAggr;
};

template <typename PolicySelector, typename InputT, typename OutputT, typename AccumT>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
allocResources(warpspeed::SyncHandler& syncHandler, warpspeed::SmemAllocator& smemAllocator, int numStages)
  -> ScanResources<PolicySelector, InputT, OutputT, AccumT>
{
  using ScanResourcesT         = ScanResources<PolicySelector, InputT, OutputT, AccumT>;
  using in_out_t               = typename ScanResourcesT::in_out_t;
  using thread_and_warp_aggr_t = typename ScanResourcesT::thread_and_warp_aggr_t;

  constexpr auto policy = _CUB::detail::current_policy<PolicySelector>().lookahead;

  const int num_aggr_exclusive_cta_stages =
    policy.lookahead_stages > 0 ? policy.lookahead_stages : (::cuda::std::max) (1, numStages + policy.lookahead_stages);

  ScanResourcesT res = {
    _CUB::detail::warpspeed::SmemResource<in_out_t>(syncHandler, smemAllocator, warpspeed::Stages{numStages}),
    _CUB::detail::warpspeed::SmemResource<uint4>(syncHandler, smemAllocator, warpspeed::Stages{numStages}),
    _CUB::detail::warpspeed::SmemResource<AccumT>(
      syncHandler, smemAllocator, warpspeed::Stages{num_aggr_exclusive_cta_stages}),
    _CUB::detail::warpspeed::SmemResource<thread_and_warp_aggr_t>(
      syncHandler, smemAllocator, warpspeed::Stages{numStages}),
  };

  _CUB::detail::scan::setup_scan_resources(
    policy,
    syncHandler,
    smemAllocator,
    res.smemInOut,
    res.smemNextBlockIdx,
    res.smemAggrExclusiveCta,
    res.smemThreadAndWarpAggr);

  return res;
}

#if __cccl_ptx_isa >= 860

// Each scheduler group issues one cancellation and accounts for the result on the phase barrier.
template <class Group>
_CCCL_DEVICE_API void request_next_tile(const Group& group, warpspeed::SmemRef<uint4>& ref)
{
  coop::invoke_one(group, [&] {
    ::cuda::ptx::clusterlaunchcontrol_try_cancel(&ref.data(), ref.ptrCurBarrierRelease());
    ref.increase_tx_count(ref.sizeBytes());
  });
}

template <class Group>
_CCCL_DEVICE_API void
request_next_tile_atomic(const Group& group, warpspeed::SmemRef<uint4>& ref, ::cuda::std::uint32_t* atomic_counter)
{
  coop::invoke_one(group, [&] {
    ref.data().x = ::atomicAdd(atomic_counter, 1u);
  });
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpReduce(const Tp input, ScanOpT& scan_op)
{
  using warp_reduce_t = WarpReduce<Tp>;
  static_assert(::cuda::std::is_same_v<typename warp_reduce_t::TempStorage, Uninitialized<NullType>>,
                "WarpReduce for a full warp must not require temporary storage");
  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpReducePartial(const Tp input, ScanOpT& scan_op, const int num_items)
{
  using warp_reduce_t = WarpReduce<Tp>;
  static_assert(::cuda::std::is_same_v<typename warp_reduce_t::TempStorage, Uninitialized<NullType>>,
                "WarpReduce for a full warp must not require temporary storage");
  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, scan_op, num_items);
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpScanExclusive(const Tp regInput, ScanOpT& scan_op)
{
  using warp_scan_t = WarpScan<Tp>;
  static_assert(::cuda::std::is_same_v<typename warp_scan_t::TempStorage, Uninitialized<NullType>>,
                "WarpScan for a full warp must not require temporary storage");
  typename warp_scan_t::TempStorage temp_storage;
  Tp result;
  warp_scan_t{temp_storage}.ExclusiveScan(regInput, result, scan_op);
  return result;
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE Tp
warpScanExclusivePartial(Tp regInput, ScanOpT& scan_op, const int num_items, bool this_lane_is_valid)
{
  // if we have an identity, just fill the out-of-bounds items with it and use the full warp scan, since it's faster
  if constexpr (cuda::has_identity_element_v<ScanOpT, Tp>)
  {
    if (!this_lane_is_valid)
    {
      regInput = cuda::identity_element<ScanOpT, Tp>();
    }
    return __scan_detail::warpScanExclusive(regInput, scan_op);
  }
  else
  {
    using warp_scan_t = WarpScan<Tp>;
    static_assert(::cuda::std::is_same_v<typename warp_scan_t::TempStorage, Uninitialized<NullType>>,
                  "WarpScan for a full warp must not require temporary storage");
    Tp result;
    typename warp_scan_t::TempStorage temp_storage;
    warp_scan_t{temp_storage}.ExclusiveScanPartial(regInput, result, scan_op, num_items);
    return result;
  }
}

template <typename ScanOpT, typename Tp, ::cuda::std::size_t ElemPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void fillWithIdentity(Tp (&regAggrInclusive)[ElemPerThread], int valid_items)
{
  // if we are in the last tile and have an identity, fill the invalid array items with it
  if constexpr (::cuda::has_identity_element_v<ScanOpT, Tp>)
  {
    for (int i = 0; i < ElemPerThread; ++i)
    {
      if (i >= valid_items)
      {
        regAggrInclusive[i] = ::cuda::identity_element<ScanOpT, Tp>();
      }
    }
  }
}

template <bool IsInclusive, bool IsLastTile, typename Tp, ::cuda::std::size_t ElemPerThread, typename ScanOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
threadScanPartial(Tp (&regAggrInclusive)[ElemPerThread], ScanOpT& scan_op, Tp prefix, bool use_prefix, int valid_items)
{
  // skip the partial scan if we have an identity
  if constexpr (IsLastTile && !::cuda::has_identity_element_v<ScanOpT, Tp>)
  {
    if constexpr (IsInclusive)
    {
      __cub_detail::ThreadScanInclusivePartial(
        regAggrInclusive, regAggrInclusive, scan_op, valid_items, prefix, use_prefix);
    }
    else
    {
      __cub_detail::ThreadScanExclusivePartial(
        regAggrInclusive, regAggrInclusive, scan_op, valid_items, prefix, use_prefix);
    }
  }
  else
  {
    if constexpr (IsInclusive)
    {
      __cub_detail::ThreadScanInclusive(regAggrInclusive, regAggrInclusive, scan_op, prefix, use_prefix);
    }
    else
    {
      __cub_detail::ThreadScanExclusive(regAggrInclusive, regAggrInclusive, scan_op, prefix, use_prefix);
    }
  }
}

// Similar to CUB agents, this closure just aggregates common constants so the device functions implementing the
// lookahead scan kernel can have lighter signatures. Each group uses an instance of this to provide its context. In
// principle, it does not hold any mutable state. But it refers to the shared scan resources (SMEM + barriers etc.).
template <typename PolicySelector,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename RealInitValueT,
          bool ForceInclusive,
          bool StableReductionOrder = false>
struct lookahead_scan_closure
{
  static constexpr ScanLookaheadPolicy policy = _CUB::detail::current_policy<PolicySelector>().lookahead;
  static constexpr int compute_threads        = policy.reduce_and_scan_warps * warp_threads;

  enum class role
  {
    reduce,
    scan,
    load_and_schedule,
    look_ahead,
    store
  };

  // If the output elements are larger than the input elements, the output of a tile does not fit into the SMEM tile
  // buffer at once and the scan group has to issue multiple bulk stores itself. Otherwise, the store group does it.
  static constexpr bool scanGroupStores = sizeof(OutputT) > sizeof(InputT);

  static constexpr int tile_size                  = policy.tile_size();
  static constexpr int lookahead_items_per_thread = policy.lookahead_items_per_thread;

  // We might try to instantiate the kernel with huge types which would lead to a small tile size. Ensure its never 0
  static constexpr int elemPerThread = policy.items_per_thread;
  static_assert(elemPerThread * compute_threads == tile_size, "Invalid tuning policy");

  // Inclusive scan if no init_value type is provided
  static constexpr bool hasInit     = !::cuda::std::is_same_v<RealInitValueT, NullType>;
  static constexpr bool isInclusive = ForceInclusive || !hasInit;

  using scan_resources_t       = ScanResources<PolicySelector, InputT, OutputT, AccumT>;
  using in_out_t               = typename scan_resources_t::in_out_t;
  using thread_and_warp_aggr_t = typename scan_resources_t::thread_and_warp_aggr_t;

  const scanKernelParams<InputT, OutputT, AccumT> params;
  mutable ScanOpT scan_op; // mutable, so we can support non-const operator()
  const RealInitValueT real_init_value;
  scan_resources_t res; // this is the only shared mutable state

  template <class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  load_next_tile_index(const Group& group, warpspeed::SmemPhase<uint4>& phaseNextBlockIdxW) const
  {
    warpspeed::SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100,
                      (__scan_detail::request_next_tile(group, refNextBlockIdxW);),
                      (__scan_detail::request_next_tile_atomic(group, refNextBlockIdxW, params.atomicCounter);));
  }

  template <class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void load_current_tile(
    const Group& group,
    warpspeed::SmemPhase<in_out_t>& phaseInOutL,
    const warpspeed::CpAsyncOobInfo<InputT>& loadInfo) const
  {
    warpspeed::SmemRef refInOutL = phaseInOutL.acquireRef();
    _CUB::detail::warpspeed::load_bulk(group, refInOutL, loadInfo);
  }

  template <class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void lookahead(
    const Group& group,
    warpspeed::SmemPhase<AccumT>& phaseAggrExclusiveCtaW,
    bool is_first_tile,
    int& idxTilePrev,
    AccumT& AggrExclusiveCtaPrev,
    int idxTile,
    int num_tiles) /*const*/ // FIXME(bgruber): this const causes a large SASS diff
  {
    const coop::this_warp warp{group.hierarchy()};
    warpspeed::SmemRef refAggrExclusiveCtaW = phaseAggrExclusiveCtaW.acquireRef();

    if (!is_first_tile)
    {
      if constexpr (StableReductionOrder)
      {
        // The stable-order version updates idxTilePrev/AggrExclusiveCtaPrev itself
        AccumT regAggrExclusiveCta =
          _CUB::detail::warpspeed::warpIncrementalLookaheadStable<lookahead_items_per_thread>(
            warp, params.ptrTileStates, idxTilePrev, AggrExclusiveCtaPrev, idxTile, scan_op, num_tiles);
        if (::cuda::gpu_thread.is_root_rank(group))
        {
          refAggrExclusiveCtaW.data() = regAggrExclusiveCta;
        }
      }
      else
      {
        AccumT regAggrExclusiveCta = _CUB::detail::warpspeed::warpIncrementalLookahead<lookahead_items_per_thread>(
          warp, params.ptrTileStates, idxTilePrev, AggrExclusiveCtaPrev, idxTile, scan_op, num_tiles);
        if (::cuda::gpu_thread.is_root_rank(group))
        {
          refAggrExclusiveCtaW.data() = regAggrExclusiveCta;
        }
        AggrExclusiveCtaPrev = regAggrExclusiveCta;
        idxTilePrev          = idxTile;
      }
    }
  }

  template <class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void reduce_tile(
    const Group& group,
    warpspeed::SmemPhase<in_out_t>& phaseInOutRW,
    warpspeed::SmemPhase<thread_and_warp_aggr_t>& phaseThreadAndWarpAggrW,
    int valid_items,
    bool is_first_tile,
    bool is_last_tile, // TODO(bgruber): should we dispatch on is_last_tile outside this function and compile it twice?
    const warpspeed::CpAsyncOobInfo<InputT>& loadInfo,
    int idxTile,
    int num_tiles) const
  {
    const coop::this_warp warp{group.hierarchy()};
    constexpr int items_per_warp = warp_threads * elemPerThread;
    const int valid_warps        = ::cuda::ceil_div(valid_items, items_per_warp);
    _CCCL_ASSERT(0 < valid_warps && valid_warps <= ::cuda::warp.count_as<int>(group), "");

    // Load tile from shared memory and reduce across thread and warp
    AccumT regThreadAggr;
    AccumT regWarpAggr;
    {
      // Acquire phaseInOutRW only in this short scope
      warpspeed::SmemRef refInOutRW = phaseInOutRW.acquireRef();
      AccumT regInput[elemPerThread];
      const auto* smem_data_start =
        reinterpret_cast<const InputT*>(&refInOutRW.data().inout[0] + loadInfo.smemStartSkipBytes);
      // in the last tile, we load some invalid elements, but don't process them later
      _CUB::detail::warpspeed::load_smem(group, regInput, smem_data_start);

      // Reduce across thread and warp
      _CCCL_IKET_RANGE_PUSH(ReduceThreadWarp);
      if (is_last_tile)
      {
        const int valid_items_this_thread =
          ::cuda::std::clamp(valid_items - ::cuda::gpu_thread.rank_as<int>(group) * elemPerThread, 0, +elemPerThread);
        const int valid_items_this_warp =
          ::cuda::std::clamp(valid_items - ::cuda::warp.rank_as<int>(group) * items_per_warp, 0, items_per_warp);
        const int valid_threads_this_warp = ::cuda::ceil_div(valid_items_this_warp, elemPerThread);
        // TODO(bgruber): for operators where we know the identity we can probably optimize this better
        regThreadAggr = __cub_detail::ThreadReducePartial(regInput, scan_op, valid_items_this_thread);
        regWarpAggr   = __scan_detail::warpReducePartial(regThreadAggr, scan_op, valid_threads_this_warp);
      }
      else
      {
        regThreadAggr = _CUB::ThreadReduce(regInput, scan_op);
        regWarpAggr   = __scan_detail::warpReduce(regThreadAggr, scan_op);
      }
      _CCCL_IKET_RANGE_POP();
    }

    warpspeed::SmemRef refThreadAndWarpAggrW = phaseThreadAndWarpAggrW.acquireRef();
    // Storing here shortens the live range of the thread aggregate across the group reduction.
    refThreadAndWarpAggrW.data()[::cuda::gpu_thread.rank_as<int>(group)] = regThreadAggr;
    if (::cuda::gpu_thread.is_root_rank(warp))
    {
      refThreadAndWarpAggrW.data()[compute_threads + ::cuda::warp.rank_as<int>(group)] = regWarpAggr;
    }
    group.sync();

    // Reduce across group
    // We need to accumulate the first element by hand because of the potential initial element and partial tiles
    _CCCL_IKET_RANGE_PUSH(ReduceGroup);
    AccumT regGroupAggr;
    if constexpr (hasInit)
    {
      if (is_first_tile)
      {
        regGroupAggr = scan_op(real_init_value, refThreadAndWarpAggrW.data()[compute_threads]);
      }
      else
      {
        regGroupAggr = refThreadAndWarpAggrW.data()[compute_threads];
      }
    }
    else
    {
      regGroupAggr = refThreadAndWarpAggrW.data()[compute_threads];
    }

    if (is_last_tile)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < policy.reduce_and_scan_warps; ++i)
      {
        if (i < valid_warps)
        {
          regGroupAggr = scan_op(regGroupAggr, refThreadAndWarpAggrW.data()[compute_threads + i]);
        }
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < policy.reduce_and_scan_warps; ++i)
      {
        regGroupAggr = scan_op(regGroupAggr, refThreadAndWarpAggrW.data()[compute_threads + i]);
      }
    }
    _CCCL_IKET_RANGE_POP();

    // Store tile aggregate for lookahead
    _CCCL_IKET_RANGE_PUSH(StoreTileAggregate);
    if (::cuda::gpu_thread.is_root_rank(group))
    {
      _CUB::detail::warpspeed::storeTileAggregate(
        params.ptrTileStates, warpspeed::scan_state::tile_aggregate, regGroupAggr, idxTile, num_tiles);
    }
    _CCCL_IKET_RANGE_POP();
  }

  template <bool IsLastTile, class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void scan_tile(
    const Group& group,
    warpspeed::SmemPhase<thread_and_warp_aggr_t>& phaseThreadAndWarpAggrR,
    warpspeed::SmemPhase<AccumT>& phaseAggrExclusiveCtaR,
    warpspeed::SmemPhase<in_out_t>& phaseInOutRW,
    bool is_first_tile,
    int valid_items,
    const warpspeed::CpAsyncOobInfo<InputT>& loadInfo,
    ::cuda::std::size_t idxTileBase) /*const*/ // FIXME(bgruber): this const causes a large SASS diff
  {
    const coop::this_warp warp{group.hierarchy()};
    // need to init these to silence nvcc warning about reading uninitialized data
    [[maybe_unused]] int valid_items_this_thread = 0;
    [[maybe_unused]] int valid_threads_this_warp = 0;
    [[maybe_unused]] int valid_warps             = 0;
    if constexpr (IsLastTile)
    {
      valid_items_this_thread =
        ::cuda::std::clamp(valid_items - (::cuda::gpu_thread.rank_as<int>(group) * elemPerThread), 0, +elemPerThread);
      valid_threads_this_warp = ::cuda::std::clamp(
        ::cuda::ceil_div(valid_items, elemPerThread) - (::cuda::warp.rank_as<int>(group) * 32), 0, 32);
      valid_warps = ::cuda::ceil_div(valid_items, elemPerThread * 32);
      _CCCL_ASSERT(0 < valid_warps && valid_warps <= ::cuda::warp.count_as<int>(group), "");
    }

    // Fill the registers with the scan identity, if there is one, before acquiring/waiting on any resources
    AccumT regAggrInclusive[elemPerThread];
    if constexpr (IsLastTile)
    {
      __scan_detail::fillWithIdentity<ScanOpT>(regAggrInclusive, valid_items_this_thread);
    }

    // Aggregate of all threads up to but not including this one
    AccumT aggrExclusive;

    // Include warp and thread aggregates of current tile
    {
      // acquire the thread and warp aggregates only for as long as we need them
      warpspeed::SmemRef refThreadAndWarpAggrR = phaseThreadAndWarpAggrR.acquireRef();
      _CCCL_IKET_RANGE_PUSH(IncludeThreadWarpAggr);
      // Add the aggregates of the preceding warps in this CTA to the cumulative aggregate. These have been calculated
      // in reduce group. We need the reduce and scan groups to be the same size to do this.

      // Include warp aggregates
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < policy.reduce_and_scan_warps; ++i)
      {
        // We want a predicated unrolled loop here.
        bool include_warp = i < ::cuda::warp.rank_as<int>(group);
        if constexpr (IsLastTile)
        {
          include_warp &= i < valid_warps;
        }
        if (include_warp)
        {
          if (i == 0)
          {
            // The first iteration initializes aggrExclusive
            aggrExclusive = refThreadAndWarpAggrR.data()[compute_threads];
          }
          else
          {
            // If loaded value belongs to previous warp, include it in aggrExclusive.
            aggrExclusive = scan_op(aggrExclusive, refThreadAndWarpAggrR.data()[compute_threads + i]);
          }
        }
      }
      // aggrExclusive contains the aggregates of previous warps.
      // It has a valid value in
      // - tile*::warp{1,2, ..}      (aggregate of previous warps)
      //
      // It is not yet initialized in
      // - tile*::warp0

      // Add the aggregates of preceding threads in this warp to the cumulative aggregate.
      // We perform an exclusive scan of:
      //
      //   {aggrT0, aggrT1, ..., aggrT30, aggrT31 }
      //
      // As a result:
      // - lane0 has undefined value
      // - lane1 has aggrT0
      // - ...
      // - lane31 has aggrT0 + ... + aggrT30
      //
      // For lane1, ..., 31, we add the result to aggrExclusive.
      //
      // If the warp contains partial data, we pass invalid elements to
      // scan_op, and aggrExclusiveIntraWarp is invalid when the inputs were
      // invalid.
      AccumT regAggrThread = refThreadAndWarpAggrR.data()[::cuda::gpu_thread.rank_as<int>(group)];
      AccumT aggrExclusiveIntraWarp;
      if constexpr (IsLastTile) // this branch would cost up to 4% BW for I8 and I16 if it were at
                                // runtime
      {
        aggrExclusiveIntraWarp = __scan_detail::warpScanExclusivePartial(
          regAggrThread,
          scan_op,
          valid_threads_this_warp,
          ::cuda::gpu_thread.rank(warp) < static_cast<uint32_t>(valid_threads_this_warp));
      }
      else
      {
        aggrExclusiveIntraWarp = __scan_detail::warpScanExclusive(regAggrThread, scan_op);
      }

      if (::cuda::warp.rank_as<int>(group) == 0)
      {
        // Warp0 does not yet have a valid value for aggrExclusive. We set it
        // here. This ensures that lane1,..,31 of tile0::warp0 have a valid
        // value for aggrExclusive.
        aggrExclusive = aggrExclusiveIntraWarp;
      }
      else
      {
        // lane0 has an undefined value for aggrExclusiveIntraWarp, so skip it
        bool includeIntraWarpAggr = ::cuda::gpu_thread.rank(warp) != 0;
        if constexpr (IsLastTile)
        {
          includeIntraWarpAggr &= ::cuda::gpu_thread.rank(warp) < static_cast<uint32_t>(valid_threads_this_warp);
        }

        if (includeIntraWarpAggr)
        {
          aggrExclusive = scan_op(aggrExclusive, aggrExclusiveIntraWarp);
        }
      }
      _CCCL_IKET_RANGE_POP();
    }
    // aggrExclusive contains the aggregates of previous warps and threads.
    //
    // - tile*::warp0::lane{1, .., 31}  (aggr of previous threads)
    // - tile*::warp{1,2, ..}           (aggr of previous warps + aggr of previous threads)
    //
    // It has an undefined value in
    // - tile*::warp0::lane0

    // Include aggregate of previous tiles
    {
      // important: we have to acquire the resource for the first tile as well to prevent a hang
      warpspeed::SmemRef refAggrExclusiveCtaR = phaseAggrExclusiveCtaR.acquireRef();
      _CCCL_IKET_RANGE_PUSH(IncludeCtaAggr);

      if (!is_first_tile)
      {
        // Add the aggregates of preceding CTAs to the cumulative aggregate.
        AccumT regAggrExclusiveCta = refAggrExclusiveCtaR.data();
        // aggrExclusive is invalid in warp_0/thread_0, so only include it in other threads/warps. Skip it, when this
        // thread has no valid items to process in the last tile.
        if (::cuda::gpu_thread.rank_as<int>(group) == 0 || (IsLastTile && valid_items_this_thread == 0))
        {
          aggrExclusive = regAggrExclusiveCta;
        }
        else
        {
          aggrExclusive = scan_op(regAggrExclusiveCta, aggrExclusive);
        }
      }
    }

    if constexpr (hasInit)
    {
      if (is_first_tile)
      {
        // The first thread cannot use scan_op because aggrExclusive holds garbage data. Also skip sumExclusive when
        // this thread has no valid items to process in the last tile.
        if (::cuda::gpu_thread.rank_as<int>(group) == 0 || (IsLastTile && valid_items_this_thread == 0))
        {
          aggrExclusive = static_cast<AccumT>(real_init_value);
        }
        else
        {
          aggrExclusive = scan_op(static_cast<AccumT>(real_init_value), aggrExclusive);
        }
      }
    }
    _CCCL_IKET_RANGE_POP();
    // aggrExclusive contains the following values:
    //
    // - tile0::warp0::lane0            (init_value)
    // - tile0::warp0::lane{1, .., 31}  (init_value + aggr of previous threads)
    // - tile0::warp{1,2, ..}           (init_value + aggr of previous warps + aggr of previous threads)
    // - tile*::warp0::lane0            (aggr of previous CTAs)
    // - tile*::warp0::lane{1, .., 31}  (aggr of previous CTAs + aggr of previous threads)
    // - tile*::warp{1,2, ..}           (aggr of previous CTAs + aggr of previous warps + aggr of previous
    // threads)
    //
    // If no init value is provided, then aggrExclusive has an undefined value in
    // - tile0::warp0::lane0

    // Scan across elements allocated to this thread

    // Acquire refInOut for remainder of scope.
    warpspeed::SmemRef refInOutRW = phaseInOutRW.acquireRef();

    // We are always loading a full tile even for the last tile, so we are loading invalid data
    _CUB::detail::warpspeed::load_smem(
      group,
      regAggrInclusive,
      reinterpret_cast<const InputT*>(&refInOutRW.data().inout[0] + loadInfo.smemStartSkipBytes));

    _CCCL_IKET_RANGE_PUSH(ThreadScan);
    // Perform inclusive scan of register array in current thread.
    // warp_0/thread_0 in the first tile when there is no initial value, we MUST NOT use aggrExclusive
    const bool use_prefix = hasInit || !is_first_tile || ::cuda::gpu_thread.rank_as<int>(group) != 0;
    __scan_detail::threadScanPartial<isInclusive, IsLastTile>(
      regAggrInclusive, scan_op, aggrExclusive, use_prefix, valid_items_this_thread);

    // Sync before storing to avoid data races on SMEM
    group.sync();
    _CCCL_IKET_RANGE_POP();

    // Store result to shared memory

    ::cuda::std::byte* smem_output_tile = refInOutRW.data().inout;
    if constexpr (sizeof(OutputT) <= sizeof(InputT))
    {
      warpspeed::CpAsyncOobInfo storeInfo =
        _CUB::detail::warpspeed::prepareCpAsyncOob(params.ptrOut + idxTileBase, valid_items);

      _CUB::detail::warpspeed::store_smem(
        group, reinterpret_cast<OutputT*>(smem_output_tile + storeInfo.smemStartSkipBytes), regAggrInclusive);
    }
    else
    {
      // otherwise, issue multiple bulk copies in chunks of the input tile size
      // TODO(bgruber): I am sure this could be implemented a lot more efficiently
      static constexpr int elem_per_chunk = static_cast<int>(policy.tile_size() * sizeof(InputT) / sizeof(OutputT));
      for (int chunk_offset = 0; chunk_offset < valid_items; chunk_offset += elem_per_chunk)
      {
        const int chunk_size = (::cuda::std::min) (valid_items - chunk_offset, elem_per_chunk);
        warpspeed::CpAsyncOobInfo storeInfo =
          _CUB::detail::warpspeed::prepareCpAsyncOob(params.ptrOut + idxTileBase + chunk_offset, chunk_size);

        // only stage elements of the current chunk to SMEM
        // storeInfo.smemStartSkipBytes < 16 and smem_output_tile contains extra 16 bytes, so we should fit
        _CCCL_ASSERT(storeInfo.smemStartSkipBytes + (elem_per_chunk * sizeof(OutputT)) <= res.smemInOut.mSizeBytes, "");
        _CUB::detail::warpspeed::store_smem_partial(
          group,
          reinterpret_cast<OutputT*>(smem_output_tile + storeInfo.smemStartSkipBytes), // different in each
                                                                                       // iteration
          regAggrInclusive,
          chunk_offset,
          chunk_offset + chunk_size);

        // We do *not* release refSmemInOut here, because we will issue a TMA
        // instruction below. Instead, we issue a group synchronization +
        // fence.proxy.async to sync the shared memory writes with the TMA store.
        group.sync();

        // Store result to global memory using TMA
        _CUB::detail::warpspeed::store_bulk_sync(group, storeInfo, smem_output_tile);

        group.sync();
      }
    }

    // Release refInOut. Will be reused by the load group for another tile in the async proxy as well, so no fence
    // needed.
  }

  template <class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void store_tile(
    const Group& group,
    warpspeed::SmemPhase<in_out_t>& phaseInOutS,
    int valid_items,
    ::cuda::std::size_t idxTileBase) const
  {
    // Wait until the scan group has written the output tile to shared memory
    warpspeed::SmemRef refInOutS = phaseInOutS.acquireRef();
    if constexpr (!scanGroupStores)
    {
      const warpspeed::CpAsyncOobInfo storeInfo =
        _CUB::detail::warpspeed::prepareCpAsyncOob(params.ptrOut + idxTileBase, valid_items);
      // Blocks until the bulk store finished reading from shared memory, then refInOutS is released for the next load
      _CUB::detail::warpspeed::store_bulk_sync(group, storeInfo, refInOutS.data().inout);
    }
  }

  // The dispatch leaf supplies a fixed role and named barrier for the entire tile loop.
  template <class Group>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void run(const Group& group, int idxTile)
  {
    constexpr auto group_role = static_cast<role>(Group::static_rank(coop::parent_group));
    const int numTiles        = static_cast<int>(::cuda::ceil_div(params.numElem, ::cuda::std::size_t(tile_size)));
    int idxTilePrev           = 0;
    AccumT AggrExclusiveCtaPrev; // only valid in the lookahead warp's lane zero

    _CCCL_IKET_RANGE_POP();

#  pragma unroll 1
    // SM90 produces bad codegen and deadlocks with a `while (true)`, so it exits via the bound check.
    // SM100 uses a explicit true condition and relies on the clusterlaunchcontrol exit below instead.
    while ([&] {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100, (return true;), (return idxTile < numTiles;));
    }())
    {
      // Get stages. When these objects go out of scope, the stage of the resource is automatically incremented.
      warpspeed::SmemStage stageNextBlockIdx      = res.smemNextBlockIdx.nextStage();
      warpspeed::SmemStage stageInOut             = res.smemInOut.nextStage();
      warpspeed::SmemStage stageThreadAndWarpAggr = res.smemThreadAndWarpAggr.nextStage();
      warpspeed::SmemStage stageAggrExclusiveCta  = res.smemAggrExclusiveCta.nextStage();

      // Split the stages into phases. Each resource goes through phases where it is writeable by a set of threads and
      // readable by a set of threads. To acquire and release a phase, we need to arrive and wait on certain barriers.
      // The selection of the barriers is handled under the hood.
      auto [phaseNextBlockIdxW, phaseNextBlockIdxR] = _CUB::detail::warpspeed::bindPhases<2>(stageNextBlockIdx);
      auto [phaseInOutL, phaseInOutRW, phaseInOutS] = _CUB::detail::warpspeed::bindPhases<3>(stageInOut);
      auto [phaseThreadAndWarpAggrW,
            phaseThreadAndWarpAggrR]                = _CUB::detail::warpspeed::bindPhases<2>(stageThreadAndWarpAggr);
      auto [phaseAggrExclusiveCtaW,
            phaseAggrExclusiveCtaR]                 = _CUB::detail::warpspeed::bindPhases<2>(stageAggrExclusiveCta);

      // We need to handle the first and the last -partial- tile differently
      const bool is_first_tile = idxTile == 0;

      const ::cuda::std::size_t idxTileBase = idxTile * ::cuda::std::size_t(tile_size);
      _CCCL_ASSERT(idxTileBase < params.numElem, "");
      const int valid_items =
        static_cast<int>((cuda::std::min) (params.numElem - idxTileBase, ::cuda::std::size_t(tile_size)));
      const bool is_last_tile = valid_items < tile_size;
      const warpspeed::CpAsyncOobInfo loadInfo =
        _CUB::detail::warpspeed::prepareCpAsyncOob(const_cast<InputT*>(params.ptrIn) + idxTileBase, valid_items);

      // Async loading and async tile index stealing are very lightweight, so one group is enough to do both

      if constexpr (group_role == role::load_and_schedule)
      {
        _CCCL_IKET_RANGE_START(GroupLoadAndNextIdx);

        _CCCL_IKET_RANGE_PUSH(Load);
        load_current_tile(group, phaseInOutL, loadInfo);
        _CCCL_IKET_RANGE_POP();

        _CCCL_IKET_RANGE_PUSH(NextIdx);
        load_next_tile_index(group, phaseNextBlockIdxW);
        _CCCL_IKET_RANGE_POP();

        _CCCL_IKET_RANGE_END(GroupLoadAndNextIdx);
      }

      _CCCL_IKET_RANGE_PUSH(GetNextIdx);
      // Get next tile index from shared memory (all groups)
      uint4 regNextBlockIdx{};
      {
        warpspeed::SmemRef refNextBlockIdxR = phaseNextBlockIdxR.acquireRef();
        regNextBlockIdx                     = refNextBlockIdxR.data();
        refNextBlockIdxR.setFenceLdsToAsyncProxy();
      }
      const bool nextIdxTileValid = [&] {
        NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100,
                          (return ::cuda::ptx::clusterlaunchcontrol_query_cancel_is_canceled(regNextBlockIdx);),
                          (return static_cast<int>(regNextBlockIdx.x) < numTiles;));
      }();
      _CCCL_IKET_RANGE_POP();

      if constexpr (group_role == role::reduce)
      {
        _CCCL_IKET_RANGE_START(GroupReduce);
        reduce_tile(
          group,
          phaseInOutRW,
          phaseThreadAndWarpAggrW,
          valid_items,
          is_first_tile,
          is_last_tile,
          loadInfo,
          idxTile,
          numTiles);
        _CCCL_IKET_RANGE_END(GroupReduce);
      }

      if constexpr (group_role == role::look_ahead)
      {
        _CCCL_IKET_RANGE_START(GroupLookahead);
        lookahead(group, phaseAggrExclusiveCtaW, is_first_tile, idxTilePrev, AggrExclusiveCtaPrev, idxTile, numTiles);
        _CCCL_IKET_RANGE_END(GroupLookahead);
      }

      if constexpr (group_role == role::scan)
      {
        static_assert(tile_size % compute_threads == 0);
        _CCCL_IKET_RANGE_START(GroupScan);
        if (is_last_tile)
        {
          scan_tile<true>(
            group,
            phaseThreadAndWarpAggrR,
            phaseAggrExclusiveCtaR,
            phaseInOutRW,
            is_first_tile,
            valid_items,
            loadInfo,
            idxTileBase);
        }
        else
        {
          scan_tile<false>(
            group,
            phaseThreadAndWarpAggrR,
            phaseAggrExclusiveCtaR,
            phaseInOutRW,
            is_first_tile,
            valid_items,
            loadInfo,
            idxTileBase);
        }
        _CCCL_IKET_RANGE_END(GroupScan);
      }

      if constexpr (group_role == role::store)
      {
        _CCCL_IKET_RANGE_START(GroupStore);
        store_tile(group, phaseInOutS, valid_items, idxTileBase);
        _CCCL_IKET_RANGE_END(GroupStore);
      }

      // All groups: Check loop condition and update next tile index
      if (!nextIdxTileValid)
      {
        break;
      }
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_100,
        (idxTile = ::cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(regNextBlockIdx);),
        (idxTile = static_cast<int>(regNextBlockIdx.x);));
    }

    // epilogue: after the load group finished, we can start ramping up the next kernel

    if constexpr (group_role == role::load_and_schedule)
    {
      _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
    }
  }
};

#endif // __cccl_ptx_isa >= 860

template <typename PolicySelector,
          bool ForceInclusive,
          typename RealInitValueT,
          bool StableReductionOrder,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_scan_lookahead_body(
  const scanKernelParams<InputT, OutputT, AccumT>& params, const ScanOpT& scan_op, const InitValueT& init_value)
{
#if __cccl_ptx_isa >= 860
  _CCCL_IKET_RANGE_PUSH(Prologue);

  static constexpr ScanLookaheadPolicy policy = _CUB::detail::current_policy<PolicySelector>().lookahead;

  const coop::this_block block{
    ::cuda::hierarchy{::cuda::grid_dims(gridDim.x), ::cuda::block_dims<__scan_detail::num_total_threads(policy)>()}};

  // Set up the shared memory resources
  auto res = [&] {
    warpspeed::SyncHandler syncHandler{};
    warpspeed::SmemAllocator smemAllocator{};
    auto r = __scan_detail::allocResources<PolicySelector, InputT, OutputT, AccumT>(
      syncHandler, smemAllocator, params.numStages);
    syncHandler.init(block);
    return r;
  }();

  // Dispatch each warp to its respective group
  using closure_t = lookahead_scan_closure<
    PolicySelector,
    InputT,
    OutputT,
    AccumT,
    ScanOpT,
    RealInitValueT,
    ForceInclusive,
    StableReductionOrder>;
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();

  int first_tile;
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100, (first_tile = block.template rank_as<int>(::cuda::grid);), ({
                      __shared__ int s_first_tile;
                      coop::invoke_one(block, [&] {
                        s_first_tile = static_cast<int>(::atomicAdd(params.atomicCounter, 1u));
                      });
                      block.sync_aligned();
                      first_tile = s_first_tile;
                    }));

  using partition = ::cuda::std::index_sequence<policy.reduce_and_scan_warps, policy.reduce_and_scan_warps, 1, 1, 1>;
  _CUB::detail::warpspeed::dispatch_groups(block, partition{}, [&](const auto& group) _CCCL_FORCEINLINE_LAMBDA {
    // Load the initial value only in roles that consume it.
    closure_t{params, scan_op, static_cast<RealInitValueT>(init_value), res}.run(group, first_tile);
  });
#endif // __cccl_ptx_isa >= 860
}

template <typename AccumT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_scan_init_lookahead_body(
  warpspeed::tile_state_t<AccumT>* tile_states,
  const int num_temp_states,
  [[maybe_unused]] ::cuda::std::uint32_t* atomic_counter)
{
  const coop::this_grid grid{::cuda::hierarchy{::cuda::grid_dims(gridDim.x), ::cuda::block_dims(blockDim.x)}};
  const int tile_id = ::cuda::gpu_thread.rank_as<int>(grid);

  // The atomic counter is only used (and thus only needs zeroing) on SM90.
  NV_IF_TARGET(NV_PROVIDES_SM_100, (), ({
                 if (tile_id == 0 && atomic_counter != nullptr)
                 {
                   *atomic_counter = 0;
                 }
               }));

  if (tile_id >= num_temp_states)
  {
    return;
  }
  // we strive to initialize the padding bits to avoid compute-sanitizer's initcheck to report reading uninitialized
  // data when reading the tile state. We use a single atomic load/store up until 16 bytes.
  // Zero initialization requires the empty state to have value zero.
  static_assert(::cuda::std::to_underlying(warpspeed::scan_state::empty) == 0);
  if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 2)
  {
    *reinterpret_cast<::cuda::std::uint16_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 4)
  {
    *reinterpret_cast<::cuda::std::uint32_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 8)
  {
    *reinterpret_cast<::cuda::std::uint64_t*>(tile_states + tile_id) = 0;
  }
  else if constexpr (sizeof(warpspeed::tile_state_t<AccumT>) == 16)
  {
    *reinterpret_cast<uint4*>(tile_states + tile_id) = {};
  }
  else
  {
    tile_states[tile_id] = warpspeed::tile_state_t<AccumT>{};
  }
}

template <typename InputT, typename OutputT, typename AccumT>
_CCCL_HOST_DEVICE_API constexpr auto smem_for_stages(const ScanLookaheadPolicy& policy, int num_stages) -> int
{
  return _CUB::detail::scan::smem_for_stages(
    policy,
    num_stages,
    static_cast<int>(sizeof(InputT)),
    static_cast<int>(alignof(InputT)),
    static_cast<int>(alignof(OutputT)),
    static_cast<int>(sizeof(AccumT)),
    static_cast<int>(alignof(AccumT)));
}
} // namespace detail::scan

CUB_NAMESPACE_END
