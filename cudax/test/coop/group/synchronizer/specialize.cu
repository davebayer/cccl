// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/coop/group>

#include "testing.cuh"

struct CustomLaneSynchronizer
{
  using LaneInstance        = cudax::coop::lane_synchronizer::__synchronizer_instance;
  using SpecializedInstance = const LaneInstance;

  struct Instance : LaneInstance
  {
    unsigned group_rank;

    template <cuda::std::size_t Rank>
    __device__ SpecializedInstance specialize() const
    {
      REQUIRE_DEVICE(Rank == group_rank);
      return {__lane_mask_};
    }
  };

  template <class Unit, class Parent, class Mapping>
  __device__ Instance make_instance(const Unit& unit, const Parent& parent, const Mapping& mapping) const
  {
    return {cudax::coop::lane_synchronizer{}.make_instance(unit, parent, mapping), mapping.group_rank()};
  }
};

// Split eight warps into groups of three and five. One warp in each group uses the
// named barrier directly so the remaining warps must agree on both the ID and count.
template <class Group>
__device__ void check_interwarp_barrier(const Group& group)
{
  const auto warp = cuda::warp.rank(cuda::block);
  if (warp == 0)
  {
    __barrier_sync_count(5, 3 * 32);
  }
  else if (warp == 3)
  {
    __barrier_sync_count(1, 5 * 32);
  }
  else
  {
    group.sync();
  }
}

template <class Config, class Counts, class Ids>
__device__ void test_static_barrier_ids(const Config& config, Counts counts, Ids ids)
{
  const cudax::coop::this_block block{config};
  const cudax::coop::interwarp_synchronizer synchronizer{ids};
  const cudax::coop::generic_group group{cuda::warp, block, cudax::coop::group_as{counts}, synchronizer};
  using Instance = cuda::std::remove_cvref_t<decltype(group.__synchronizer_instance())>;
  static_assert(!cuda::std::is_same_v<decltype(cuda::std::declval<const Instance&>().template specialize<0>()),
                                      decltype(cuda::std::declval<const Instance&>().template specialize<1>())>);
  check_interwarp_barrier(group);

  group.dispatch(cudax::coop::by_unit_count, [&](auto same_size) {
    static_assert(cuda::std::is_same_v<typename decltype(same_size)::__synchronizer_instance_type, Instance>);
    same_size.dispatch(cudax::coop::by_group_rank, [&](auto view) {
      constexpr auto rank = view.static_rank(cudax::coop::parent_group);
      using Specialized   = decltype(group.__synchronizer_instance().template specialize<rank>());
      static_assert(cuda::std::is_same_v<typename decltype(view)::__synchronizer_instance_type, Specialized>);
      check_interwarp_barrier(view);
    });
  });

  // A virtual child has local rank zero in both parents and keeps its parent's barrier.
  const cudax::coop::virtual_group child{cuda::warp, group, cudax::coop::identity_mapping{}};
  child.dispatch(cudax::coop::by_group_rank, [](auto view) {
    static_assert(view.static_rank(cudax::coop::parent_group) == 0);
    check_interwarp_barrier(view);
  });

  // A view of a derived virtual group also inherits the barrier selected by its parent.
  struct DerivedChild : cuda::std::remove_cv_t<decltype(child)>
  {};
  using DerivedView = decltype(cudax::coop::group_view{cuda::std::declval<const DerivedChild&>()});
  static_assert(DerivedView::__mapping_result_type::static_group_rank() == 0);
  static_assert(cuda::std::is_same_v<typename DerivedView::__synchronizer_instance_type, Instance>);

  auto token = group.arrive();
  group.dispatch(cudax::coop::by_group_rank, [&](auto view) {
    constexpr auto rank = view.static_rank(cudax::coop::parent_group);
    using Specialized   = decltype(group.__synchronizer_instance().template specialize<rank>());
    static_assert(cuda::std::is_same_v<typename decltype(view)::__synchronizer_instance_type, Specialized>);
    static_assert(cuda::std::is_same_v<typename Specialized::arrival_token, typename Instance::arrival_token>);
    if constexpr (cuda::warp.static_count(view) == cuda::std::dynamic_extent)
    {
      static_assert(sizeof(Specialized) == sizeof(unsigned));
    }
    else
    {
      static_assert(cuda::std::is_empty_v<Specialized>);
    }
    view.wait(cuda::std::move(token));

    const cudax::coop::group_view threads{cuda::gpu_thread, view};
    threads.dispatch(cudax::coop::by_group_rank, [&](auto again) {
      static_assert(cuda::std::is_same_v<typename decltype(again)::__synchronizer_instance_type, Specialized>);
      check_interwarp_barrier(again);
    });

    const cudax::coop::virtual_group specialized_child{cuda::warp, view, cudax::coop::identity_mapping{}};
    specialized_child.dispatch(cudax::coop::by_group_rank, [](auto nested) {
      check_interwarp_barrier(nested);
    });
  });
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config) const
  {
    const cudax::coop::this_warp warp{config};
    const cudax::coop::generic_group lanes{
      cuda::gpu_thread, warp, cudax::coop::group_by{cuda::std::cw<16>}, CustomLaneSynchronizer{}};
    lanes.dispatch(cudax::coop::by_unit_count, [](auto view) {
      static_assert(
        cuda::std::is_same_v<typename decltype(view)::__synchronizer_instance_type, CustomLaneSynchronizer::Instance>);
    });
    lanes.dispatch(cudax::coop::by_group_rank, [](auto view) {
      static_assert(cuda::std::is_same_v<typename decltype(view)::__synchronizer_instance_type,
                                         CustomLaneSynchronizer::LaneInstance>);
      static_assert(cuda::std::is_copy_assignable_v<decltype(view)>);
      view.sync();
    });

    const cudax::coop::this_block block{config};
    using Ids = cuda::std::integer_sequence<int, 5, 1>;
    test_static_barrier_ids(config, cuda::std::index_sequence<3, 5>{}, Ids{});
    test_static_barrier_ids(config, cuda::std::array{3, 5}, Ids{});
#if _CCCL_STD_VER >= 2020
    test_static_barrier_ids(config, cuda::std::array{3, 5}, cuda::std::cw<cuda::std::array{5, 1}>);
#endif // _CCCL_STD_VER >= 2020

    // The single selected group already has rank zero before dispatch. Other warps have no valid mapping.
    const auto partial = cudax::coop::make_group<cudax::coop::generic_group>(
      cudax::coop::non_exhaustive,
      cuda::warp,
      block,
      cudax::coop::take{cuda::std::cw<2>},
      cudax::coop::interwarp_synchronizer{cuda::std::index_sequence<7>{}});
    REQUIRE_DEVICE(partial.has_value() == (cuda::warp.rank(cuda::block) < 2));
    if (partial)
    {
      partial->sync();
    }
  }
};

C2H_TEST("Per-group synchronizer specialization", "[group]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<256>()), TestKernel{});
  stream.sync();
}
