//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/type_traits>

#include <cuda/experimental/coop/group>

#include <nv/target>

#include "group_testing.cuh"

struct WarpsInBlockMappingResult
{
  __device__ static constexpr ::cuda::std::size_t static_group_count()
  {
    return 1;
  }

  __device__ unsigned group_count() const
  {
    return 1;
  }

  __device__ static constexpr cuda::std::size_t static_group_rank()
  {
    return 0;
  }

  __device__ unsigned group_rank() const
  {
    return 0;
  }

  __device__ static constexpr ::cuda::std::size_t static_unit_count()
  {
    return cuda::std::dynamic_extent;
  }

  __device__ static constexpr bool has_uniform_static_unit_count()
  {
    return true;
  }

  __device__ static constexpr cuda::std::size_t static_unit_count(cuda::std::size_t)
  {
    return cuda::std::dynamic_extent;
  }

  __device__ unsigned unit_count() const
  {
    return cuda::warp.count_as<unsigned>(cuda::block);
  }

  __device__ unsigned unit_rank() const
  {
    return cuda::warp.rank_as<unsigned>(cuda::block);
  }

  __device__ cuda::device::lane_mask lane_mask() const
  {
    return cuda::device::lane_mask::all();
  }

  __device__ bool is_valid() const
  {
    return true;
  }

  __device__ static constexpr bool is_always_exhaustive()
  {
    return true;
  }

  __device__ static constexpr bool is_always_contiguous()
  {
    return true;
  }
};

struct MutableBarrierId
{
  __host__ __device__ int operator()(int id)
  {
    return id;
  }
};

template <bool MutableOnly, class Config>
__device__ void test_interwarp_synchronizer(Config config)
{
  const auto& hierarchy = config.hierarchy();

  // Test array input range.
  {
    // Keep these disjoint from the iota range used by other groups later in this kernel.
    const int barrier_ids[]{5, 7, 15, 6};

    using Synchronizer = decltype(cudax::coop::interwarp_synchronizer{barrier_ids});

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Synchronizer>);

    // Test constructor from range of barrier ids.
    {
      // NOLINTNEXTLINE(readability-redundant-parentheses)
      static_assert(cuda::std::is_constructible_v<Synchronizer, decltype((barrier_ids))>);
      [[maybe_unused]] const cudax::coop::interwarp_synchronizer synchronizer{barrier_ids};
    }

    // Test make_instance(...).
    {
      const cudax::coop::this_block parent_group{config};
      const WarpsInBlockMappingResult prev_mapping_result;

      const cudax::coop::group_by mapping{2};
      using View = cuda::std::conditional_t<MutableOnly, NonConstView<const int>, cuda::std::span<const int, 4>>;
      using RangeSynchronizer     = cudax::coop::interwarp_synchronizer<View>;
      using QualifiedSynchronizer = cuda::std::conditional_t<MutableOnly, RangeSynchronizer, const RangeSynchronizer>;
      QualifiedSynchronizer synchronizer{View{barrier_ids, 4}};

      const auto mapping_result = mapping.map(cuda::warp, parent_group, prev_mapping_result);
      static_assert(
        can_make_synchronizer_instance<const RangeSynchronizer,
                                       cuda::warp_level,
                                       decltype(parent_group),
                                       decltype(mapping_result)>
        == !MutableOnly);
      auto synchronizer_instance = synchronizer.make_instance(cuda::warp, parent_group, mapping_result);

      // Test that synchronizer instance has the expected barrier id.
      REQUIRE((cuda::std::bit_cast<cuda::std::array<unsigned, 2>>(synchronizer_instance)[0]
               == static_cast<unsigned>(barrier_ids[mapping_result.group_rank()])));

      using SynchronizerInstance = decltype(synchronizer_instance);

      static_assert(cuda::is_trivially_copyable_v<SynchronizerInstance>);
      static_assert(cuda::std::is_trivially_destructible_v<SynchronizerInstance>);

      using ArrivalToken = typename SynchronizerInstance::arrival_token;

      // Test arrival_token type.
      static_assert(cuda::std::is_move_constructible_v<ArrivalToken>);
      static_assert(cuda::std::is_move_assignable_v<ArrivalToken>);
      static_assert(cuda::std::is_destructible_v<ArrivalToken>);

      // Test do_sync(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(hierarchy))>);
      synchronizer_instance.do_sync(hierarchy);

      // Test do_arrive(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive(hierarchy))>);
      auto arrival_token = synchronizer_instance.do_arrive(hierarchy);

      // Test do_wait(...).
      static_assert(
        // NOLINTNEXTLINE(cccl-simplify-type-traits)
        cuda::std::is_same_v<void, decltype(synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token)))>);
      synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token));

      // Test do_sync_aligned(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(hierarchy))>);
      synchronizer_instance.do_sync_aligned(hierarchy);

      // Test do_arrive_aligned(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive_aligned(hierarchy))>);
      auto arrival_token_a = synchronizer_instance.do_arrive_aligned(hierarchy);

      // Test do_wait_aligned(...).
      static_assert(
        cuda::std:: // NOLINT(cccl-simplify-type-traits)
        is_same_v<void, decltype(synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a)))>);
      synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a));

      // Test deinit(...);
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
      synchronizer_instance.deinit(mapping_result, hierarchy);
    }
  }

  // Test infinite input range.
  {
    const auto barrier_ids = cuda::std::views::iota(1);

    using Synchronizer = decltype(cudax::coop::interwarp_synchronizer{barrier_ids});

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Synchronizer>);

    // Test constructor from range of barrier ids.
    {
      static_assert(cuda::std::is_constructible_v<Synchronizer, decltype(barrier_ids)>);
      [[maybe_unused]] const cudax::coop::interwarp_synchronizer synchronizer{barrier_ids};
    }

    // Test make_instance(...).
    {
      const cudax::coop::this_block parent_group{config};
      const WarpsInBlockMappingResult prev_mapping_result;

      const cudax::coop::group_by mapping{2};
      auto ids = [&] {
        if constexpr (MutableOnly)
        {
          return cuda::std::views::transform(barrier_ids, MutableBarrierId{});
        }
        else
        {
          return barrier_ids;
        }
      }();
      using RangeSynchronizer     = cudax::coop::interwarp_synchronizer<decltype(ids)>;
      using QualifiedSynchronizer = cuda::std::conditional_t<MutableOnly, RangeSynchronizer, const RangeSynchronizer>;
      QualifiedSynchronizer synchronizer{cuda::std::move(ids)};

      const auto mapping_result = mapping.map(cuda::warp, parent_group, prev_mapping_result);
      static_assert(
        can_make_synchronizer_instance<const RangeSynchronizer,
                                       cuda::warp_level,
                                       decltype(parent_group),
                                       decltype(mapping_result)>
        == !MutableOnly);
      auto synchronizer_instance = synchronizer.make_instance(cuda::warp, parent_group, mapping_result);

      // Test that synchronizer instance has the expected barrier id.
      REQUIRE((cuda::std::bit_cast<cuda::std::array<unsigned, 2>>(synchronizer_instance)[0]
               == mapping_result.group_rank() + 1));

      using SynchronizerInstance = decltype(synchronizer_instance);

      static_assert(cuda::is_trivially_copyable_v<SynchronizerInstance>);
      static_assert(cuda::std::is_trivially_destructible_v<SynchronizerInstance>);

      using ArrivalToken = typename SynchronizerInstance::arrival_token;

      // Test arrival_token type.
      static_assert(cuda::std::is_move_constructible_v<ArrivalToken>);
      static_assert(cuda::std::is_move_assignable_v<ArrivalToken>);
      static_assert(cuda::std::is_destructible_v<ArrivalToken>);

      // Test do_sync(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(hierarchy))>);
      synchronizer_instance.do_sync(hierarchy);

      // Test do_arrive(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive(hierarchy))>);
      auto arrival_token = synchronizer_instance.do_arrive(hierarchy);

      // Test do_wait(...).
      static_assert(
        // NOLINTNEXTLINE(cccl-simplify-type-traits)
        cuda::std::is_same_v<void, decltype(synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token)))>);
      synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token));

      // Test do_sync_aligned(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(hierarchy))>);
      synchronizer_instance.do_sync_aligned(hierarchy);

      // Test do_arrive_aligned(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive_aligned(hierarchy))>);
      auto arrival_token_a = synchronizer_instance.do_arrive_aligned(hierarchy);

      // Test do_wait_aligned(...).
      static_assert(
        cuda::std:: // NOLINT(cccl-simplify-type-traits)
        is_same_v<void, decltype(synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a)))>);
      synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a));

      // Test deinit(...);
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
      synchronizer_instance.deinit(mapping_result, hierarchy);
    }
  }
}

template <class StaticIds, class View>
__device__ void check_interwarp_instance(View view)
{
  constexpr unsigned barrier_ids[]{8, 14, 10};
  constexpr auto count = cuda::gpu_thread.static_count(view);
  constexpr auto rank  = view.static_rank(cudax::coop::parent_group);
  using Instance       = typename View::__synchronizer_instance_type;

  if constexpr (!cuda::std::is_void_v<StaticIds> && rank != cuda::std::dynamic_extent)
  {
    using Expected = cudax::coop::__interwarp_synchronizer_instance<count, StaticIds, barrier_ids[rank]>;
    static_assert(cuda::std::is_same_v<Instance, Expected>);
    static_assert(cuda::std::is_empty_v<Instance> == (count != cuda::std::dynamic_extent));
  }
  else
  {
    using Expected = cudax::coop::__interwarp_synchronizer_instance<count, StaticIds>;
    static_assert(cuda::std::is_same_v<Instance, Expected>);
    CHECK(cuda::std::bit_cast<unsigned>(view.__synchronizer_instance())
          == barrier_ids[view.rank(cudax::coop::parent_group)]);
  }
}

template <class StaticIds, class Group, class Policy>
__device__ void test_interwarp_dispatch(const Group& group, Policy policy)
{
  __shared__ unsigned values[256];
  const auto thread = cuda::gpu_thread.rank_as<unsigned>(cuda::block);
  values[thread]    = thread;

  group.dispatch(policy, [&](auto view) {
    constexpr auto count = cuda::gpu_thread.static_count(view);
    static_assert(count == 64 || count == 128);
    check_interwarp_instance<StaticIds>(view);
    const cudax::coop::group_view threads{cuda::gpu_thread, view};
    check_interwarp_instance<StaticIds>(threads);
    const auto specialized = group.__synchronizer_instance().specialize(threads.__mapping_result());
    static_assert(
      cuda::std::is_same_v<decltype(specialized), const typename decltype(threads)::__synchronizer_instance_type>);

    // Exchange values with the opposite end of this group across a warp boundary.
    specialized.do_wait(threads.hierarchy(), group.arrive());
    const auto rank = cuda::gpu_thread.rank(view);
    const auto peer = thread - rank + (count - 1 - rank);
    CHECK(values[peer] == peer);
    group.wait(threads.arrive());
  });
}

template <class StaticIds, class Config, class Synchronizer>
__device__ void test_specialized_interwarp_synchronizer(const Config& config, const Synchronizer& synchronizer)
{
  const cudax::coop::this_block block{config};
  const cudax::coop::generic_group group{
    cuda::warp, block, cudax::coop::group_as{cuda::std::integer_sequence<int, 2, 4, 2>{}}, synchronizer};

  test_interwarp_dispatch<StaticIds>(group, cudax::coop::by_unit_count);
  test_interwarp_dispatch<StaticIds>(group, cudax::coop::by_group_rank);
  const cudax::coop::group_view threads{cuda::gpu_thread, group};
  static_assert(
    cuda::std::is_same_v<typename decltype(threads)::__mapping_result_type::unit_level, cuda::thread_level>);
  test_interwarp_dispatch<StaticIds>(threads, cudax::coop::by_unit_count);
  test_interwarp_dispatch<StaticIds>(threads, cudax::coop::by_group_rank);

  group.dispatch(cudax::coop::by_unit_count, [&](auto parent) {
    // Rank dispatch on the thread view keeps the count already specialized in warps.
    const cudax::coop::group_view parent_threads{cuda::gpu_thread, parent};
    parent_threads.dispatch(cudax::coop::by_group_rank, [&](auto ranked) {
      static_assert(cuda::gpu_thread.static_count(ranked) == 32 * cuda::warp.static_count(parent));
      check_interwarp_instance<StaticIds>(ranked);
      ranked.wait(parent.arrive());
    });

    // A one-warp virtual subgroup and its thread view both share the larger parent's barrier.
    const cudax::coop::virtual_group child{cuda::warp, parent, cudax::coop::group_by{cuda::std::cw<1>}};
    const auto check_inherited = [&](auto view) {
      static_assert(cuda::gpu_thread.static_count(view) == 32);
      static_assert(cuda::std::is_same_v<typename decltype(view)::__synchronizer_instance_type,
                                         typename decltype(parent)::__synchronizer_instance_type>);
      view.wait(parent.arrive());
    };
    child.dispatch(cudax::coop::by_group_rank, check_inherited);
    cudax::coop::group_view{cuda::gpu_thread, child}.dispatch(cudax::coop::by_group_rank, check_inherited);
  });

  if constexpr (!cuda::std::is_void_v<StaticIds>)
  {
    // A static rank selects a constant barrier ID even when the participant count is supplied at runtime.
    const unsigned counts[]{2, 4, 2};
    const cudax::coop::generic_group dynamic{cuda::warp, block, cudax::coop::group_as{counts}, synchronizer};
    const auto check_dynamic_count = [&](auto view) {
      static_assert(cuda::gpu_thread.static_count(view) == cuda::std::dynamic_extent);
      check_interwarp_instance<StaticIds>(view);
      constexpr auto rank = view.static_rank(cudax::coop::parent_group);
      CHECK(cuda::std::bit_cast<unsigned>(view.__synchronizer_instance()) == 32 * counts[rank]);
      view.sync();
    };
    dynamic.dispatch(cudax::coop::by_group_rank, check_dynamic_count);
    cudax::coop::group_view{cuda::gpu_thread, dynamic}.dispatch(cudax::coop::by_group_rank, check_dynamic_count);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // Look up every native barrier ID, including both sides of the eight-ID boundary.
    using AllBarrierIds = cuda::std::index_sequence<15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0>;
    const auto rank     = cuda::gpu_thread.rank_as<unsigned>(cuda::warp) % 16;
    CHECK(cudax::coop::__interwarp_barrier_id_at(AllBarrierIds{}, rank) == 15 - rank);

    test_interwarp_synchronizer<false>(config);
    test_interwarp_synchronizer<true>(config);
    const unsigned ids[]{8, 14, 10};
    test_specialized_interwarp_synchronizer<void>(config, cudax::coop::interwarp_synchronizer{ids});
    using StaticIds = cuda::std::index_sequence<8, 14, 10>;
    test_specialized_interwarp_synchronizer<StaticIds>(
      config, cudax::coop::interwarp_synchronizer{cuda::std::integer_sequence<int, 8, 14, 10>{}});
#if _CCCL_STD_VER >= 2020
    test_specialized_interwarp_synchronizer<StaticIds>(
      config, cudax::coop::interwarp_synchronizer{cuda::std::cw<cuda::std::array{8, 14, 10}>});
#endif // _CCCL_STD_VER >= 2020
  }
};

C2H_TEST("Barrier synchronizer", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<256>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{256}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
