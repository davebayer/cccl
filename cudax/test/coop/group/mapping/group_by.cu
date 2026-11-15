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
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
template <bool Exhaustive, class Parent, class Prev>
__device__ void check_singleton_composition(const Parent& parent, const Prev& prev, unsigned mask)
{
  const cudax::coop::group_by mapping{cuda::std::cw<1>};
  const auto result = [&] {
    if constexpr (Exhaustive)
    {
      return mapping.map(cuda::gpu_thread, parent, prev);
    }
    else
    {
      return mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);
    }
  }();
  static_assert(decltype(result)::static_unit_count() == 1);
  static_assert(decltype(result)::is_always_exhaustive() == Exhaustive);

  const auto lane     = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  const bool is_valid = (mask & (1u << lane)) != 0;
  CHECK(result.is_valid() == is_valid);
  if (is_valid)
  {
    CHECK(result.group_count() == cuda::std::popcount(mask));
    CHECK(result.group_rank() == cuda::std::popcount(mask & ((1u << lane) - 1)));
    CHECK(result.unit_count() == 1);
    CHECK(result.unit_rank() == 0);
    CHECK(result.lane_mask() == cuda::device::lane_mask{1u << lane});
  }
}

template <class Config>
__device__ void test_singleton_composition(const Config& config)
{
  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;
  const auto tiled = cudax::coop::group_by{4u}.map(cuda::gpu_thread, parent, prev);
  check_singleton_composition<true>(parent, tiled, ~0u);

  // Empty logical groups do not add singleton leaders.
  const unsigned counts[]{1, 0, 7, 24};
  const auto nonuniform = cudax::coop::group_as{counts}.map(cuda::gpu_thread, parent, prev);
  check_singleton_composition<true>(parent, nonuniform, ~0u);

  const auto partial = cudax::coop::group_by{3u}.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);
  check_singleton_composition<false>(parent, partial, 0x3fffffffu);
}

template <unsigned Mask, class Config>
__device__ void test_sparse_singleton_composition(const Config& config)
{
  constexpr auto count = static_cast<unsigned>(cuda::std::popcount(Mask));
  const auto lane      = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  if (Mask & (1u << lane))
  {
    const cudax::coop::coalesced_group parent{config};
    const auto& prev = parent.__mapping_result();
    const auto tiled = cudax::coop::group_by{count > 1 ? 2u : 1u}.map(cuda::gpu_thread, parent, prev);
    check_singleton_composition<true>(parent, tiled, Mask);

    const auto partial = cudax::coop::group_by{3u}.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);
    unsigned mask      = 0;
    unsigned rank      = 0;
    for (unsigned i = 0; i < 32; ++i)
    {
      if (Mask & (1u << i))
      {
        if (rank++ < count / 3 * 3)
        {
          mask |= 1u << i;
        }
      }
    }
    check_singleton_composition<false>(parent, partial, mask);
  }
}

template <cuda::std::size_t N, class Config>
__device__ void test_group_by(Config config)
{
  // Test static N.
  {
    using Mapping = cudax::coop::group_by<N>;

    // Test the mapping is empty.
    static_assert(cuda::std::is_empty_v<Mapping>);

    // Test default constructor is deleted.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible from uint32_t.
    static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::uint32_t>);

    // Test the mapping is constructible & deductible from constant_wrapper<N>.
    {
      static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::constant_wrapper<N>>);

      // NOLINTNEXTLINE(misc-const-correctness)
      cudax::coop::group_by mapping{cuda::std::cw<N>};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    }

    // Test the mapping is not constructible from non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&>);

    // Test the mapping is not constructible from non_exhaustive_t and uint32_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::uint32_t>);

    // Test the mapping is not constructible from non_exhaustive_t and constant_wrapper<N>.
    static_assert(
      !cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::constant_wrapper<N>>);

    // Test map(...).
    {
      const cudax::coop::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cuda::gpu_thread, parent_group, prev_mapping_result))>);

      const cudax::coop::group_by mapping{cuda::std::cw<N>};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 32 / N);
      static_assert(Result::static_group_rank() == (32 / N == 1 ? 0 : cuda::std::dynamic_extent));
      CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
      CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

      static_assert(Result::static_unit_count() == N);
      CHECK(result.unit_count() == N);
      CHECK(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp) % N);

      const auto lane_mask_ref = ((N < 32) ? ((1u << N) - 1) : ~0u) << ((cuda::gpu_thread.rank(cuda::warp) / N) * N);
      CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});

      CHECK(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());
    }
  }

  // Test dynamic N.
  {
    using Mapping = cudax::coop::group_by<cuda::std::dynamic_extent>;

    // Test default constructor is deleted.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible & deducible from uint32_t.
    {
      static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::uint32_t>);

      // NOLINTNEXTLINE(misc-const-correctness)
      cudax::coop::group_by mapping{static_cast<cuda::std::uint32_t>(N)};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    }

    // Test the mapping is constructible from constant_wrapper<N>.
    static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::constant_wrapper<N>>);

    // Test the mapping is not constructible from non_exhaustive_t and uint32_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::uint32_t>);

    // Test the mapping is not constructible from non_exhaustive_t and constant_wrapper<N>.
    static_assert(
      !cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::constant_wrapper<N>>);

    // Test map(...).
    {
      const cudax::coop::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cuda::gpu_thread, parent_group, prev_mapping_result))>);

      const cudax::coop::group_by mapping{static_cast<cuda::std::uint32_t>(N)};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
      static_assert(Result::static_group_rank() == cuda::std::dynamic_extent);
      CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
      CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

      static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
      CHECK(result.unit_count() == N);
      CHECK(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp) % N);

      const auto lane_mask_ref = ((N < 32) ? ((1u << N) - 1) : ~0u) << ((cuda::gpu_thread.rank(cuda::warp) / N) * N);
      CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});

      CHECK(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());
    }
  }
}

template <cuda::std::size_t N, class Config>
__device__ void test_group_by_non_exhaustive(Config config)
{
  // Test static N.
  {
    using Mapping = cudax::coop::group_by<N>;

    // Test the mapping is empty.
    static_assert(cuda::std::is_empty_v<Mapping>);

    // Test default constructor is deleted.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible from uint32_t.
    static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::uint32_t>);

    // Test the mapping is constructible & deductible from constant_wrapper<N>.
    {
      static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::constant_wrapper<N>>);

      // NOLINTNEXTLINE(misc-const-correctness)
      cudax::coop::group_by mapping{cuda::std::cw<N>};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    }

    // Test the mapping is not constructible from non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&>);

    // Test the mapping is not constructible from non_exhaustive_t and uint32_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::uint32_t>);

    // Test the mapping is not constructible from non_exhaustive_t and constant_wrapper<N>.
    static_assert(
      !cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::constant_wrapper<N>>);

    // Test map(...).
    {
      const cudax::coop::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result))>);

      const cudax::coop::group_by mapping{cuda::std::cw<N>};
      auto result  = mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 32 / N);
      static_assert(Result::static_group_rank() == (32 / N == 1 ? 0 : cuda::std::dynamic_extent));
      static_assert(Result::static_unit_count() == N);
      static_assert(Result::is_always_exhaustive() == (32 % N == 0));
      static_assert(Result::is_always_contiguous());

      const auto is_valid_ref = cuda::gpu_thread.rank(cuda::warp) < (cuda::gpu_thread.count(cuda::warp) / N) * N;
      CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
        CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

        CHECK(result.unit_count() == N);
        CHECK(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp) % N);

        const auto lane_mask_ref = ((N < 32) ? ((1u << N) - 1) : ~0u) << ((cuda::gpu_thread.rank(cuda::warp) / N) * N);
        CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});
      }
    }
  }

  // Test dynamic N.
  {
    using Mapping = cudax::coop::group_by<cuda::std::dynamic_extent>;

    // Test default constructor is deleted.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible & deducible from uint32_t.
    {
      static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::uint32_t>);

      // NOLINTNEXTLINE(misc-const-correctness)
      cudax::coop::group_by mapping{static_cast<cuda::std::uint32_t>(N)};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    }

    // Test the mapping is constructible from constant_wrapper<N>.
    static_assert(cuda::std::is_constructible_v<Mapping, cuda::std::constant_wrapper<N>>);

    // Test the mapping is not constructible from non_exhaustive_t and uint32_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::uint32_t>);

    // Test the mapping is not constructible from non_exhaustive_t and constant_wrapper<N>.
    static_assert(
      !cuda::std::is_constructible_v<Mapping, const cudax::coop::non_exhaustive_t&, cuda::std::constant_wrapper<N>>);

    // Test map(...).
    {
      const cudax::coop::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result))>);

      const cudax::coop::group_by mapping{static_cast<cuda::std::uint32_t>(N)};
      auto result  = mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
      static_assert(Result::static_group_rank() == cuda::std::dynamic_extent);
      static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
      static_assert(!Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());

      const auto is_valid_ref = cuda::gpu_thread.rank(cuda::warp) < (cuda::gpu_thread.count(cuda::warp) / N) * N;
      CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
        CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

        CHECK(result.unit_count() == N);
        CHECK(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp) % N);

        const auto lane_mask_ref = ((N < 32) ? ((1u << N) - 1) : ~0u) << ((cuda::gpu_thread.rank(cuda::warp) / N) * N);
        CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});
      }
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_singleton_composition(config);
    test_sparse_singleton_composition<0x80000000u>(config);
    test_sparse_singleton_composition<0x80000001u>(config);
    test_sparse_singleton_composition<0x84218421u>(config);
    test_sparse_singleton_composition<0x55555555u>(config);

    test_group_by<1>(config);
    test_group_by<2>(config);
    test_group_by<4>(config);
    test_group_by<16>(config);
    test_group_by<32>(config);

    test_group_by_non_exhaustive<1>(config);
    test_group_by_non_exhaustive<2>(config);
    test_group_by_non_exhaustive<3>(config);
    test_group_by_non_exhaustive<4>(config);
    test_group_by_non_exhaustive<14>(config);
    test_group_by_non_exhaustive<16>(config);
    test_group_by_non_exhaustive<30>(config);
    test_group_by_non_exhaustive<32>(config);
  }
};
} // namespace

C2H_TEST("Group-by mapping", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
