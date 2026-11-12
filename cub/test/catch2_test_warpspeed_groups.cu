// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/warpspeed/groups.cuh>
#include <cub/detail/warpspeed/resource/smem_phase.cuh>
#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/resource/smem_resource.cuh>
#include <cub/detail/warpspeed/resource/smem_stage.cuh>
#include <cub/detail/warpspeed/sync_handler.cuh>

#include <thrust/memory.h>

#include <cuda/hierarchy>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/coop/group>

#include <nv/target>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/vector.h>

using nonuniform_layout = cub::detail::warpspeed::group_layout<1, 3, 2, 1>;
using uniform_layout    = cub::detail::warpspeed::group_layout<1, 1, 1, 1>;

static_assert(nonuniform_layout::total_threads == 224);
static_assert(nonuniform_layout::thread_count<0>() == 32);
static_assert(nonuniform_layout::thread_count<1>() == 96);
static_assert(nonuniform_layout::thread_count<2>() == 64);
static_assert(nonuniform_layout::thread_count<3>() == 32);

inline constexpr int record_size = 10;
inline constexpr int sync_rounds = 5;

template <typename Group, typename Parent, typename Block>
__device__ void record_group(
  const Group& group,
  const Parent& parent,
  const Block& block,
  int role,
  int static_threads,
  cuda::std::int32_t* scratch,
  cuda::std::int32_t* output)
{
  const int threads = cuda::gpu_thread.count_as<int>(block);
  const int thread  = cuda::gpu_thread.rank_as<int>(block);
  const int rank    = cuda::gpu_thread.rank_as<int>(group);
  const int count   = cuda::gpu_thread.count_as<int>(group);
  const int begin   = thread - rank;
  const int offset  = (block.template rank_as<int>(cuda::grid) * threads + thread) * record_size;

  output[offset]     = role;
  output[offset + 1] = rank;
  output[offset + 2] = count;
  output[offset + 3] = cuda::warp.rank_as<int>(group);
  output[offset + 4] = cuda::warp.count_as<int>(group);
  output[offset + 5] = cuda::gpu_thread.is_root_rank(group);
  output[offset + 6] = static_threads;
  output[offset + 8] = group.template rank_as<int>(parent);
  output[offset + 9] = group.template count_as<int>(parent);

  int sum = 0;
  for (int round = 0; round < sync_rounds + role; ++round)
  {
    scratch[thread] = rank + round * threads;
    group.sync();
    sum += scratch[begin + (rank + 1) % count];
    group.sync();
  }
  output[offset + 7] = sum;
}

template <bool Refine, typename Layout>
__global__ void groups_kernel(cuda::std::int32_t* output)
{
  constexpr int threads = Layout::total_threads;
  const cuda::hierarchy hierarchy{cuda::grid_dims(gridDim.x), cuda::block_dims<threads>()};
  const cuda::experimental::coop::this_block block{hierarchy};
  const auto work_group = Layout::make_group(block);
  __shared__ cuda::std::int32_t scratch[threads];

  work_group.with_static_unit_count([&](auto group) {
    const auto role               = group.rank(block);
    constexpr auto static_threads = cuda::gpu_thread.static_count(group);
    static_assert(static_threads != cuda::std::dynamic_extent);
    if constexpr (Refine)
    {
      const auto refined             = Layout::make_static_group(group, role);
      constexpr auto refined_threads = cuda::gpu_thread.static_count(refined);
      static_assert(refined_threads == static_threads);
      record_group(refined, group, block, role, refined_threads, scratch, output);
    }
    else
    {
      record_group(group, block, block, role, static_threads, scratch, output);
    }
  });
}

template <typename Layout>
void launch_groups(bool refine, cuda::std::int32_t* output)
{
  if (refine)
  {
    groups_kernel<true, Layout><<<3, Layout::total_threads>>>(output);
  }
  else
  {
    groups_kernel<false, Layout><<<3, Layout::total_threads>>>(output);
  }
}

CUB_TEST("Warpspeed groups preserve membership and synchronization", "[warpspeed][group]", CUB_SMALL)
{
  const bool refine       = GENERATE(false, true);
  const bool uniform      = GENERATE(false, true);
  constexpr int blocks    = 3;
  const int threads       = uniform ? uniform_layout::total_threads : nonuniform_layout::total_threads;
  const int group_warps[] = {1, uniform ? 1 : 3, uniform ? 1 : 2, 1};
  c2h::device_vector<cuda::std::int32_t> output(blocks * threads * record_size);
  c2h::host_vector<cuda::std::int32_t> expected(output.size());

  for (int block = 0; block < blocks; ++block)
  {
    int begin = 0;
    for (int group = 0; group < 4; ++group)
    {
      const int count = group_warps[group] * 32;
      for (int rank = 0; rank < count; ++rank)
      {
        const int offset     = (block * threads + begin + rank) * record_size;
        expected[offset]     = group;
        expected[offset + 1] = rank;
        expected[offset + 2] = count;
        expected[offset + 3] = rank / 32;
        expected[offset + 4] = group_warps[group];
        expected[offset + 5] = rank == 0;
        expected[offset + 6] = count;
        const int rounds     = sync_rounds + group;
        expected[offset + 7] = rounds * ((rank + 1) % count) + threads * rounds * (rounds - 1) / 2;
        expected[offset + 8] = refine ? 0 : group;
        expected[offset + 9] = refine ? 1 : 4;
      }
      begin += count;
    }
  }

  auto* output_ptr = thrust::raw_pointer_cast(output.data());
  if (uniform)
  {
    launch_groups<uniform_layout>(refine, output_ptr);
  }
  else
  {
    launch_groups<nonuniform_layout>(refine, output_ptr);
  }
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(expected == output);
}

inline constexpr int pipeline_rounds  = 11;
inline constexpr int consumer_threads = nonuniform_layout::thread_count<1>();

template <int Stages, bool Refine>
__device__ void resource_pipeline(cuda::std::int32_t* output)
{
  const cuda::hierarchy hierarchy{cuda::block_dims<nonuniform_layout::total_threads>()};
  const cuda::experimental::coop::this_block block{hierarchy};
  __shared__ cuda::std::int32_t storage[Stages];
  __shared__ cuda::std::uint64_t barriers[2][Stages];
  cub::detail::warpspeed::SyncHandler sync;
  cub::detail::warpspeed::SmemResource<cuda::std::int32_t> resource{sync, storage};
  resource.addPhase(sync, barriers[0], nonuniform_layout::thread_count<0>());
  resource.addPhase(sync, barriers[1], consumer_threads);
  sync.init(block);

  const auto work_group = nonuniform_layout::make_group(block);
  work_group.with_static_unit_count([&](const auto& group) {
    const auto group_rank = group.rank(block);
    if (group_rank > 1)
    {
      return;
    }
    auto run_pipeline = [&](const auto& active_group) {
      for (int round = 0; round < pipeline_rounds; ++round)
      {
        auto stage         = resource.nextStage();
        auto [write, read] = cub::detail::warpspeed::bindPhases<2>(stage);
        if (group_rank == 0)
        {
          auto ref = write.acquireRef();
          if (cuda::gpu_thread.is_root_rank(active_group))
          {
            ref.data() = round;
          }
          active_group.sync();
          ref.expect_transaction(active_group, 0);
        }
        else
        {
          auto ref                                = read.acquireRef();
          const int rank                          = cuda::gpu_thread.rank_as<int>(active_group);
          output[round * consumer_threads + rank] = ref.data();
          active_group.sync();
        }
      }
    };
    if constexpr (Refine)
    {
      const auto refined = nonuniform_layout::make_static_group(group, group_rank);
      run_pipeline(refined);
    }
    else
    {
      run_pipeline(group);
    }
  });
}

template <int Stages, bool Refine>
__global__ void resource_pipeline_kernel(cuda::std::int32_t* output)
{
  // Resource phase waits require the SM90 mbarrier parity instructions.
  NV_IF_TARGET(NV_PROVIDES_SM_90, (resource_pipeline<Stages, Refine>(output);))
}

template <bool Refine>
void launch_resource_pipeline(int stages, cuda::std::int32_t* output)
{
  if (stages == 1)
  {
    resource_pipeline_kernel<1, Refine><<<1, nonuniform_layout::total_threads>>>(output);
  }
  else if (stages == 2)
  {
    resource_pipeline_kernel<2, Refine><<<1, nonuniform_layout::total_threads>>>(output);
  }
  else
  {
    resource_pipeline_kernel<3, Refine><<<1, nonuniform_layout::total_threads>>>(output);
  }
}

CUB_TEST("Warpspeed resources recycle stages between nonuniform groups", "[warpspeed][group]", CUB_SMALL)
{
  int device{};
  cudaDeviceProp properties{};
  REQUIRE_CUDART(cudaGetDevice(&device));
  REQUIRE_CUDART(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 9)
  {
    SKIP("Warpspeed resource phase waits require SM90 or newer");
  }

  const bool refine = GENERATE(false, true);
  const int stages  = GENERATE(1, 2, 3);
  c2h::device_vector<cuda::std::int32_t> output(pipeline_rounds * consumer_threads);
  c2h::host_vector<cuda::std::int32_t> expected(output.size());
  for (int round = 0; round < pipeline_rounds; ++round)
  {
    for (int rank = 0; rank < consumer_threads; ++rank)
    {
      expected[round * consumer_threads + rank] = round;
    }
  }

  auto* output_ptr = thrust::raw_pointer_cast(output.data());
  if (refine)
  {
    launch_resource_pipeline<true>(stages, output_ptr);
  }
  else
  {
    launch_resource_pipeline<false>(stages, output_ptr);
  }
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(expected == output);
}
