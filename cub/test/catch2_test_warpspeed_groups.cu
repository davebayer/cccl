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

static_assert(nonuniform_layout::total_threads == 224);
static_assert(nonuniform_layout::thread_count<0>() == 32);
static_assert(nonuniform_layout::thread_count<1>() == 96);
static_assert(nonuniform_layout::thread_count<2>() == 64);
static_assert(nonuniform_layout::thread_count<3>() == 32);

inline constexpr int record_size = 8;
inline constexpr int sync_rounds = 5;

__global__ void nonuniform_groups_kernel(cuda::std::int32_t* output)
{
  constexpr int threads = nonuniform_layout::total_threads;
  const cuda::hierarchy hierarchy{cuda::grid_dims(gridDim.x), cuda::block_dims<threads>()};
  const cuda::experimental::coop::this_block block{hierarchy};
  const auto work_group = nonuniform_layout::make_group(block);
  __shared__ cuda::std::int32_t scratch[threads];

  work_group.with_static_unit_count([&](auto group) {
    constexpr auto static_threads = cuda::gpu_thread.static_count(group);
    static_assert(static_threads != cuda::std::dynamic_extent);

    const int thread = cuda::gpu_thread.rank_as<int>(block);
    const int rank   = cuda::gpu_thread.rank_as<int>(group);
    const int count  = cuda::gpu_thread.count_as<int>(group);
    const int begin  = thread - rank;
    const int offset = (block.rank_as<int>(cuda::grid) * threads + thread) * record_size;

    output[offset]     = group.template rank_as<int>(block);
    output[offset + 1] = rank;
    output[offset + 2] = count;
    output[offset + 3] = cuda::warp.rank_as<int>(group);
    output[offset + 4] = cuda::warp.count_as<int>(group);
    output[offset + 5] = cuda::gpu_thread.is_root_rank(group);
    output[offset + 6] = static_cast<int>(static_threads);

    int sum = 0;
    for (int round = 0; round < sync_rounds; ++round)
    {
      scratch[thread] = rank + round * threads;
      group.sync();
      sum += scratch[begin + (rank + 1) % count];
      group.sync();
    }
    output[offset + 7] = sum;
  });
}

CUB_TEST("Warpspeed groups preserve nonuniform membership and synchronization", "[warpspeed][group]", CUB_SMALL)
{
  constexpr int blocks        = 3;
  constexpr int threads       = nonuniform_layout::total_threads;
  constexpr int group_warps[] = {1, 3, 2, 1};
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
        expected[offset + 7] = sync_rounds * ((rank + 1) % count) + threads * sync_rounds * (sync_rounds - 1) / 2;
      }
      begin += count;
    }
  }

  nonuniform_groups_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(output.data()));
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(expected == output);
}

inline constexpr int pipeline_rounds  = 11;
inline constexpr int consumer_threads = nonuniform_layout::thread_count<1>();

template <int Stages>
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
    for (int round = 0; round < pipeline_rounds; ++round)
    {
      auto stage         = resource.nextStage();
      auto [write, read] = cub::detail::warpspeed::bindPhases<2>(stage);
      if (group_rank == 0)
      {
        auto ref = write.acquireRef();
        if (cuda::gpu_thread.is_root_rank(group))
        {
          ref.data() = round;
        }
        ref.expect_transaction(group, 0);
      }
      else
      {
        auto ref                                = read.acquireRef();
        const int rank                          = cuda::gpu_thread.rank_as<int>(group);
        output[round * consumer_threads + rank] = ref.data();
      }
    }
  });
}

template <int Stages>
__global__ void resource_pipeline_kernel(cuda::std::int32_t* output)
{
  // Resource phase waits require the SM90 mbarrier parity instructions.
  NV_IF_TARGET(NV_PROVIDES_SM_90, (resource_pipeline<Stages>(output);))
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

  const int stages = GENERATE(1, 2, 3);
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
  if (stages == 1)
  {
    resource_pipeline_kernel<1><<<1, nonuniform_layout::total_threads>>>(output_ptr);
  }
  else if (stages == 2)
  {
    resource_pipeline_kernel<2><<<1, nonuniform_layout::total_threads>>>(output_ptr);
  }
  else
  {
    resource_pipeline_kernel<3><<<1, nonuniform_layout::total_threads>>>(output_ptr);
  }
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(expected == output);
}
