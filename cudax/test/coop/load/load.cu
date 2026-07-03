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
#include <cuda/std/ranges>
#include <cuda/stream>
#include <cuda/type_traits>

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

#include "testing.cuh"

template <class T, class Group>
__device__ void test_group(const Group& group)
{
  // Exit all threads that are not part of the group.
  if (!cuda::gpu_thread.is_part_of(group))
  {
    return;
  }

  const auto my_rank = cuda::gpu_thread.rank_as<T>(group);

  {
    T result;

    cuda::std::ranges::iota_view range{T{0}, cuda::gpu_thread.count_as<T>(group)};
    const auto ret = cudax::coop::load_n(group, range, 1, &result);

    REQUIRE(ret == 1);
    REQUIRE(result == my_rank);
  }

  {
    T result[4];

    cuda::std::ranges::iota_view range{T{0}, cuda::gpu_thread.count_as<T>(group) * 4};
    const auto ret = cudax::coop::load_n(group, range, 4, result);

    REQUIRE(ret == 4);
    REQUIRE(result[0] == my_rank * 4);
    REQUIRE(result[1] == my_rank * 4 + 1);
    REQUIRE(result[2] == my_rank * 4 + 2);
    REQUIRE(result[3] == my_rank * 4 + 3);
  }
}

template <class T, class Config>
__device__ void test_type(const Config& config)
{
  test_group<T>(cudax::this_thread{config});
  test_group<T>(cudax::this_warp{config});
  test_group<T>(cudax::this_block{config});
  test_group<T>(cudax::this_cluster{config});
  test_group<T>(cudax::this_grid{config});

  test_group<T>(
    cudax::group{cuda::gpu_thread, cudax::this_warp{config}, cudax::group_by<4>{}, cudax::lane_synchronizer{}});
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_type<unsigned>(config);
    test_type<long long>(config);
  }
};

C2H_TEST("load/load_n", "[load][load_n]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<4>(), cuda::block_dims<128>());
  cuda::launch(stream, config, TestKernel{});

  if (cuda::device_attributes::compute_capability_major(device) >= 9)
  {
    const auto config = cuda::make_config(cuda::grid_dims<4>(), cuda::cluster_dims<2>(), cuda::block_dims<128>());
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
