//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-90, force-tile
// Uses the host runtime to launch a cluster kernel and query SIMT coordinates.

#include <cuda/hierarchy>
#include <cuda/std/cassert>

#include "hierarchy_queries.h"

template <class Hierarchy>
__global__ void check_indices(Hierarchy hier)
{
  const auto block_idx   = __clusterRelativeBlockIdx();
  const auto cluster_idx = __clusterIdx();
  test_result(cuda::block.index(cuda::cluster, hier), block_idx);
  test_result(cuda::cluster.index(cuda::grid, hier), cluster_idx);
  test_result(cuda::gpu_thread.index(cuda::cluster, hier), uint3{(64 * block_idx.x) + threadIdx.x, block_idx.y, 0});
  test_result(cuda::warp.index(cuda::cluster, hier), uint3{(2 * block_idx.x) + (threadIdx.x / 32), block_idx.y, 0});

  const auto cluster_rank = (cluster_idx.y * __clusterGridDimInClusters().x) + cluster_idx.x;
  assert(cuda::cluster.rank(cuda::grid) == cluster_rank);
  assert(cuda::cluster.rank(cuda::grid, hier) == cluster_rank);
  assert(cuda::block.rank(cuda::grid, hier) == (blockIdx.y * gridDim.x) + blockIdx.x);
  assert(cuda::gpu_thread.rank(cuda::cluster, hier) == (__clusterRelativeBlockRank() * 64) + threadIdx.x);
}

void test()
{
  int device{};
  int supports_clusters{};
  assert(cudaGetDevice(&device) == cudaSuccess);
  assert(cudaDeviceGetAttribute(&supports_clusters, cudaDevAttrClusterLaunch, device) == cudaSuccess);
  if (!supports_clusters)
  {
    return;
  }

  auto hier = cuda::make_hierarchy(cuda::grid_dims<4, 4>(), cuda::cluster_dims<2>(), cuda::block_dims<64>());
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id             = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim = {2, 1, 1};

  cudaLaunchConfig_t config{};
  config.gridDim  = {8, 4, 1};
  config.blockDim = {64, 1, 1};
  config.attrs    = attributes;
  config.numAttrs = 1;
  assert(cudaLaunchKernelEx(&config, check_indices<decltype(hier)>, hier) == cudaSuccess);

#if _CCCL_CTK_AT_LEAST(12, 8)
  int major{};
  assert(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess);
  if (major >= 10)
  {
    // CUDA can choose the regular or the preferred shape independently for each cluster.
    attributes[1].id                      = cudaLaunchAttributePreferredClusterDimension;
    attributes[1].val.preferredClusterDim = {4, 2, 1};
    config.numAttrs                       = 2;
    assert(cudaLaunchKernelEx(&config, check_indices<decltype(hier)>, hier) == cudaSuccess);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 8)
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
