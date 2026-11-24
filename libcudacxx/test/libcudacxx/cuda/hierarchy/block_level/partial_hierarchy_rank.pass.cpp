//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: accessing gridDim/blockIdx is unsupported in tile code

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include <nv/target>

template <class Hierarchy>
__global__ void check_rank(Hierarchy hier)
{
  const unsigned expected = ((((blockIdx.z * gridDim.y) + blockIdx.y) * gridDim.x) + blockIdx.x);
  assert(cuda::block.rank_as<unsigned>(cuda::grid, hier) == expected);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 // Without a grid dimension, use all three dimensions of the actual launch.
                 check_rank<<<dim3{2, 3, 4}, 1>>>(cuda::hierarchy{cuda::block_dims<1>()});

                 // A grid-only hierarchy can describe its one-dimensional shape with a rank-one extents type.
                 using grid_shape = cuda::hierarchy_level_desc<cuda::grid_level, cuda::std::extents<unsigned, 17>>;
                 check_rank<<<17, 1>>>(cuda::hierarchy{cuda::block, grid_shape{}});
                 assert(cudaDeviceSynchronize() == cudaSuccess);
               }))
  return 0;
}
