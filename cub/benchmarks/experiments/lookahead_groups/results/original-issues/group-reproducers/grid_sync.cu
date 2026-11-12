// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/utility>

#include <cuda/experimental/coop/group>
using H = decltype(cuda::make_hierarchy(cuda::grid_dims(2u), cuda::block_dims(32u)));
__global__ void use_grid_sync(H h)
{
  const cuda::experimental::coop::this_grid grid{h};
  grid.sync();
  grid.wait(grid.arrive());
}
int main() {}
