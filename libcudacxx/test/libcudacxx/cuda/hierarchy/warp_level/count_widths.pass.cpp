//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/hierarchy>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_FUNC constexpr bool test_warp_count_widths()
{
  // The thread extent need not fit in the requested warp count type.
  auto fixed = cuda::make_hierarchy(cuda::block_dims<257>());
  auto sized = cuda::make_hierarchy(cuda::block_dims(257));
  assert(cuda::warp.count_as<unsigned char>(cuda::block, fixed) == 9);
  assert(cuda::warp.count_as<unsigned char>(cuda::block, sized) == 9);

  auto max_x = cuda::make_hierarchy(cuda::block_dims(0xffff'ffffu));
  assert(cuda::warp.count(cuda::block, max_x) == (1u << 27));

  // A hierarchy can describe dimensions larger than a running block.
  auto large = cuda::make_hierarchy(cuda::block_dims(dim3{65536, 65536, 32}));
  assert(cuda::warp.count_as<unsigned long long>(cuda::block, large) == (1ull << 32));
  return true;
}

int main(int, char**)
{
  test_warp_count_widths();
  static_assert(test_warp_count_widths());
  return 0;
}
