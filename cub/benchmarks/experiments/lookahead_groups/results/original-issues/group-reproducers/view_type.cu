// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/utility>

#include <cuda/experimental/coop/group>
namespace coop = cuda::experimental::coop;
using H        = decltype(cuda::make_hierarchy(cuda::block_dims(128u)));
using B        = coop::this_block<H>;
using V        = decltype(coop::group_view{cuda::warp, cuda::std::declval<const B&>()});
static_assert(cuda::std::is_same_v<V::unit_type, cuda::warp_level>);
int main() {}
