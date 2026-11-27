//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  using pointer = range_element<I>*;
  using forward = cuda::std::common_iterator<pointer, cuda::std::unreachable_sentinel_t>;
  static constexpr bool value =
    cuda::std::ranges::bidirectional_range<cursor_range<cuda::std::reverse_iterator<pointer>>>
    && !cuda::std::ranges::bidirectional_range<cursor_range<forward>>;
};

static_assert(instantiate<probe, 32>());
