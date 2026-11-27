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
  using iterator = range_element<I>*;
  static constexpr bool value =
    cuda::std::ranges::common_range<cursor_range<iterator>>
    && !cuda::std::ranges::common_range<cursor_range<iterator, cuda::std::unreachable_sentinel_t>>;
};

static_assert(instantiate<probe, 32>());
