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
#include <cuda/std/type_traits>

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  using pointer = range_element<I>*;
  using counted = cursor_range<cuda::std::counted_iterator<pointer>, cuda::std::default_sentinel_t>;
  static constexpr bool value =
    cuda::std::is_same_v<cuda::std::ranges::sentinel_t<cursor_range<pointer>>, pointer>
    && cuda::std::is_same_v<cuda::std::ranges::sentinel_t<counted>, cuda::std::default_sentinel_t>;
};

static_assert(instantiate<probe, 32>());
