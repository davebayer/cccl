//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  using owner = cuda::std::ranges::single_view<range_element<I>>;
  using ref   = cuda::std::ranges::ref_view<owner>;
  static constexpr bool value =
    cuda::std::is_same_v<cuda::std::ranges::borrowed_iterator_t<owner&>, range_element<I>*>
    && cuda::std::is_same_v<cuda::std::ranges::borrowed_iterator_t<ref>, range_element<I>*>
    && cuda::std::is_same_v<cuda::std::ranges::borrowed_iterator_t<owner>, cuda::std::ranges::dangling>;
};

static_assert(instantiate<probe, 32>());
