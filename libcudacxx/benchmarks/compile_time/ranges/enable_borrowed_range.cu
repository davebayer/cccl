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

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  using owner = cuda::std::ranges::single_view<range_element<I>>;
  using ref   = cuda::std::ranges::ref_view<owner>;
  static constexpr bool value =
    cuda::std::ranges::enable_borrowed_range<ref> && !cuda::std::ranges::enable_borrowed_range<owner>;
};

static_assert(instantiate<probe, 32>());
