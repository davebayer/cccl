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
struct member_range
{
  _CCCL_HOST_DEVICE range_element<I>* begin();
  _CCCL_HOST_DEVICE range_element<I>* end();
};

template <int I>
struct probe
{
  using range                 = member_range<I>;
  static constexpr bool value = cuda::std::ranges::range<range> && !cuda::std::ranges::range<const range>;
};

static_assert(instantiate<probe, 32>());
