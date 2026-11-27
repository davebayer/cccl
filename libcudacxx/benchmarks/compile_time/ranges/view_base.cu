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
struct marked_range : cuda::std::ranges::view_base
{
  _CCCL_HOST_DEVICE range_element<I>* begin() const;
  _CCCL_HOST_DEVICE range_element<I>* end() const;
};

template <int I>
struct probe
{
  using view = marked_range<I>;
  static constexpr bool value =
    cuda::std::is_convertible_v<view*, cuda::std::ranges::view_base*> && cuda::std::ranges::view<view>;
};

static_assert(instantiate<probe, 32>());
