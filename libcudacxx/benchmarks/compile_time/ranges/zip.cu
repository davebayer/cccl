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
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  using first    = element<I>[3];
  using second   = int[4];
  using view     = decltype(cuda::std::views::zip(cuda::std::declval<first&>(), cuda::std::declval<second&>()));
  using expected = cuda::std::ranges::zip_view<cuda::std::ranges::ref_view<first>, cuda::std::ranges::ref_view<second>>;
  static constexpr bool value =
    cuda::std::is_same_v<view, expected>
    && cuda::std::is_same_v<decltype(cuda::std::get<0>(*cuda::std::declval<view&>().begin())), element<I>&>
    && cuda::std::is_same_v<decltype(cuda::std::get<1>(*cuda::std::declval<view&>().begin())), int&>;
};

static_assert(cuda::std::views::zip().empty());
static_assert(instantiate<probe, 24>());
