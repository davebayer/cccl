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
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int I>
struct below
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr bool operator()(int x) const
  {
    return x < I;
  }
};

template <int I>
struct probe
{
  using buffer   = int[4];
  using view     = decltype(cuda::std::views::filter(cuda::std::declval<buffer&>(), below<I>{}));
  using iterator = decltype(cuda::std::declval<view&>().begin());
  using sentinel = decltype(cuda::std::declval<view&>().end());
  static constexpr bool value =
    cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().base()), cuda::std::ranges::ref_view<buffer>>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().pred()), const below<I>&>
    && cuda::std::is_same_v<decltype(cuda::std::declval<iterator&>() != cuda::std::declval<const sentinel&>()), bool>
    && cuda::std::is_same_v<decltype(*cuda::std::declval<iterator&>()), int&>
    && cuda::std::is_same_v<decltype(++cuda::std::declval<iterator&>()), iterator&>
    && cuda::std::is_same_v<decltype(--cuda::std::declval<iterator&>()), iterator&>;
};

static_assert(instantiate<probe, 32>());
