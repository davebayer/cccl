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
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  using bounded   = decltype(cuda::std::views::repeat(element<I>{}, 3));
  using unbounded = decltype(cuda::std::views::repeat(element<I>{}));
  static constexpr bool value =
    cuda::std::is_same_v<bounded, cuda::std::ranges::repeat_view<element<I>, int>>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const bounded&>().begin()[1]), const element<I>&>
    && cuda::std::is_unsigned_v<decltype(cuda::std::declval<const bounded&>().size())>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const unbounded&>().end()), cuda::std::unreachable_sentinel_t>;
};

static_assert(instantiate<probe, 32>());
