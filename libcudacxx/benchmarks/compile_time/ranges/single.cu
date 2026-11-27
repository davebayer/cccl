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

template <int>
struct element
{};

template <int I>
struct probe
{
  using view = decltype(cuda::std::views::single(element<I>{}));
  static constexpr bool value =
    cuda::std::is_same_v<view, cuda::std::ranges::single_view<element<I>>> && view::size() == 1
    && cuda::std::is_same_v<decltype(cuda::std::declval<view&>().begin()), element<I>*>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().end()), const element<I>*>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().data()), const element<I>*>;
};

static_assert(instantiate<probe>());
