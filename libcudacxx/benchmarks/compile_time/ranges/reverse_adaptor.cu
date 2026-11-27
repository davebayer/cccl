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
  using buffer            = element<I>[3];
  using ptr               = element<I>*;
  using ref               = cuda::std::ranges::ref_view<buffer>;
  using reverse           = decltype(cuda::std::views::reverse(cuda::std::declval<buffer&>()));
  using reversed_subrange = cuda::std::ranges::subrange<cuda::std::reverse_iterator<ptr>>;
  static constexpr bool value =
    cuda::std::is_same_v<reverse, cuda::std::ranges::reverse_view<ref>>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const reverse&>().begin()), cuda::std::reverse_iterator<ptr>>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const reverse&>().end()), cuda::std::reverse_iterator<ptr>>
    && cuda::std::is_unsigned_v<decltype(cuda::std::declval<const reverse&>().size())>
    && cuda::std::is_same_v<decltype(cuda::std::views::reverse(cuda::std::declval<reverse>())), ref>
    && cuda::std::is_same_v<decltype(cuda::std::views::reverse(cuda::std::declval<reversed_subrange>())),
                            cuda::std::ranges::subrange<ptr>>;
};

static_assert(instantiate<probe, 24>());
