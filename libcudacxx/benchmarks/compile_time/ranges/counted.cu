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
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  using ptr     = element<I>*;
  using reverse = cuda::std::reverse_iterator<ptr>;
  using forward = cuda::std::common_iterator<ptr, cuda::std::unreachable_sentinel_t>;
  using counted_forward =
    cuda::std::ranges::subrange<cuda::std::counted_iterator<forward>, cuda::std::default_sentinel_t>;
  static constexpr bool value =
    cuda::std::is_same_v<decltype(cuda::std::views::counted(cuda::std::declval<ptr>(), 3)), cuda::std::span<element<I>>>
    && cuda::std::is_same_v<decltype(cuda::std::views::counted(cuda::std::declval<reverse>(), 3)),
                            cuda::std::ranges::subrange<reverse>>
    && cuda::std::is_same_v<decltype(cuda::std::views::counted(cuda::std::declval<forward>(), 3)), counted_forward>;
};

static_assert(instantiate<probe, 24>());
