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
struct range
{
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> begin() const;
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> end() const;
};

template <int I>
struct probe
{
  // This range always uses the generic take_view.
  using result = decltype(cuda::std::views::take(cuda::std::declval<range<I>&>(), 2));
  static constexpr bool value =
    cuda::std::is_same_v<result, cuda::std::ranges::take_view<cuda::std::ranges::ref_view<range<I>>>>;
};

static_assert(instantiate<probe>());
