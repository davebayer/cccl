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

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct common_view : cuda::std::ranges::view_base
{
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> begin() const;
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> end() const;
};

template <int I>
struct probe
{
  // A common view can construct its reverse iterator directly from end().
  static constexpr bool value =
    cuda::std::is_constructible_v<cuda::std::ranges::reverse_view<common_view<I>>, common_view<I>>;
};

static_assert(instantiate<probe>());
