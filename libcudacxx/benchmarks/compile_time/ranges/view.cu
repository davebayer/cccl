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

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  // A product type has no begin or end, so checking whether its elements make it movable is unnecessary.
  static constexpr bool value = !cuda::std::ranges::view<cuda::std::tuple<element<I>, int, long>>;
};

static_assert(instantiate<probe>());
