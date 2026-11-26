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

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  static constexpr bool value =
    !cuda::std::ranges::viewable_range<element<I>> && !cuda::std::ranges::viewable_range<const element<I>&>;
};

static_assert(instantiate<probe>());
