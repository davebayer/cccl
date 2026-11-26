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

#include "instantiate.cuh"

template <int I>
struct callable
{
  _CCCL_HOST_DEVICE int operator()(int x) const
  {
    return x + I;
  }
};

template <int I>
struct probe
{
  using pipeline = decltype(cuda::std::views::transform(callable<I>{}) | cuda::std::views::filter(callable<I>{}));
  static constexpr bool value = cuda::std::is_move_constructible_v<pipeline>;
};

static_assert(instantiate<probe, 32>());
