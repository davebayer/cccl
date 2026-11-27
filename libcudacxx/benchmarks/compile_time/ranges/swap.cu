//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "instantiate.cuh"

template <int>
struct record
{
  int value;
  _CCCL_HOST_DEVICE friend constexpr void swap(record& lhs, record& rhs) noexcept
  {
    const int temp = lhs.value;
    lhs.value      = rhs.value;
    rhs.value      = temp;
  }
};

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool check()
  {
    record<I> a{1};
    record<I> b{2};
    cuda::std::ranges::swap(a, b);
    return a.value == 2 && b.value == 1;
  }

  static constexpr bool value = check();
};

static_assert(instantiate<probe, 32>());
