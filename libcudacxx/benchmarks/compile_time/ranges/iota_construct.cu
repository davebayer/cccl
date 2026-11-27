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
  _CCCL_HOST_DEVICE static constexpr bool check()
  {
    element<I> elements[3]{};
    const auto range = cuda::std::views::iota(elements + 0, elements + 3);
    return range.size() == 3 && *range.begin() == elements && range.begin()[2] == elements + 2;
  }

  static constexpr bool value = check();
};

static_assert(instantiate<probe, 32>());
