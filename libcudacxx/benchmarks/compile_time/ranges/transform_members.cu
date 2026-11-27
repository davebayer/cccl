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

template <int I>
struct offset
{
  _CCCL_HOST_DEVICE constexpr int operator()(int value) const noexcept
  {
    return value + I;
  }
};

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    int values[]{1, 2, 3};
    const cuda::std::ranges::transform_view view{cuda::std::ranges::ref_view{values}, offset<I>{}};
    auto current = view.begin();
    ++current;
    return view.base().data() == values && view.size() == 3 && current.base() == values + 1 && *current == I + 2
        && current[1] == I + 3 && view.end() - current == 2;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
