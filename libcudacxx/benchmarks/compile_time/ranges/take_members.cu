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
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    int values[I + 2]{};
    const cuda::std::ranges::ref_view source{values};
    const cuda::std::ranges::take_view prefix{source, 1};
    const cuda::std::ranges::take_view all{source, I + 3};
    return prefix.base().data() == values && prefix.begin() == values && prefix.end() == values + 1
        && prefix.size() == 1 && all.end() == values + I + 2 && all.size() == I + 2;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
