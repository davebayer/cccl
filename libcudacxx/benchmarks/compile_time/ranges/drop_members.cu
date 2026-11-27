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
    const cuda::std::ranges::drop_view tail{source, 1};
    const cuda::std::ranges::drop_view exhausted{source, I + 3};
    return tail.base().data() == values && tail.begin() == values + 1 && tail.end() == values + I + 2
        && tail.size() == I + 1 && exhausted.begin() == exhausted.end() && exhausted.size() == 0;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
