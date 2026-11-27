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
struct skip_first : cuda::std::ranges::range_adaptor_closure<skip_first<I>>
{
  _CCCL_HOST_DEVICE constexpr cuda::std::ranges::subrange<int*> operator()(int (&values)[I + 2]) const
  {
    return {values + 1, values + I + 2};
  }
};

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    int values[I + 2]{};
    const auto tail = values | skip_first<I>{};
    return tail.begin() == values + 1 && tail.size() == I + 1;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
