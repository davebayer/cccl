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
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    int values[I + 2]{};
    const cuda::std::ranges::ref_view view{values};
    static_assert(!cuda::std::is_constructible_v<decltype(view), decltype(values)&&>);
    *view.begin() = I;
    return view.base()[0] == I && !view.empty() && view.data() == values && view.end() == values + I + 2
        && view.size() == I + 2;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
