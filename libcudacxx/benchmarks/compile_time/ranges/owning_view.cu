//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    using storage = cuda::std::array<int, I + 2>;
    cuda::std::ranges::owning_view view{storage{}};
    const auto& shared = view;
    static_assert(cuda::std::is_same_v<decltype(shared.base()), const storage&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(view).base()), storage&&>);
    *view.begin() = I;
    return shared.base()[0] == I && !shared.empty() && shared.data() == view.data() && shared.begin() == view.begin()
        && shared.end() == view.end() && shared.size() == I + 2;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
