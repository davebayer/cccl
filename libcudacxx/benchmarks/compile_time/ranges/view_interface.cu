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
struct buffer : cuda::std::ranges::view_interface<buffer<I>>
{
  int values[3]{};

  _CCCL_HOST_DEVICE constexpr int* begin() noexcept
  {
    return values;
  }
  _CCCL_HOST_DEVICE constexpr int* end() noexcept
  {
    return values + 3;
  }
  _CCCL_HOST_DEVICE constexpr const int* begin() const noexcept
  {
    return values;
  }
  _CCCL_HOST_DEVICE constexpr const int* end() const noexcept
  {
    return values + 3;
  }
};

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    buffer<I> view{};
    view.front()       = I;
    view[1]            = I + 1;
    view.back()        = I + 2;
    const auto& shared = view;
    return !shared.empty() && static_cast<bool>(shared) && shared.size() == 3 && shared.data() == view.begin()
        && shared.front() == I && shared.at(1) == I + 1 && shared.back() == I + 2;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
