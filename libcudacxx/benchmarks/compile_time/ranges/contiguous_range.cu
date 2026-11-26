//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct range
{
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> begin() const;
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> end() const;
  // The elements are stored contiguously but visited in reverse order.
  _CCCL_HOST_DEVICE const cuda::std::ranges::subrange<element<I>*>& storage() const;
  _CCCL_HOST_DEVICE auto data() const
  {
    return cuda::std::ranges::data(storage());
  }
};

template <int I>
struct probe
{
  static constexpr bool value = !cuda::std::ranges::contiguous_range<range<I>>;
};

static_assert(instantiate<probe>());
