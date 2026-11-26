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

template <int Index>
struct element
{};

template <int Index>
struct range
{
  _CCCL_HOST_DEVICE constexpr bool empty() const
  {
    return true;
  }

  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<Index>*> begin() const;
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<Index>*> end() const;
};

template <int Index>
struct probe
{
  static constexpr bool value = cuda::std::ranges::empty(range<Index>{});
};

static_assert(instantiate<probe>());
