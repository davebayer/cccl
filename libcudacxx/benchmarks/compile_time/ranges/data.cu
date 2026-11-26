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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct range
{
  _CCCL_HOST_DEVICE const element<I>* data() const;

  // The contiguous-iterator fallback is unnecessary when data() returns an object pointer.
  _CCCL_HOST_DEVICE cuda::std::reverse_iterator<element<I>*> begin() const;
};

template <int I>
struct probe
{
  static constexpr bool value =
    cuda::std::is_same_v<decltype(cuda::std::ranges::data(cuda::std::declval<const range<I>&>())), const element<I>*>;
};

static_assert(instantiate<probe>());
