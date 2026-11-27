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
struct forward_view : cuda::std::ranges::view_base
{
  using iterator = cuda::std::common_iterator<element<I>*, cuda::std::unreachable_sentinel_t>;
  _CCCL_HOST_DEVICE iterator begin() const;
  _CCCL_HOST_DEVICE iterator end() const;
  // The backing storage has constant-time distance even though the public iterator is forward-only.
  _CCCL_HOST_DEVICE const cuda::std::ranges::subrange<element<I>*>& storage() const;
  _CCCL_HOST_DEVICE auto size() const
  {
    return cuda::std::ranges::distance(storage());
  }
};

template <int I>
struct probe
{
  using base = forward_view<I>;
  using view = cuda::std::ranges::drop_view<base>;
  static constexpr bool value =
    cuda::std::is_same_v<decltype(cuda::std::declval<view&>().begin()), typename base::iterator>;
};

static_assert(instantiate<probe, 32>());
