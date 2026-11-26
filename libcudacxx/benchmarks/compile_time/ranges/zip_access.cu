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
struct view : cuda::std::ranges::view_base
{
  _CCCL_HOST_DEVICE element<I>* begin() const;
  _CCCL_HOST_DEVICE cuda::std::unreachable_sentinel_t end() const;
};

template <int I>
struct probe
{
  using zip                   = cuda::std::ranges::zip_view<view<I>, view<I + 1>>;
  using iter                  = ::cuda::zip_iterator<element<I>*, element<I + 1>*>;
  static constexpr bool value = cuda::std::is_same_v<decltype(cuda::std::declval<zip&>().begin()), iter>
                             && !cuda::std::is_same_v<decltype(cuda::std::declval<zip&>().end()), iter>;
};

static_assert(instantiate<probe, 24>());
