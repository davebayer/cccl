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

template <int>
struct element
{};

template <int I>
struct queue
{
  _CCCL_HOST_DEVICE const cuda::std::ranges::subrange<element<I>*>& storage() const;
  _CCCL_HOST_DEVICE auto size() const
  {
    return cuda::std::ranges::size(storage());
  }
};

template <class T, class = void>
constexpr bool has_size_type = false;

template <class T>
constexpr bool has_size_type<T, cuda::std::void_t<cuda::std::ranges::range_size_t<T>>> = true;

template <int I>
struct probe
{
  static constexpr bool value = !has_size_type<queue<I>>;
};

static_assert(instantiate<probe>());
