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
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct iterator
{
  using value_type = cuda::std::tuple<element<I>, int>;
  value_type* ptr;

  _CCCL_HOST_DEVICE value_type& operator*() const;
  _CCCL_HOST_DEVICE friend void iter_swap(iterator lhs, value_type* rhs) noexcept
  {
    lhs.ptr->swap(*rhs);
  }
};

template <int I>
struct probe
{
  // A customization exchanges both elements directly without probing tuple reference swapping.
  static constexpr bool value = cuda::std::
    is_nothrow_invocable_v<decltype(cuda::std::ranges::iter_swap), iterator<I>, typename iterator<I>::value_type*>;
};

static_assert(instantiate<probe>());
