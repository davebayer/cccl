//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct strided_iterator
{
  using difference_type = cuda::std::ptrdiff_t;
  using value_type      = element<I>;

  value_type* current;
  difference_type stride;
  _CCCL_HOST_DEVICE value_type& operator*() const;
  _CCCL_HOST_DEVICE strided_iterator& operator++();
  _CCCL_HOST_DEVICE void operator++(int);

  _CCCL_HOST_DEVICE friend auto operator-(strided_iterator lhs, strided_iterator rhs)
  {
    return cuda::std::ranges::distance(rhs.current, lhs.current) / lhs.stride;
  }
};

template <int I>
struct strided_range
{
  _CCCL_HOST_DEVICE strided_iterator<I> begin();
  _CCCL_HOST_DEVICE cuda::std::unreachable_sentinel_t end();
};

template <int I>
struct probe
{
  static constexpr bool value =
    cuda::std::is_same_v<cuda::std::ranges::range_difference_t<strided_range<I>>, cuda::std::ptrdiff_t>;
};

static_assert(instantiate<probe>());
