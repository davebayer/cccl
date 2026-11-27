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

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct value_cursor
{
  using difference_type = cuda::std::ptrdiff_t;
  using value_type      = range_element<I>;
  const value_type* current;
  _CCCL_HOST_DEVICE const value_type& operator*() const
  {
    return *current;
  }
  _CCCL_HOST_DEVICE value_cursor& operator++();
  _CCCL_HOST_DEVICE void operator++(int);
  _CCCL_HOST_DEVICE friend value_type iter_move(value_cursor it)
  {
    return *it;
  }
};

template <int I>
struct probe
{
  using item       = range_element<I>;
  using customized = cursor_range<value_cursor<I>, cuda::std::unreachable_sentinel_t>;
  static constexpr bool value =
    cuda::std::is_same_v<cuda::std::ranges::range_rvalue_reference_t<const item[4]>, const item&&>
    && cuda::std::is_same_v<cuda::std::ranges::range_rvalue_reference_t<customized>, item>;
};

static_assert(instantiate<probe, 32>());
