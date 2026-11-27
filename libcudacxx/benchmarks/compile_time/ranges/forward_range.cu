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

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct input_cursor
{
  using iterator_concept = cuda::std::input_iterator_tag;
  using difference_type  = cuda::std::ptrdiff_t;
  using value_type       = range_element<I>;
  _CCCL_HOST_DEVICE value_type& operator*() const;
  _CCCL_HOST_DEVICE input_cursor& operator++();
  _CCCL_HOST_DEVICE void operator++(int);
};

template <int I>
struct probe
{
  using input = cursor_range<input_cursor<I>, cuda::std::unreachable_sentinel_t>;
  static constexpr bool value =
    cuda::std::ranges::forward_range<range_element<I>[4]> && !cuda::std::ranges::forward_range<input>;
};

static_assert(instantiate<probe, 32>());
