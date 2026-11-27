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
struct list_cursor
{
  using iterator_concept = cuda::std::bidirectional_iterator_tag;
  using difference_type  = cuda::std::ptrdiff_t;
  using value_type       = range_element<I>;
  _CCCL_HOST_DEVICE value_type& operator*() const;
  _CCCL_HOST_DEVICE list_cursor& operator++();
  _CCCL_HOST_DEVICE list_cursor operator++(int);
  _CCCL_HOST_DEVICE list_cursor& operator--();
  _CCCL_HOST_DEVICE list_cursor operator--(int);
  _CCCL_HOST_DEVICE friend bool operator==(list_cursor lhs, list_cursor rhs)
  {
    return lhs.current == rhs.current;
  }
  _CCCL_HOST_DEVICE friend bool operator!=(list_cursor lhs, list_cursor rhs)
  {
    return !(lhs == rhs);
  }
  value_type* current;
};

template <int I>
struct probe
{
  using reverse               = cuda::std::reverse_iterator<range_element<I>*>;
  static constexpr bool value = cuda::std::ranges::random_access_range<cursor_range<reverse>>
                             && !cuda::std::ranges::random_access_range<cursor_range<list_cursor<I>>>;
};

static_assert(instantiate<probe, 32>());
