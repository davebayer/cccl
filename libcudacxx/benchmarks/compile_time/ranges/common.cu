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
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct forward_iterator
{
  using value_type       = element<I>;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::forward_iterator_tag;

  element<I>* ptr;
  _CCCL_HOST_DEVICE value_type& operator*() const;
  _CCCL_HOST_DEVICE forward_iterator& operator++();
  _CCCL_HOST_DEVICE forward_iterator operator++(int);
  _CCCL_HOST_DEVICE friend bool operator==(forward_iterator lhs, forward_iterator rhs)
  {
    return lhs.ptr == rhs.ptr;
  }
  _CCCL_HOST_DEVICE friend bool operator!=(forward_iterator lhs, forward_iterator rhs)
  {
    return !(lhs == rhs);
  }
};

template <int I>
struct counted_range : cuda::std::ranges::view_base
{
  using iterator = cuda::std::counted_iterator<forward_iterator<I>>;
  _CCCL_HOST_DEVICE iterator begin() const;
  _CCCL_HOST_DEVICE cuda::std::default_sentinel_t end() const;
  _CCCL_HOST_DEVICE auto size() const
  {
    // The counted iterator and its sentinel support constant-time distance.
    return cuda::std::ranges::distance(begin(), end());
  }
};

template <int I>
struct probe
{
  using view   = cuda::std::ranges::common_view<counted_range<I>>;
  using result = cuda::std::common_iterator<typename counted_range<I>::iterator, cuda::std::default_sentinel_t>;
  static constexpr bool value =
    cuda::std::is_same_v<decltype(cuda::std::declval<view&>().begin()), result>
    && cuda::std::is_same_v<decltype(cuda::std::declval<view&>().end()), result>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().begin()), result>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().end()), result>;
};

static_assert(instantiate<probe, 32>());
