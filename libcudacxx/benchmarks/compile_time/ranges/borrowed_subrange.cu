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
struct reverse_cursor
{
  using difference_type = cuda::std::ptrdiff_t;
  using value_type      = element<I>;

  element<I>* current;
  _CCCL_HOST_DEVICE value_type& operator*() const;
  _CCCL_HOST_DEVICE reverse_cursor& operator++();
  _CCCL_HOST_DEVICE void operator++(int);

  _CCCL_HOST_DEVICE friend bool operator==(reverse_cursor lhs, reverse_cursor rhs)
  {
    return lhs.current == rhs.current;
  }
  _CCCL_HOST_DEVICE friend bool operator!=(reverse_cursor lhs, reverse_cursor rhs)
  {
    return !(lhs == rhs);
  }
  _CCCL_HOST_DEVICE friend auto operator-(reverse_cursor lhs, reverse_cursor rhs)
  {
    return cuda::std::ranges::distance(
      cuda::std::make_reverse_iterator(rhs.current), cuda::std::make_reverse_iterator(lhs.current));
  }
};

template <int I>
struct reverse_range
{
  _CCCL_HOST_DEVICE reverse_cursor<I> begin();
  _CCCL_HOST_DEVICE cuda::std::unreachable_sentinel_t end();
};

template <int I>
struct probe
{
  static constexpr bool value =
    cuda::std::is_same_v<cuda::std::ranges::borrowed_subrange_t<reverse_range<I>>, cuda::std::ranges::dangling>;
};

static_assert(instantiate<probe>());
