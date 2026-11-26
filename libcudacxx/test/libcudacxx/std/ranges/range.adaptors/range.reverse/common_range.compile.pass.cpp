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

#include "test_macros.h"

template <class T>
struct iterator
{
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using iterator_category = cuda::std::random_access_iterator_tag;

  TEST_FUNC int& operator*() const;
  TEST_FUNC iterator& operator++();
  TEST_FUNC iterator operator++(int);
  TEST_FUNC iterator& operator--();
  TEST_FUNC iterator operator--(int);
  TEST_FUNC friend bool operator==(iterator, iterator)
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(iterator, iterator)
  {
    return false;
  }

  // A common bidirectional range needs no random-access operations to be reversed.
  TEST_FUNC friend auto operator<(iterator, iterator)
  {
    return T::less();
  }
};

struct view : cuda::std::ranges::view_base
{
  TEST_FUNC iterator<void> begin() const;
  TEST_FUNC iterator<void> end() const;
};

int main(int, char**)
{
  static_assert(cuda::std::ranges::common_range<view>);
  static_assert(cuda::std::ranges::bidirectional_range<view>);
  static_assert(cuda::std::is_constructible_v<cuda::std::ranges::reverse_view<view>, view>);
  return 0;
}
