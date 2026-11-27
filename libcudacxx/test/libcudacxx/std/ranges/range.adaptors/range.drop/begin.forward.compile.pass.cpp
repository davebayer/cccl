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

#include "test_iterators.h"
#include "test_macros.h"

template <class Iterator>
struct view : cuda::std::ranges::view_base
{
  TEST_FUNC Iterator begin() const;
  TEST_FUNC sentinel_wrapper<Iterator> end() const;
  TEST_FUNC auto size() const
  {
    return Iterator::unused;
  }
};

template <class Iterator>
TEST_FUNC constexpr bool test()
{
  using dropped = cuda::std::ranges::drop_view<view<Iterator>>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<dropped&>().begin()), Iterator>);
  static_assert(!cuda::std::ranges::range<const dropped>);
  return true;
}

int main(int, char**)
{
  static_assert(test<cpp20_input_iterator<int*>>());
  static_assert(test<forward_iterator<int*>>());
  return 0;
}
