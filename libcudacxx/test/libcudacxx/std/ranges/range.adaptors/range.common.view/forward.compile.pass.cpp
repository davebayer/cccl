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

template <class T>
struct forward_view : cuda::std::ranges::view_base
{
  using iterator = forward_iterator<int*>;
  TEST_FUNC iterator begin() const;
  TEST_FUNC sentinel_wrapper<iterator> end() const;
  TEST_FUNC auto size() const
  {
    return T::size();
  }
};

int main(int, char**)
{
  // A forward range uses common_iterator even when deducing its size() would be ill-formed.
  using iterator = forward_view<void>::iterator;
  using result   = cuda::std::common_iterator<iterator, sentinel_wrapper<iterator>>;
  using view     = cuda::std::ranges::common_view<forward_view<void>>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<view&>().begin()), result>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<view&>().end()), result>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().begin()), result>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const view&>().end()), result>);
  return 0;
}
