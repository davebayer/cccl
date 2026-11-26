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
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct custom_iterator
{
  int* ptr;

  TEST_FUNC auto operator*() const
  {
    return T::unused;
  }

  TEST_FUNC friend void iter_swap(custom_iterator lhs, custom_iterator rhs) noexcept
  {
    const int tmp = *lhs.ptr;
    *lhs.ptr      = *rhs.ptr;
    *rhs.ptr      = tmp;
  }
};

template <class T>
struct iterator
{
  using value_type = T;
  T* ptr;
  TEST_FUNC constexpr T& operator*() const
  {
    return *ptr;
  }
};

// Only this order has an ADL customization. The opposite order uses move-exchange.
TEST_FUNC constexpr void iter_swap(iterator<int> lhs, iterator<long> rhs) noexcept
{
  const auto tmp = *lhs;
  *lhs           = static_cast<int>(*rhs);
  *rhs           = tmp;
}

TEST_FUNC constexpr bool test()
{
  int x  = 1;
  long y = 2;
  cuda::std::ranges::iter_swap(iterator<int>{&x}, iterator<long>{&y});
  return x == 2 && y == 1;
}

int main(int, char**)
{
  using customized = custom_iterator<void>;
  static_assert(cuda::std::is_nothrow_invocable_v<decltype(cuda::std::ranges::iter_swap), customized, customized>);
  static_assert(cuda::std::is_invocable_v<decltype(cuda::std::ranges::iter_swap), iterator<long>, iterator<int>>);
  static_assert(test());
  return 0;
}
