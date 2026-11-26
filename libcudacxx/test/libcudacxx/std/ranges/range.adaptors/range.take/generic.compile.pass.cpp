//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
struct range
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;

  // The generic take_view constructor does not need the size of its argument.
  TEST_FUNC auto size() const
  {
    return T::size();
  }
};

struct invalid_count
{};

int main(int, char**)
{
  using input = range<void>;
  using view  = cuda::std::ranges::take_view<cuda::std::ranges::ref_view<input>>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::views::take(cuda::std::declval<input&>(), 2)), view>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::take), input&, invalid_count>);
  return 0;
}
