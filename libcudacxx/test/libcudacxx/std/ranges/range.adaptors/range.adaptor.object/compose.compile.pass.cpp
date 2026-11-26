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

struct front : cuda::std::ranges::range_adaptor_closure<front>
{
  template <class Range>
  TEST_FUNC constexpr auto operator()(Range&& range) const
  {
    return *cuda::std::ranges::begin(range);
  }
};

struct non_range : cuda::std::ranges::view_base
{};

template <class T>
struct poison_begin
{
  TEST_FUNC auto begin() const
  {
    return T::missing;
  }
};

template <class Left, class Right, class = void>
constexpr bool can_pipe = false;

template <class Left, class Right>
constexpr bool
  can_pipe<Left, Right, cuda::std::void_t<decltype(cuda::std::declval<Left>() | cuda::std::declval<Right>())>> = true;

// Neither pipe overload should inspect the right operand after the left operand fails its constraint.
static_assert(!can_pipe<non_range, poison_begin<void>>);

TEST_FUNC constexpr bool test()
{
  // Deducing front's return type for views::all would try to find the beginning of a closure.
  auto pipeline = cuda::std::views::all | front{};
  int values[]  = {42, 50};
  return pipeline(values) == 42;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
