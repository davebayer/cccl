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

#include "test_macros.h"

template <class T, class = void>
constexpr bool has_size_type = false;

template <class T>
constexpr bool has_size_type<T, cuda::std::void_t<cuda::std::ranges::range_size_t<T>>> = true;

template <class T>
struct non_range
{
  TEST_FUNC auto size() const
  {
    return T::size();
  }
};

int main(int, char**)
{
  static_assert(!has_size_type<non_range<void>>);
  static_assert(has_size_type<int[3]>);
  return 0;
}
