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

template <class T>
struct MemberAccess
{
  int value;

  TEST_FUNC constexpr const int* begin() const
  {
    return &value;
  }
  TEST_FUNC constexpr const int* end() const
  {
    return &value + 1;
  }

  TEST_FUNC friend constexpr auto begin(const MemberAccess&)
  {
    return T::begin();
  }
  TEST_FUNC friend constexpr auto end(const MemberAccess&)
  {
    return T::end();
  }
};

template <class T>
struct MemberData
{
  TEST_FUNC constexpr const int* data() const
  {
    return nullptr;
  }

  TEST_FUNC constexpr auto begin() const
  {
    return T::begin();
  }
};

template <class T>
struct NonBorrowed
{
  TEST_FUNC constexpr auto data() const
  {
    return T::data();
  }
  TEST_FUNC constexpr auto begin() const
  {
    return T::begin();
  }
  TEST_FUNC constexpr auto end() const
  {
    return T::end();
  }
};

int main(int, char**)
{
  namespace ranges = cuda::std::ranges;
  constexpr MemberAccess<void> range{42};
  constexpr MemberData<void> with_data{};

  static_assert(*ranges::begin(range) == 42);
  static_assert(ranges::end(range) - ranges::begin(range) == 1);
  static_assert(ranges::data(with_data) == nullptr);
  static_assert(!cuda::std::is_invocable_v<decltype(ranges::begin), NonBorrowed<void>>);
  static_assert(!cuda::std::is_invocable_v<decltype(ranges::end), NonBorrowed<void>>);
  static_assert(!cuda::std::is_invocable_v<decltype(ranges::data), NonBorrowed<void>>);
  return 0;
}
