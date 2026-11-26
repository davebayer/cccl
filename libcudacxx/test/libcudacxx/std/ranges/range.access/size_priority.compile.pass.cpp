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
struct with_member_size
{
  TEST_FUNC constexpr int size() const
  {
    return 1;
  }

  TEST_FUNC friend constexpr auto size(const with_member_size&)
  {
    return T::size();
  }
};

template <class T>
struct with_adl_size
{
  TEST_FUNC friend constexpr int size(const with_adl_size&)
  {
    return 2;
  }

  TEST_FUNC constexpr auto begin() const
  {
    return T::begin();
  }
};

struct sentinel
{
  TEST_FUNC friend bool operator==(int*, sentinel);
  TEST_FUNC friend bool operator==(sentinel, int*);
  TEST_FUNC friend bool operator!=(int*, sentinel);
  TEST_FUNC friend bool operator!=(sentinel, int*);

  TEST_FUNC friend cuda::std::ptrdiff_t operator-(const sentinel&, int* const&);
  TEST_FUNC friend cuda::std::ptrdiff_t operator-(int* const&, const sentinel&);
  TEST_FUNC friend void operator-(sentinel&&, int*&&) = delete;
};

struct with_sentinel
{
  TEST_FUNC int* begin() const;
  TEST_FUNC sentinel end() const;
};

int main(int, char**)
{
  // Deducing a lower-priority customization's return type is ill-formed.
  static_assert(cuda::std::ranges::size(with_member_size<void>{}) == 1);
  static_assert(cuda::std::ranges::size(with_adl_size<void>{}) == 2);

  // sized_sentinel_for checks const lvalues, while ranges::size subtracts the results of end() and begin().
  static_assert(cuda::std::sized_sentinel_for<sentinel, int*>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::size), with_sentinel&>);
  return 0;
}
