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

template <class Size, bool NoThrow>
struct sized_view : cuda::std::ranges::view_base
{
  Size n;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;

  TEST_FUNC constexpr Size size() noexcept
  {
    return n;
  }
  TEST_FUNC constexpr Size size() const noexcept(NoThrow)
  {
    return n;
  }
};

TEST_FUNC constexpr bool test()
{
  cuda::std::ranges::zip_view zip{sized_view<short, true>{{}, 9}, sized_view<unsigned long, false>{{}, 4}};
  static_assert(cuda::std::is_same_v<decltype(zip.size()), unsigned long>);
  static_assert(noexcept(zip.size()));
  static_assert(!noexcept(cuda::std::as_const(zip).size()));
  return zip.size() == 4 && cuda::std::as_const(zip).size() == 4;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
