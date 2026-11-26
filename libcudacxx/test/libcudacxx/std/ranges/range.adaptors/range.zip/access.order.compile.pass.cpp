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

#include "test_macros.h"

struct tracked_view : cuda::std::ranges::view_base
{
  int* calls;
  int id;

  TEST_FUNC constexpr int* begin() const
  {
    *calls = *calls * 10 + id;
    return nullptr;
  }

  TEST_FUNC constexpr cuda::std::unreachable_sentinel_t end() const
  {
    *calls = *calls * 10 + id;
    return {};
  }
};

TEST_FUNC constexpr bool test()
{
  int calls = 0;
  const cuda::std::ranges::zip_view view(tracked_view{{}, &calls, 1}, tracked_view{{}, &calls, 2});
  (void) view.begin();
  (void) view.end();
  return calls == 1212;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
