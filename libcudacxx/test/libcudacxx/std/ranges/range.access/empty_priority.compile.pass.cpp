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

#include "test_macros.h"

template <class T>
struct with_member_empty
{
  TEST_FUNC constexpr bool empty() const
  {
    return true;
  }

  TEST_FUNC auto size() const
  {
    return T::size();
  }

  TEST_FUNC auto begin() const
  {
    return T::begin();
  }
};

int main(int, char**)
{
  // Deducing either fallback's return type is ill-formed.
  static_assert(cuda::std::ranges::empty(with_member_empty<void>{}));
  return 0;
}
