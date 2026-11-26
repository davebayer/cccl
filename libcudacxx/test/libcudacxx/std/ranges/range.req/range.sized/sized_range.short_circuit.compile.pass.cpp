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
struct non_range
{
  TEST_FUNC auto size() const
  {
    return T::size();
  }
};

int main(int, char**)
{
  static_assert(!cuda::std::ranges::sized_range<non_range<void>>);
  static_assert(cuda::std::ranges::sized_range<int[3]>);
  return 0;
}
