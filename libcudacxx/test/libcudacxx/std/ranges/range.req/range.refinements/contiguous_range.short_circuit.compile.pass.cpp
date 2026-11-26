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

template <class T>
struct reversed
{
  TEST_FUNC cuda::std::reverse_iterator<int*> begin() const;
  TEST_FUNC cuda::std::reverse_iterator<int*> end() const;

  TEST_FUNC auto data() const
  {
    return T::data();
  }
};

int main(int, char**)
{
  static_assert(cuda::std::ranges::random_access_range<reversed<void>>);
  static_assert(!cuda::std::ranges::contiguous_range<reversed<void>>);
  return 0;
}
