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

struct move_only_function
{
  move_only_function()                     = default;
  move_only_function(move_only_function&&) = default;
  TEST_FUNC int operator()(int) const;
};

TEST_FUNC int twice(int);

int main(int, char**)
{
  using transform = decltype(cuda::std::views::transform);
  using array     = int (&)[2];
  static_assert(cuda::std::is_invocable_v<transform, array, move_only_function>);
  static_assert(!cuda::std::is_invocable_v<transform, array, move_only_function&>);
  static_assert(
    cuda::std::is_same_v<cuda::std::invoke_result_t<transform, array, decltype(twice)&>,
                         cuda::std::ranges::transform_view<cuda::std::ranges::ref_view<int[2]>, int (*)(int)>>);
  return 0;
}
