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

TEST_FUNC bool below_limit(int);

using function_reference_result =
  cuda::std::invoke_result_t<decltype(cuda::std::views::take_while), int (&)[3], decltype(below_limit)&>;
static_assert(
  cuda::std::is_same_v<function_reference_result,
                       cuda::std::ranges::take_while_view<cuda::std::ranges::ref_view<int[3]>, bool (*)(int)>>);

int main(int, char**)
{
  return 0;
}
