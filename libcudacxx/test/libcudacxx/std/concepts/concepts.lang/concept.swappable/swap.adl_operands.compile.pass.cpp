//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct first
{};
struct second
{};

TEST_FUNC void swap(first*&, second*&) noexcept;
TEST_FUNC void swap(first (&)[2], second (&)[2]) noexcept;

using swap_fn = decltype(cuda::std::ranges::swap);

// Associated element types do not make a pointer or an array a class or enum operand.
static_assert(!cuda::std::is_invocable_v<swap_fn, first*&, second*&>);
static_assert(!cuda::std::is_invocable_v<swap_fn, first (&)[2], second (&)[2]>);

int main(int, char**)
{
  return 0;
}
