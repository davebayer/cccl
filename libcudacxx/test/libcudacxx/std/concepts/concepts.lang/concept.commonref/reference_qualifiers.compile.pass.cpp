//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/concepts>

using cuda::std::common_reference_with;

struct immovable
{
  immovable(const immovable&) = delete;
};

static_assert(common_reference_with<immovable&, immovable&&>);
static_assert(common_reference_with<const immovable&&, immovable&>);
static_assert(common_reference_with<volatile immovable&, const volatile immovable&>);
static_assert(!common_reference_with<volatile immovable&, immovable&&>);
static_assert(!common_reference_with<const volatile immovable&&, const immovable&>);

struct incomplete;
static_assert(common_reference_with<const incomplete&, incomplete&&>);
static_assert(common_reference_with<int (&)(), int (&&)()>);
static_assert(common_reference_with<const int (&)[2], int (&&)[2]>);

int main(int, char**)
{
  return 0;
}
