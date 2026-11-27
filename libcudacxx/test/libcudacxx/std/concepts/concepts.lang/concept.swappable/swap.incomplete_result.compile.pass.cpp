//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/concepts>

#include "test_macros.h"

struct incomplete_result;
struct nonmovable
{
  nonmovable(nonmovable&&) = delete;
  friend TEST_FUNC incomplete_result swap(nonmovable&, nonmovable&);
};

// The ADL expression must also permit discarding its result.
static_assert(!cuda::std::swappable<nonmovable>);

int main(int, char**)
{
  return 0;
}
