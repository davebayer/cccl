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

// A const rvalue constructor cannot satisfy the separate const lvalue requirement.
struct CopyCtorRequiresMutableLvalue
{
  CopyCtorRequiresMutableLvalue(CopyCtorRequiresMutableLvalue&) = default;
  TEST_FUNC CopyCtorRequiresMutableLvalue(const CopyCtorRequiresMutableLvalue&&) noexcept;
};

static_assert(cuda::std::move_constructible<CopyCtorRequiresMutableLvalue>);
static_assert(!cuda::std::copy_constructible<CopyCtorRequiresMutableLvalue>);

int main(int, char**)
{
  return 0;
}
