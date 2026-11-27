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
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct customized
{
  TEST_FUNC friend constexpr int iter_move(customized) noexcept
  {
    return 42;
  }
  TEST_FUNC auto operator*() const
  {
    return T::unused;
  }
};

struct ref_qualified
{
  TEST_FUNC int& operator*() &;
  TEST_FUNC long operator*() && noexcept;
};

struct rvalue_only
{
  TEST_FUNC int& operator*() &&;
};

template <class T>
struct custom_result
{
  TEST_FUNC friend int iter_move(custom_result) noexcept
  {
    return T::unused;
  }
};

int main(int, char**)
{
  using move = decltype(cuda::std::ranges::iter_move);
  static_assert(cuda::std::is_nothrow_invocable_v<move, customized<void>>);
  static_assert(cuda::std::ranges::iter_move(customized<void>{}) == 42);
  // Querying the customization result does not require the definition of its body.
  static_assert(cuda::std::is_same_v<cuda::std::invoke_result_t<move, custom_result<void>>, int>);
  static_assert(cuda::std::is_same_v<cuda::std::invoke_result_t<move, ref_qualified>, long>);
  static_assert(cuda::std::is_nothrow_invocable_v<move, ref_qualified>);
  static_assert(cuda::std::is_same_v<cuda::std::invoke_result_t<move, ref_qualified&>, int&&>);
  static_assert(!cuda::std::is_invocable_v<move, rvalue_only&>);
  static_assert(cuda::std::is_same_v<cuda::std::invoke_result_t<move, rvalue_only>, int&&>);
  return 0;
}
