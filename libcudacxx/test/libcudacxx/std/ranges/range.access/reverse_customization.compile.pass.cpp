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

struct MemberReverse
{
  TEST_FUNC int* rbegin() const;
  TEST_FUNC int* rend() const;
};

template <class T = void>
TEST_FUNC auto rbegin(MemberReverse&)
{
  return T::unreachable;
}

template <class T = void>
TEST_FUNC auto rend(MemberReverse&)
{
  return T::unreachable;
}

struct AdlReverse
{
  template <class T = void>
  TEST_FUNC auto begin() const
  {
    return T::unreachable;
  }
};

TEST_FUNC int* rbegin(AdlReverse&);
TEST_FUNC int* rend(AdlReverse&);

struct NonBorrowed
{
  template <class T = void>
  TEST_FUNC auto rbegin() const
  {
    return T::unreachable;
  }

  template <class T = void>
  TEST_FUNC auto rend() const
  {
    return T::unreachable;
  }
};

using RBegin = decltype(cuda::std::ranges::rbegin);
using REnd   = decltype(cuda::std::ranges::rend);
static_assert(cuda::std::is_invocable_r_v<int*, RBegin, MemberReverse&>);
static_assert(cuda::std::is_invocable_r_v<int*, REnd, MemberReverse&>);
static_assert(cuda::std::is_invocable_r_v<int*, RBegin, AdlReverse&>);
static_assert(cuda::std::is_invocable_r_v<int*, REnd, AdlReverse&>);
static_assert(!cuda::std::is_invocable_v<RBegin, NonBorrowed>);
static_assert(!cuda::std::is_invocable_v<REnd, NonBorrowed>);

int main(int, char**)
{
  return 0;
}
