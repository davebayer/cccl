//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class I>
// using iter_rvalue_reference;

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

static_assert(cuda::std::same_as<cuda::std::iter_rvalue_reference_t<int*>, int&&>);
static_assert(cuda::std::same_as<cuda::std::iter_rvalue_reference_t<const int*>, const int&&>);

template <class T, class = void>
constexpr bool has_rvalue_reference = false;

template <class T>
constexpr bool has_rvalue_reference<T, cuda::std::void_t<cuda::std::iter_rvalue_reference_t<T>>> = true;

template <class T>
struct not_dereferenceable
{
  TEST_FUNC friend auto iter_move(not_dereferenceable)
  {
    return T::unused;
  }
};

struct void_result
{
  TEST_FUNC int operator*() const;
  TEST_FUNC friend void iter_move(void_result);
};

static_assert(!has_rvalue_reference<not_dereferenceable<void>>);
static_assert(!has_rvalue_reference<void_result>);

TEST_FUNC void test_undefined_internal()
{
  struct A
  {
    TEST_FUNC int& operator*() const;
  };
  static_assert(cuda::std::same_as<cuda::std::iter_rvalue_reference_t<A>, int&&>);
}

int main(int, char**)
{
  return 0;
}
