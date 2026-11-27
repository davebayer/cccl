//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<class T>
// using iterator_t = decltype(ranges::begin(declval<T&>()));

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "test_range.h"

static_assert(
  cuda::std::same_as<cuda::std::ranges::iterator_t<test_range<cpp17_input_iterator>>, cpp17_input_iterator<int*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_range<cpp17_input_iterator> const>,
                                 cpp17_input_iterator<int const*>>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_non_const_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_common_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_common_range<cpp17_input_iterator> const>,
                                 cpp17_input_iterator<int const*>>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_non_const_common_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>);

// iterator_t only needs a valid begin on an lvalue, even if the range argument is an rvalue reference.
struct BeginOnly
{
  TEST_FUNC int* begin() &;
  TEST_FUNC const int* begin() const&;
};

template <class T, class = void>
inline constexpr bool has_iterator = false;

template <class T>
inline constexpr bool has_iterator<T, cuda::std::void_t<cuda::std::ranges::iterator_t<T>>> = true;

static_assert(!cuda::std::ranges::range<BeginOnly>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<BeginOnly&&>, int*>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<const BeginOnly&>, const int*>);
static_assert(!has_iterator<volatile BeginOnly&>);
static_assert(!has_iterator<void>);

int main(int, char**)
{
  return 0;
}
