//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::borrowed_subrange_t;

#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/string_view>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING

#include "test_iterators.h"
#include "test_macros.h"

#ifdef _LIBCUDACXX_HAS_STRING
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string>, cuda::std::ranges::dangling>);
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string&&>, cuda::std::ranges::dangling>);
#endif
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::inplace_vector<int, 3>>,
                                 cuda::std::ranges::dangling>);

#ifdef _LIBCUDACXX_HAS_STRING
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string&>,
                                 cuda::std::ranges::subrange<cuda::std::string::iterator>>);
#endif
static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::span<int>>,
                                 cuda::std::ranges::subrange<cuda::std::span<int>::iterator>>);

static_assert(cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<cuda::std::string_view>,
                                 cuda::std::ranges::subrange<cuda::std::string_view::iterator>>);

template <class T, class = void>
constexpr bool has_type = false;

template <class T>
constexpr bool has_type<T, cuda::std::void_t<cuda::std::ranges::borrowed_subrange_t<T>>> = true;

struct move_only_input_range
{
  using iterator = cpp20_input_iterator<int*>;
  TEST_FUNC iterator begin();
  TEST_FUNC sentinel_wrapper<iterator> end();
};

static_assert(cuda::std::ranges::range<move_only_input_range>);
static_assert(has_type<move_only_input_range>);
static_assert(
  cuda::std::same_as<cuda::std::ranges::borrowed_subrange_t<move_only_input_range>, cuda::std::ranges::dangling>);
// This iterator cannot act as its own sentinel in a borrowed subrange.
static_assert(!has_type<move_only_input_range&>);

static_assert(!has_type<int>);

struct S
{};
static_assert(!has_type<S>);

int main(int, char**)
{
  return 0;
}
