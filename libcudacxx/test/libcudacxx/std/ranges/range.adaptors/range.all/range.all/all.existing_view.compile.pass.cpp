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
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
struct range
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

#if TEST_STD_VER < 2020
// Selecting the decay-copy branch of views::all does not need to instantiate either swap.
template <class T, cuda::std::enable_if_t<cuda::std::is_void_v<T>, int> = 0>
TEST_FUNC auto swap(cuda::std::ranges::ref_view<range<T>>&, cuda::std::ranges::ref_view<range<T>>&)
{
  return T::unused;
}

template <class T, cuda::std::enable_if_t<cuda::std::is_void_v<T>, int> = 0>
TEST_FUNC auto swap(cuda::std::ranges::owning_view<range<T>>&, cuda::std::ranges::owning_view<range<T>>&)
{
  return T::unused;
}

#endif // TEST_STD_VER < 2020

namespace ranges = cuda::std::ranges;

template <>
class ranges::ref_view<range<int>>;

template <>
class ranges::owning_view<range<int>>
    : public ranges::view_base
    , public ::range<int>
{
public:
  owning_view& operator=(const owning_view&) = delete;
};

template <>
inline constexpr bool ranges::enable_view<ranges::owning_view<range<char>>> = false;

struct derived : ranges::ref_view<range<long>>
{
  derived& operator=(const derived&) = delete;
};

#if TEST_STD_VER >= 2020
using all_fn = cuda::std::remove_cv_t<decltype(cuda::std::views::all)>;

struct lvalue_view_overload : all_fn
{
  using all_fn::operator();

  template <class T>
    requires ranges::view<cuda::std::decay_t<T>> && cuda::std::is_lvalue_reference_v<T>
  TEST_FUNC int operator()(T&&) const;
};

#endif // TEST_STD_VER >= 2020

int main(int, char**)
{
  using ref   = cuda::std::ranges::ref_view<range<void>>;
  using owner = cuda::std::ranges::owning_view<range<void>>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::views::all(cuda::std::declval<const ref&>())), ref>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::views::all(cuda::std::declval<owner>())), owner>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::all), owner&>);
  using all = decltype(cuda::std::views::all);
  static_assert(!cuda::std::is_invocable_v<all, ranges::ref_view<range<int>>&>);
  using specialized = ranges::owning_view<range<int>>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::views::all(cuda::std::declval<specialized&>())),
                                     ranges::ref_view<specialized>>);

  using opted_out = ranges::owning_view<range<char>>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::views::all(cuda::std::declval<opted_out&>())),
                                     ranges::ref_view<opted_out>>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::views::all(cuda::std::declval<derived&>())), ranges::ref_view<derived>>);

#if TEST_STD_VER >= 2020
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::declval<const lvalue_view_overload&>()(cuda::std::declval<ref&>())), int>);
#endif // TEST_STD_VER >= 2020
  return 0;
}
