//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// constant_wrapper

// static constexpr const auto & value = X.data;
// using type = constant_wrapper;
// using value_type = decltype(X)::type;

#include <cuda/std/algorithm>
#include <cuda/std/concepts>
#include <cuda/std/utility>

static_assert(cuda::std::constant_wrapper<42>::value == 42);
static_assert(cuda::std::same_as<decltype(cuda::std::constant_wrapper<42>::value), const int&>);
static_assert(cuda::std::same_as<cuda::std::constant_wrapper<42>::type, cuda::std::constant_wrapper<42>>);
static_assert(cuda::std::same_as<cuda::std::constant_wrapper<42>::value_type, int>);

struct S
{
  int member = 42;
};

static_assert(cuda::std::constant_wrapper<S{5}>::value.member == 5);
static_assert(cuda::std::same_as<decltype(cuda::std::constant_wrapper<S{5}>::value), const S&>);
static_assert(cuda::std::same_as<cuda::std::constant_wrapper<S{5}>::type, cuda::std::constant_wrapper<S{5}>>);
static_assert(cuda::std::same_as<cuda::std::constant_wrapper<S{5}>::value_type, S>);

static_assert(cuda::std::ranges::equal(cuda::std::constant_wrapper<"abcd">::value, "abcd"));
static_assert(cuda::std::same_as<decltype(cuda::std::constant_wrapper<"abcd">::value), const char (&)[5]>);
static_assert(cuda::std::same_as<cuda::std::constant_wrapper<"abcd">::type, cuda::std::constant_wrapper<"abcd">>);
static_assert(cuda::std::same_as<cuda::std::constant_wrapper<"abcd">::value_type, const char[5]>);

int main(int, char**)
{
  return 0;
}
