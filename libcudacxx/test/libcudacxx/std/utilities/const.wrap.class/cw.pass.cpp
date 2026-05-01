//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// constant_wrapper

//   template<cw-fixed-value X>
//    constexpr auto cw = constant_wrapper<X>{};

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "test_macros.h"

struct S
{
  int value;

  TEST_FUNC constexpr S(int v)
      : value(v)
  {}

  TEST_FUNC constexpr bool operator==(const S& other) const
  {
    return value == other.value;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    // int constant
    cuda::std::same_as<const cuda::std::constant_wrapper<42>> decltype(auto) cw_val = cuda::std::cw<42>;
    static_assert(cw_val == 42);
  }

  {
    // struct constant
    constexpr S s{13};
    cuda::std::same_as<const cuda::std::constant_wrapper<s>> decltype(auto) cw_val = cuda::std::cw<s>;
    static_assert(cw_val == s);
  }

  {
    // array constant
    constexpr int arr[] = {1, 2, 3};
    // gcc complains that cw_val is unused
    [[maybe_unused]] cuda::std::same_as<const cuda::std::constant_wrapper<arr>> decltype(auto) cw_val =
      cuda::std::cw<arr>;
    static_assert(cw_val[0] == 1);
    static_assert(cw_val[1] == 2);
    static_assert(cw_val[2] == 3);
  }

  {
    // string literals
    [[maybe_unused]] cuda::std::same_as<const cuda::std::constant_wrapper<"hello">> decltype(auto) cw_val =
      cuda::std::cw<"hello">;
    static_assert(cw_val[0] == 'h');
    static_assert(cw_val[1] == 'e');
    static_assert(cw_val[2] == 'l');
    static_assert(cw_val[3] == 'l');
    static_assert(cw_val[4] == 'o');
    static_assert(cw_val[5] == '\0');
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
