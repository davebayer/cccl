//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator,(L, R) noexcept = delete;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "test_macros.h"

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator,(const WithOps& /*l*/, WithOps r)
  {
    return WithOps{r.value};
  }
};

struct NoOps
{};

template <class L, class R, class = void>
inline constexpr bool HasComma = false;
template <class L, class R>
inline constexpr bool HasComma<L, R, cuda::std::void_t<decltype(cuda::std::declval<L>(), cuda::std::declval<R>())>> =
  true;

// template <class L, class R>
// concept HasComma = requires(L l, R r) {
//   { l, r };
// };

// Comma operator is deleted for constant_wrapper operands
static_assert(!HasComma<cuda::std::constant_wrapper<6>, cuda::std::constant_wrapper<3>>);
static_assert(!HasComma<cuda::std::constant_wrapper<WithOps{6}>, cuda::std::constant_wrapper<WithOps{3}>>);
static_assert(!HasComma<cuda::std::constant_wrapper<NoOps{}>, cuda::std::constant_wrapper<NoOps{}>>);

// Mixed operands - one constant_wrapper, one runtime type (uses built-in operator)
static_assert(HasComma<cuda::std::constant_wrapper<42>, int>);
static_assert(HasComma<int, cuda::std::constant_wrapper<42>>);

constexpr bool test()
{
  {
    // only mixed with runtime parameters
    cuda::std::constant_wrapper<42> cw42;
    int i                                           = 0;
    cuda::std::same_as<int&> decltype(auto) result1 = (cw42, i);
    assert(result1 == 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
