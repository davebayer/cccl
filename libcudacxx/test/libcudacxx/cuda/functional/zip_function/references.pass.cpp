//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class Arg>
struct TakeReference
{
  TEST_FUNC constexpr int operator()(Arg) const noexcept
  {
    return 42;
  }

  TEST_FUNC constexpr int operator()(Arg, Arg) const noexcept
  {
    return 42;
  }
};

template <bool Expected, class Arg, class Tuple>
TEST_FUNC constexpr void test_invocable(Tuple&& tuple)
{
  using Fn = cuda::zip_function<TakeReference<Arg>>;
  static_assert(cuda::std::is_invocable_v<Fn&, Tuple> == Expected);
  static_assert(cuda::std::is_invocable_v<const Fn&, Tuple> == Expected);
  if constexpr (Expected)
  {
    Fn fn{};
    const Fn const_fn{};
    static_assert(cuda::std::is_nothrow_invocable_v<Fn&, Tuple>);
    static_assert(cuda::std::is_nothrow_invocable_v<const Fn&, Tuple>);
    assert(fn(cuda::std::forward<Tuple>(tuple)) == 42);
    assert(const_fn(cuda::std::forward<Tuple>(tuple)) == 42);
  }
}

template <class Tuple>
TEST_FUNC constexpr void test_cvref()
{
  using Value = cuda::std::tuple_element_t<0, Tuple>;
  Tuple tuple{1};
  test_invocable<true, Value&>(tuple);
  test_invocable<true, const Value&>(tuple);
  test_invocable<false, Value&&>(tuple);
  test_invocable<false, const Value&&>(tuple);

  test_invocable<false, Value&>(cuda::std::as_const(tuple));
  test_invocable<true, const Value&>(cuda::std::as_const(tuple));
  test_invocable<false, Value&&>(cuda::std::as_const(tuple));
  test_invocable<false, const Value&&>(cuda::std::as_const(tuple));

  test_invocable<false, Value&>(cuda::std::move(tuple));
  test_invocable<true, const Value&>(cuda::std::move(tuple));
  test_invocable<true, Value&&>(cuda::std::move(tuple));
  test_invocable<true, const Value&&>(cuda::std::move(tuple));

  test_invocable<false, Value&>(cuda::std::move(cuda::std::as_const(tuple)));
  test_invocable<true, const Value&>(cuda::std::move(cuda::std::as_const(tuple)));
  test_invocable<false, Value&&>(cuda::std::move(cuda::std::as_const(tuple)));
  test_invocable<true, const Value&&>(cuda::std::move(cuda::std::as_const(tuple)));
}

struct Mutate
{
  TEST_FUNC constexpr int operator()(int& a, float& b) const noexcept
  {
    ++a;
    b += 1.0f;
    return a;
  }
};

struct MutableMutate
{
  TEST_FUNC constexpr int operator()(int& a, float& b)
  {
    return Mutate{}(a, b);
  }
};

template <class Tuple>
TEST_FUNC constexpr void test_mutation()
{
  Tuple tuple{40, 1.0f};
  cuda::zip_function<MutableMutate> fn{};
  const cuda::zip_function<Mutate> const_fn{};
  static_assert(!cuda::std::is_invocable_v<const decltype(fn)&, Tuple&>);
  assert(fn(tuple) == 41);
  assert(const_fn(tuple) == 42);
  assert(cuda::std::get<0>(tuple) == 42);
  assert(cuda::std::get<1>(tuple) == 3.0f);

  // zip_iterator dereferences produce prvalue tuples of references.
  assert(const_fn(cuda::std::tie(cuda::std::get<0>(tuple), cuda::std::get<1>(tuple))) == 43);
  assert(cuda::std::get<0>(tuple) == 43);
  assert(cuda::std::get<1>(tuple) == 4.0f);
}

TEST_FUNC constexpr bool test()
{
  test_cvref<cuda::std::tuple<int>>();
  test_cvref<cuda::std::array<int, 1>>();
  test_cvref<cuda::std::complex<float>>();
  test_mutation<cuda::std::tuple<int, float>>();
  test_mutation<cuda::std::pair<int, float>>();
  static_assert(!cuda::std::is_invocable_v<cuda::zip_function<TakeReference<int&>>&, int&>);
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
