//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

// <cuda/std/variant>

// class variant;

// template<class R, class Self, class Visitor>
//   constexpr R visit(this Self&&, Visitor&&);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include <string>

#include "test_macros.h"
#include "variant_test_helpers.h"

template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

void test_overload_ambiguity()
{
  using V = cuda::std::variant<float, long, std::string>;
  V v{"baba"};

  v.visit(overloaded{[]([[maybe_unused]] auto x) {
                       assert(false);
                     },
                     [](const std::string& x) {
                       assert(x == "baba");
                     }});
  assert(cuda::std::get<std::string>(v) == "baba");

  // Test the constraint.
  v = cuda::std::move(v).visit<V>(overloaded{
    []([[maybe_unused]] auto x) {
      assert(false);
      return 0;
    },
    [](const std::string& x) {
      assert(x == "baba");
      return x + " zmt";
    }});
  assert(cuda::std::get<std::string>(v) == "baba zmt");
}

template <typename ReturnType>
void test_call_operator_forwarding()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;

  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    V v(42);

    v.visit<ReturnType>(obj);
    assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
    v.visit<ReturnType>(cobj);
    assert(Fn::check_call<int&>(CT_Const | CT_LValue));
    v.visit<ReturnType>(cuda::std::move(obj));
    assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
    v.visit<ReturnType>(cuda::std::move(cobj));
    assert(Fn::check_call<int&>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    V v(42L);

    v.visit<ReturnType>(obj);
    assert(Fn::check_call<long&>(CT_NonConst | CT_LValue));
    v.visit<ReturnType>(cobj);
    assert(Fn::check_call<long&>(CT_Const | CT_LValue));
    v.visit<ReturnType>(cuda::std::move(obj));
    assert(Fn::check_call<long&>(CT_NonConst | CT_RValue));
    v.visit<ReturnType>(cuda::std::move(cobj));
    assert(Fn::check_call<long&>(CT_Const | CT_RValue));
  }
}

template <typename ReturnType>
void test_argument_forwarding()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const auto val = CT_LValue | CT_NonConst;

  { // single argument - value type
    using V = cuda::std::variant<int>;
    V v(42);
    const V& cv = v;

    v.visit<ReturnType>(obj);
    assert(Fn::check_call<int&>(val));
    cv.visit<ReturnType>(obj);
    assert(Fn::check_call<const int&>(val));
    cuda::std::move(v).visit<ReturnType>(obj);
    assert(Fn::check_call<int&&>(val));
    cuda::std::move(cv).visit<ReturnType>(obj);
    assert(Fn::check_call<const int&&>(val));
  }
}

template <typename ReturnType>
void test_return_type()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;

  { // test call operator forwarding - no variant
    // non-member
    {
      static_assert(cuda::std::is_same<decltype(cuda::std::visit<ReturnType>(obj)), ReturnType>::value, "");
      static_assert(cuda::std::is_same<decltype(cuda::std::visit<ReturnType>(cobj)), ReturnType>::value, "");
      static_assert(cuda::std::is_same<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj))), ReturnType>::value,
                    "");
      static_assert(
        cuda::std::is_same<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj))), ReturnType>::value, "");
    }
  }
  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    V v(42);

    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(obj)), ReturnType>::value, "");
    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(cobj)), ReturnType>::value, "");
    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(cuda::std::move(obj))), ReturnType>::value, "");
    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(cuda::std::move(cobj))), ReturnType>::value, "");
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    V v(42L);

    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(obj)), ReturnType>::value, "");
    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(cobj)), ReturnType>::value, "");
    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(cuda::std::move(obj))), ReturnType>::value, "");
    static_assert(cuda::std::is_same<decltype(v.visit<ReturnType>(cuda::std::move(cobj))), ReturnType>::value, "");
  }
}

void test_constexpr_void()
{
  constexpr ReturnFirst obj{};

  {
    using V = cuda::std::variant<int>;
    constexpr V v(42);

    static_assert((v.visit<void>(obj), 42) == 42, "");
  }
  {
    using V = cuda::std::variant<short, long, char>;
    constexpr V v(42L);

    static_assert((v.visit<void>(obj), 42) == 42, "");
  }
}

void test_constexpr_int()
{
  constexpr ReturnFirst obj{};

  {
    using V = cuda::std::variant<int>;
    constexpr V v(42);

    static_assert(v.visit<int>(obj) == 42, "");
  }
  {
    using V = cuda::std::variant<short, long, char>;
    constexpr V v(42L);

    static_assert(v.visit<int>(obj) == 42, "");
  }
}

template <typename ReturnType>
void test_exceptions()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  ReturnArity obj{};

  auto test = [&](auto&& v) {
    try
    {
      v.template visit<ReturnType>(obj);
    }
    catch (const cuda::std::bad_variant_access&)
    {
      return true;
    }
    catch (...)
    {}
    return false;
  };

  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v;
    makeEmpty(v);

    assert(test(v));
  }
#endif
}

// See https://bugs.llvm.org/show_bug.cgi?id=31916
template <typename ReturnType>
void test_caller_accepts_nonconst()
{
  struct A
  {};
  struct Visitor
  {
    auto operator()(A&)
    {
      if constexpr (!cuda::std::is_void_v<ReturnType>)
      {
        return ReturnType{};
      }
    }
  };
  cuda::std::variant<A> v;

  v.template visit<ReturnType>(Visitor{});
}

void test_constexpr_explicit_side_effect()
{
  auto test_lambda = [](int arg) constexpr {
    cuda::std::variant<int> v = 101;

    {
      v.template visit<void>([arg](int& x) constexpr {
        x = arg;
      });
    }

    return cuda::std::get<int>(v);
  };

  static_assert(test_lambda(202) == 202, "");
}

void test_derived_from_variant()
{
  struct MyVariant : cuda::std::variant<short, long, float>
  {};

  MyVariant{42}.template visit<bool>([](auto x) {
    assert(x == 42);
    return true;
  });
  MyVariant{-1.3f}.template visit<bool>([](auto x) {
    assert(x == -1.3f);
    return true;
  });

  // Check that visit does not take index nor valueless_by_exception members from the base class.
  struct EvilVariantBase
  {
    int index;
    char valueless_by_exception;
  };

  struct EvilVariant1
      : cuda::std::variant<int, long, double>
      , cuda::std::tuple<int>
      , EvilVariantBase
  {
    using cuda::std::variant<int, long, double>::variant;
  };

  EvilVariant1{12}.template visit<bool>([](auto x) {
    assert(x == 12);
    return true;
  });
  EvilVariant1{12.3}.template visit<bool>([](auto x) {
    assert(x == 12.3);
    return true;
  });

  // Check that visit unambiguously picks the variant, even if the other base has __impl member.
  struct ImplVariantBase
  {
    struct Callable
    {
      bool operator()() const
      {
        assert(false);
        return false;
      }
    };

    Callable __impl;
  };

  struct EvilVariant2
      : cuda::std::variant<int, long, double>
      , ImplVariantBase
  {
    using cuda::std::variant<int, long, double>::variant;
  };

  EvilVariant2{12}.template visit<bool>([](auto x) {
    assert(x == 12);
    return true;
  });
  EvilVariant2{12.3}.template visit<bool>([](auto x) {
    assert(x == 12.3);
    return true;
  });
}

int main(int, char**)
{
  test_overload_ambiguity();
  test_call_operator_forwarding<void>();
  test_argument_forwarding<void>();
  test_return_type<void>();
  test_constexpr_void();
  test_exceptions<void>();
  test_caller_accepts_nonconst<void>();
  test_call_operator_forwarding<int>();
  test_argument_forwarding<int>();
  test_return_type<int>();
  test_constexpr_int();
  test_exceptions<int>();
  test_caller_accepts_nonconst<int>();
  test_constexpr_explicit_side_effect();
  test_derived_from_variant();

  return 0;
}
