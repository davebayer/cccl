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
// template<class Self, class Visitor>
//   constexpr decltype(auto) visit(this Self&&, Visitor&&);
// template<class R, class Self, class Visitor>
//   constexpr R visit(this Self&&, Visitor&&);

#include <cuda/std/variant>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

struct Visitor1
{
  template <class T>
  __host__ __device__ constexpr void operator()(T&&) const
  {}
};

struct Visitor2
{
  template <class T>
  __host__ __device__ constexpr Holder<T>* operator()(T&&) const
  {
    return nullptr;
  }
};

__host__ __device__ constexpr bool test(bool do_it)
{
  if (do_it)
  {
    cuda::std::variant<Holder<Incomplete>*, int> v = nullptr;

    v.visit(Visitor1{});
    v.visit(Visitor2{});
    v.visit<void>(Visitor1{});
    v.visit<void*>(Visitor2{});
  }
  return true;
}

int main(int, char**)
{
  test(true);
  static_assert(test(true), "");

  return 0;
}
