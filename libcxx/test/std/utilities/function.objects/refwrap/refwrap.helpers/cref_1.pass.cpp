//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(const T& t);

#include <functional>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test()
{
  int i = 0;
  std::reference_wrapper<const int> r = std::cref(i);
  assert(&r.get() == &i);
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
