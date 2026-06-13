//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// struct format_to_n_result

// [format.syn]/1
// The class template format_to_n_result has the template parameters, data
// members, and special members specified above. It has no base classes or
// members other than those specified.

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_macros.h"

template <class CharT>
TEST_FUNC constexpr void test()
{
  cuda::std::format_to_n_result<CharT*> v{nullptr, cuda::std::iter_difference_t<CharT*>{42}};

  auto [out, size] = v;
  static_assert(cuda::std::same_as<decltype(out), CharT*>);
  assert(out == v.out);
  static_assert(cuda::std::same_as<decltype(size), cuda::std::iter_difference_t<CharT*>>);
  assert(size == v.size);
}

TEST_FUNC constexpr bool test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
