//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// explicit operator bool() const noexcept
//
// Note more testing is done in the unit test for:
// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <cuda/std/__format_>
#include <cuda/std/cassert>

template <class CharT>
void test()
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  cuda::std::basic_format_arg<Context> format_arg{};
  static_assert(noexcept(!format_arg));
  assert(!format_arg);
  static_assert(noexcept(static_cast<bool>(format_arg)));
  assert(!static_cast<bool>(format_arg));
}

void test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
