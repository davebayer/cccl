//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// basic_format_arg() noexcept;

// The class has several exposition only private constructors. These are tested
// in visit_format_arg.pass.cpp

#include <cuda/std/__format_>
#include <cuda/std/cassert>

template <class CharT>
void test()
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  static_assert(noexcept(cuda::std::basic_format_arg<Context>{}));

  cuda::std::basic_format_arg<Context> format_arg{};
  assert(!format_arg);
}

void test()
{
  test<char>();
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();

  return 0;
}
