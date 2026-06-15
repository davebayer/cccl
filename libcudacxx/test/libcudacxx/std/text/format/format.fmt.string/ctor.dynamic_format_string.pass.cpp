//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class charT, class... Args>
// class basic_format_string<charT, type_identity_t<Args>...>
//
// basic_format_string(dynamic-format-string<charT> s) noexcept : str(s.str) {}
//
// Additional testing is done in
// - std/text/format/format.functions/format.dynamic_format.pass.cpp

#include <cuda/std/__format_>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_FUNC void test()
{
  static_assert(noexcept(cuda::std::format_string<>{cuda::std::dynamic_format(cuda::std::string_view{})}));
  {
    constexpr const auto& str          = "}{invalid format string}{";
    cuda::std::format_string<> fmt_str = cuda::std::dynamic_format(str);
    assert(fmt_str.get() == str);
  }

#if _CCCL_HAS_WCHAR_T()
  static_assert(noexcept(cuda::std::wformat_string<>{cuda::std::dynamic_format(cuda::std::wstring_view{})}));
  {
    constexpr const auto& str           = L"}{invalid format string}{";
    cuda::std::wformat_string<> fmt_str = cuda::std::dynamic_format(str);
    assert(fmt_str.get() == str);
  }
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();

  return 0;
}
