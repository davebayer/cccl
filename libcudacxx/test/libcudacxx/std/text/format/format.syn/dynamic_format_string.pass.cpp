//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class charT> struct dynamic-format-string {  // exposition-only
// private:
//   basic_string_view<charT> str;  // exposition-only
//
// public:
//   dynamic-format-string(basic_string_view<charT> s) noexcept : str(s) {}
//
//   dynamic-format-string(const dynamic-format-string&) = delete;
//   dynamic-format-string& operator=(const dynamic-format-string&) = delete;
// };
//
// dynamic-format-string<char> dynamic_format(string_view fmt) noexcept;
// dynamic-format-string<wchar_t> dynamic_format(wstring_view fmt) noexcept;
//
// Additional testing is done in
// - std/text/format/format.functions/format.dynamic_format.pass.cpp

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC void test()
{
  {
    static_assert(noexcept(cuda::std::dynamic_format(cuda::std::string_view{})));
    auto fmt_str = cuda::std::dynamic_format(cuda::std::string_view{});

    using FormatString = decltype(fmt_str);
    static_assert(cuda::std::is_same_v<FormatString, cuda::std::__dynamic_format_string<char>>);

    static_assert(cuda::std::is_nothrow_convertible_v<cuda::std::string_view, FormatString>);
    static_assert(cuda::std::is_nothrow_constructible_v<FormatString, cuda::std::string_view>);

    static_assert(!cuda::std::is_copy_constructible_v<FormatString>);
    static_assert(!cuda::std::is_copy_assignable_v<FormatString>);

    static_assert(!cuda::std::is_move_constructible_v<FormatString>);
    static_assert(!cuda::std::is_move_assignable_v<FormatString>);
  }

#if _CCCL_HAS_WCHAR_T()
  {
    static_assert(noexcept(cuda::std::dynamic_format(cuda::std::wstring_view{})));
    auto fmt_str = cuda::std::dynamic_format(cuda::std::wstring_view{});

    using FormatString = decltype(fmt_str);
    static_assert(cuda::std::is_same_v<FormatString, cuda::std::__dynamic_format_string<wchar_t>>);

    static_assert(cuda::std::is_nothrow_convertible_v<cuda::std::wstring_view, FormatString>);
    static_assert(cuda::std::is_nothrow_constructible_v<FormatString, cuda::std::wstring_view>);

    static_assert(!cuda::std::is_copy_constructible_v<FormatString>);
    static_assert(!cuda::std::is_copy_assignable_v<FormatString>);

    static_assert(!cuda::std::is_move_constructible_v<FormatString>);
    static_assert(!cuda::std::is_move_assignable_v<FormatString>);
  }
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();

  return 0;
}
