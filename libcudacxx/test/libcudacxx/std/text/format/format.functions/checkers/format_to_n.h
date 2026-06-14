//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__format_>
#include <cuda/std/algorithm>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/string_view>

#include "format_functions_common.h"
#include "test_macros.h"

// Marking checkers with _CCCL_NOINLINE greatly improves ptxas compile times.

auto test = []<class CharT, class... Args>(
              std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) constexpr {
  {
    std::list<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 0, fmt, std::forward<Args>(args)...);
    // To avoid signedness warnings make sure formatted_size uses the same type
    // as result.size.
    using diff_type          = decltype(result.size);
    diff_type formatted_size = std::formatted_size(fmt, std::forward<Args>(args)...);

    assert(result.size == formatted_size);
    assert(out.empty());
  }
  {
    std::vector<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 5, fmt, std::forward<Args>(args)...);
    using diff_type                = decltype(result.size);
    diff_type formatted_size       = std::formatted_size(fmt, std::forward<Args>(args)...);
    diff_type size                 = std::min<diff_type>(5, formatted_size);

    assert(result.size == formatted_size);
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.begin() + size));
  }
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result = std::format_to_n(std::back_inserter(out), 1000, fmt, std::forward<Args>(args)...);
    using diff_type                = decltype(result.size);
    diff_type formatted_size       = std::formatted_size(fmt, std::forward<Args>(args)...);
    diff_type size                 = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  {
    // Test the returned iterator.
    std::basic_string<CharT> out(10, CharT(' '));
    std::format_to_n_result result = std::format_to_n(out.begin(), 10, fmt, std::forward<Args>(args)...);
    using diff_type                = decltype(result.size);
    diff_type formatted_size       = std::formatted_size(fmt, std::forward<Args>(args)...);
    diff_type size                 = std::min<diff_type>(10, formatted_size);

    assert(result.size == formatted_size);
    assert(result.out == out.begin() + size);
    assert(out.substr(0, size) == expected.substr(0, size));
  }
  {
    static_assert(std::is_signed_v<std::iter_difference_t<CharT*>>,
                  "If the difference type isn't negative the test will fail "
                  "due to using a large positive value.");
    CharT buffer[1]                = {CharT(0)};
    std::format_to_n_result result = std::format_to_n(buffer, -1, fmt, std::forward<Args>(args)...);
    using diff_type                = decltype(result.size);
    diff_type formatted_size       = std::formatted_size(fmt, std::forward<Args>(args)...);

    assert(result.size == formatted_size);
    assert(result.out == buffer);
    assert(buffer[0] == CharT(0));
  }
};

template <class CharT, class... Args>
TEST_FUNC _CCCL_NOINLINE bool check_exception(cuda::std::string_view, cuda::std::basic_string_view<CharT>, Args&&...) {
  // After P2216 most exceptions thrown by std::format_to_n become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in format_to_n.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
};
