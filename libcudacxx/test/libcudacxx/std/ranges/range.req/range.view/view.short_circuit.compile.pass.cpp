//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/ranges>

#include "test_macros.h"

struct incomplete;

template <class Tag>
struct immovable
{
  TEST_FUNC immovable(immovable&&) = delete;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

template <class Tag>
inline constexpr bool cuda::std::ranges::enable_view<immovable<Tag>> = Tag::is_view;

int main(int, char**)
{
  // An incomplete non-range must not be checked for movability.
  static_assert(!cuda::std::ranges::view<incomplete>);
  // The specialization of enable_view only becomes relevant for movable ranges.
  static_assert(!cuda::std::ranges::view<immovable<void>>);
  return 0;
}
