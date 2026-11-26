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

template <class>
struct not_a_range
{};

// A non-range cannot be a viewable range, regardless of its enable_view specialization.
template <class T>
inline constexpr bool cuda::std::ranges::enable_view<not_a_range<T>> = T::enabled;

int main(int, char**)
{
  static_assert(!cuda::std::ranges::viewable_range<not_a_range<void>>);
  static_assert(!cuda::std::ranges::viewable_range<const not_a_range<void>&>);
  return 0;
}
