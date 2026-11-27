//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/iterator>

struct incomplete;

struct unsigned_difference
{};

namespace cuda::std
{
template <>
struct iterator_traits<unsigned_difference*>
{
  using difference_type = unsigned;
};
} // namespace cuda::std

template <class Pointer, bool Expected>
constexpr bool check_pointer()
{
  static_assert(cuda::std::weakly_incrementable<Pointer> == Expected);
  static_assert(cuda::std::incrementable<Pointer> == Expected);
  static_assert(cuda::std::input_or_output_iterator<Pointer> == Expected);
  static_assert(cuda::std::sentinel_for<Pointer, Pointer> == Expected);
  return true;
}

static_assert(check_pointer<const volatile int*, true>());
static_assert(check_pointer<int (*)[3], true>());
static_assert(check_pointer<int* const, false>());
static_assert(check_pointer<void*, false>());
static_assert(check_pointer<const volatile void*, false>());
static_assert(check_pointer<int (*)(), false>());
static_assert(check_pointer<incomplete*, false>());
static_assert(check_pointer<int (*)[], false>());
static_assert(check_pointer<unsigned_difference*, false>());

int main(int, char**)
{
  return 0;
}
