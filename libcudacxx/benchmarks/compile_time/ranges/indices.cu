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
#include <cuda/std/type_traits>

template <class T>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr bool check_indices()
{
  const auto range = cuda::std::views::indices(T{4});
  return range.size() == 4 && *range.begin() == T{0} && range.begin()[3] == T{3} && range.end() - range.begin() == 4;
}

template <class... Integers>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr bool check_integer_types()
{
  return (check_indices<Integers>() && ...);
}

static_assert(check_integer_types<int, unsigned int, long, unsigned long, long long, unsigned long long>());
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::indices), float>);
