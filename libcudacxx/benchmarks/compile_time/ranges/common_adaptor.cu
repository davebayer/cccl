//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  using buffer     = element<I>[3];
  using ptr        = element<I>*;
  using non_common = cuda::std::ranges::subrange<ptr, cuda::std::unreachable_sentinel_t>;
  using adapted    = decltype(cuda::std::views::common(cuda::std::declval<non_common>()));
  static constexpr bool value =
    cuda::std::is_same_v<decltype(cuda::std::views::common(cuda::std::declval<buffer&>())),
                         cuda::std::ranges::ref_view<buffer>>
    && cuda::std::is_same_v<adapted, cuda::std::ranges::common_view<non_common>>
    && cuda::std::is_same_v<decltype(cuda::std::declval<const adapted&>().base()), non_common>;
};

static_assert(instantiate<probe, 32>());
