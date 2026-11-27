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

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  using item                  = range_element<I>;
  using moving                = cursor_range<cuda::std::move_iterator<item*>>;
  static constexpr bool value = cuda::std::is_same_v<cuda::std::ranges::range_reference_t<const item[4]>, const item&>
                             && cuda::std::is_same_v<cuda::std::ranges::range_reference_t<moving>, item&&>;
};

static_assert(instantiate<probe, 32>());
