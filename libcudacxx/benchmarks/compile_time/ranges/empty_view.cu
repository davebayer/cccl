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

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  using view                  = cuda::std::remove_cv_t<decltype(cuda::std::views::empty<element<I>>)>;
  static constexpr bool value = cuda::std::is_same_v<view, cuda::std::ranges::empty_view<element<I>>> && view::empty()
                             && view::size() == 0 && view::begin() == view::end() && view::data() == nullptr;
};

static_assert(instantiate<probe>());
