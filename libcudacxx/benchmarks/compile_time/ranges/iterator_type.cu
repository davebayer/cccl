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

#include "access_buffer.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  using range = access_buffer<I>;
  static constexpr bool value =
    cuda::std::is_same_v<cuda::std::ranges::iterator_t<range>, access_element<I>*>
    && cuda::std::is_same_v<cuda::std::ranges::iterator_t<const range>, const access_element<I>*>;
};

static_assert(instantiate<probe, 32>());
