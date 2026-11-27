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

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct sequence : cuda::std::ranges::view_interface<sequence<I>>
{
  _CCCL_HOST_DEVICE range_element<I>* begin() const;
  _CCCL_HOST_DEVICE range_element<I>* end() const;
};

template <int I>
struct probe
{
  static constexpr bool value =
    cuda::std::ranges::enable_view<sequence<I>> && !cuda::std::ranges::enable_view<range_element<I>[4]>;
};

static_assert(instantiate<probe, 32>());
