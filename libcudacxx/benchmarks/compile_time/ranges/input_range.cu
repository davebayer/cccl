//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "concept_range.cuh"
#include "instantiate.cuh"

template <int I>
struct output_cursor
{
  using difference_type = cuda::std::ptrdiff_t;
  _CCCL_HOST_DEVICE output_cursor& operator*();
  _CCCL_HOST_DEVICE output_cursor& operator++();
  _CCCL_HOST_DEVICE output_cursor operator++(int);
  _CCCL_HOST_DEVICE output_cursor& operator=(range_element<I>);
};

template <int I>
struct probe
{
  using sink = cursor_range<output_cursor<I>, cuda::std::unreachable_sentinel_t>;
  static constexpr bool value =
    cuda::std::ranges::input_range<range_element<I>[4]> && !cuda::std::ranges::input_range<sink>;
};

static_assert(instantiate<probe, 32>());
