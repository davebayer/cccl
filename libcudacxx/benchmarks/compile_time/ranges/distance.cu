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

#include "access_buffer.cuh"
#include "instantiate.cuh"

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool check()
  {
    access_buffer<I> buffer{};
    const auto first = cuda::std::make_reverse_iterator(buffer.end());
    const auto last  = cuda::std::make_reverse_iterator(buffer.begin());
    return cuda::std::ranges::distance(first, last) == 4 && cuda::std::ranges::distance(buffer) == 4;
  }

  static constexpr bool value = check();
};

static_assert(instantiate<probe, 32>());
