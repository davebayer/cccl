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
  _CCCL_HOST_DEVICE static constexpr bool check()
  {
    access_buffer<I> buffer{};
    using result = decltype(cuda::std::ranges::cdata(buffer));
    return cuda::std::is_same_v<result, const access_element<I>*> && cuda::std::ranges::cdata(buffer) == buffer.values;
  }

  static constexpr bool value = check();
};

static_assert(instantiate<probe, 32>());
