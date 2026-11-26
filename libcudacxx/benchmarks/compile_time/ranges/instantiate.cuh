// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/utility>

template <template <int> class Probe, int... Indices>
_CCCL_API constexpr bool instantiate(cuda::std::integer_sequence<int, Indices...>)
{
  return (Probe<Indices>::value && ...);
}

template <template <int> class Probe, int Count = 64>
_CCCL_API constexpr bool instantiate()
{
  return instantiate<Probe>(cuda::std::make_integer_sequence<int, Count>{});
}
