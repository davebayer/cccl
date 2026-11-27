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
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct view : cuda::std::ranges::view_base
{
  _CCCL_HOST_DEVICE element<I>* begin() const;
  _CCCL_HOST_DEVICE element<I>* end() const;
  _CCCL_HOST_DEVICE unsigned size() const noexcept;
};

template <int I>
struct probe
{
  using zip                   = cuda::std::ranges::zip_view<view<I>, view<I + 1>, view<I + 2>, view<I + 3>>;
  static constexpr bool value = cuda::std::is_same_v<decltype(cuda::std::declval<zip&>().size()), unsigned>
                             && noexcept(cuda::std::declval<const zip&>().size());
};

static_assert(instantiate<probe, 48>());
