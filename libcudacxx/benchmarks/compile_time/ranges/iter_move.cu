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
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct reverse_zip_cursor
{
  element<I>* first_end;
  int* second_end;

  _CCCL_HOST_DEVICE auto operator*() const
  {
    return cuda::std::tie(*cuda::std::make_reverse_iterator(first_end), *cuda::std::make_reverse_iterator(second_end));
  }

  _CCCL_HOST_DEVICE friend cuda::std::tuple<element<I>&&, int&&> iter_move(reverse_zip_cursor it) noexcept
  {
    return {cuda::std::move(it.first_end[-1]), cuda::std::move(it.second_end[-1])};
  }
};

template <int I>
struct probe
{
  using result                = decltype(cuda::std::ranges::iter_move(cuda::std::declval<reverse_zip_cursor<I>&>()));
  static constexpr bool value = cuda::std::is_same_v<result, cuda::std::tuple<element<I>&&, int&&>>;
};

static_assert(instantiate<probe>());
