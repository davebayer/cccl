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
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct element
{};

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    using item = element<I>;
    item values[2]{};
    const item* last = values + 2;
    const cuda::std::ranges::subrange view{values, last};
    using range = decltype(view);
    static_assert(cuda::std::tuple_size_v<range> == 2);
    static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<0, range>, item*>);
    static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<1, range>, const item*>);
#if _CCCL_HAS_CONCEPTS()
    static_assert(cuda::std::is_convertible_v<range, cuda::std::pair<item*, const item*>>);
#endif
    const auto [first, end] = view;
    return first == values && end == last && cuda::std::get<0>(view) == first && cuda::std::ranges::get<1>(view) == end;
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
