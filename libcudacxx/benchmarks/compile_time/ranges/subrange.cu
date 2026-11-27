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

template <class T>
struct sentinel
{
  T* last;
  _CCCL_HOST_DEVICE friend constexpr bool operator==(T* it, sentinel s) noexcept
  {
    return it == s.last;
  }
  _CCCL_HOST_DEVICE friend constexpr bool operator==(sentinel s, T* it) noexcept
  {
    return it == s;
  }
  _CCCL_HOST_DEVICE friend constexpr bool operator!=(T* it, sentinel s) noexcept
  {
    return !(it == s);
  }
  _CCCL_HOST_DEVICE friend constexpr bool operator!=(sentinel s, T* it) noexcept
  {
    return !(s == it);
  }
};

template <int I>
struct probe
{
  _CCCL_HOST_DEVICE static constexpr bool test()
  {
    using item = element<I>;
    using kind = cuda::std::ranges::subrange_kind;
    item values[4]{};
    const sentinel<item> last{values + 4};
    const cuda::std::ranges::subrange unsized{values, last};
    cuda::std::ranges::subrange sized{values, last, 4};
    static_assert(
      cuda::std::is_same_v<decltype(unsized), const cuda::std::ranges::subrange<item*, sentinel<item>, kind::unsized>>);
    static_assert(
      cuda::std::is_same_v<decltype(sized), cuda::std::ranges::subrange<item*, sentinel<item>, kind::sized>>);
    sized.advance(2);
    return sized.size() == 2 && sized.prev().begin() == values + 1 && sized.next(2).empty()
        && unsized.next(4).begin() == unsized.end();
  }

  static constexpr bool value = test();
};

static_assert(instantiate<probe, 24>());
