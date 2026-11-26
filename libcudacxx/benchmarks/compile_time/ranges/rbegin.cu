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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "instantiate.cuh"

template <int>
struct Element
{};

// A range exposing a forward sequence through reverse iterators can provide its own rbegin().
template <int Id>
struct Range
{
  using Iterator = cuda::std::reverse_iterator<Element<Id>*>;

  _CCCL_HOST_DEVICE Iterator begin() const;
  _CCCL_HOST_DEVICE Iterator end() const;
  _CCCL_HOST_DEVICE Element<Id>* rbegin() const;
};

template <int Id>
struct Probe : cuda::std::is_same<decltype(cuda::std::ranges::rbegin(cuda::std::declval<Range<Id>&>())), Element<Id>*>
{};

static_assert(instantiate<Probe>());
