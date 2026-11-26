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

template <int Id>
struct Range
{
  _CCCL_HOST_DEVICE Element<Id>* rbegin() const;
  _CCCL_HOST_DEVICE Element<Id>* rend() const;
};

// An ADL customization competes with the selected member rend().
template <int Id>
_CCCL_HOST_DEVICE cuda::std::reverse_iterator<Element<Id>*> rend(const Range<Id>&);

template <int Id>
struct Probe : cuda::std::is_same<decltype(cuda::std::ranges::rend(cuda::std::declval<Range<Id>&>())), Element<Id>*>
{};

static_assert(instantiate<Probe>());
