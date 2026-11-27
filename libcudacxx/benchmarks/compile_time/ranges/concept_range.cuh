//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/std/detail/__config>

template <int>
struct range_element
{};

template <class Iterator, class Sentinel = Iterator>
struct cursor_range
{
  _CCCL_HOST_DEVICE Iterator begin() const;
  _CCCL_HOST_DEVICE Sentinel end() const;
};
