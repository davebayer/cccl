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
struct access_element
{};

template <int I>
struct access_buffer
{
  access_element<I> values[4]{};

  _CCCL_HOST_DEVICE_API constexpr access_element<I>* begin() noexcept
  {
    return values;
  }
  _CCCL_HOST_DEVICE_API constexpr const access_element<I>* begin() const noexcept
  {
    return values;
  }
  _CCCL_HOST_DEVICE_API constexpr access_element<I>* end() noexcept
  {
    return values + 4;
  }
  _CCCL_HOST_DEVICE_API constexpr const access_element<I>* end() const noexcept
  {
    return values + 4;
  }
};
