//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

// A formatter can inspect other arguments via the standard format_context and delegate to a built-in formatter.

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct inspect_argument
{};

struct is_promoted_int
{
  template <class T>
  TEST_HOST_DEVICE_FUNC bool operator()(T value) const
  {
    if constexpr (cuda::std::is_same_v<T, int>)
    {
      return value == 42;
    }
    else
    {
      return false;
    }
  }
};

template <>
struct cuda::std::formatter<inspect_argument, char> : cuda::std::formatter<bool, char>
{
  cuda::std::size_t id{};

  TEST_HOST_DEVICE_FUNC constexpr auto parse(cuda::std::format_parse_context& ctx)
  {
    id = ctx.next_arg_id();
    return cuda::std::formatter<bool, char>::parse(ctx);
  }

  TEST_HOST_DEVICE_FUNC auto format(inspect_argument, cuda::std::format_context& ctx) const
  {
    return cuda::std::formatter<bool, char>::format(cuda::std::visit_format_arg(is_promoted_int{}, ctx.arg(id)), ctx);
  }
};

int main(int, char**)
{
  inspect_argument inspect;
  short value = 42;
  char out[4]{};
  auto end = cuda::std::format_to(out, "{}", inspect, value);
  assert((cuda::std::string_view{out, end} == "true"));

  char erased[4]{};
  auto args = cuda::std::make_format_args(inspect, value);
  end       = cuda::std::vformat_to(erased, "{}", args);
  assert((cuda::std::string_view{erased, end} == "true"));

  assert(cuda::std::formatted_size("{}", inspect, value) == 4);
  char limited[4]{};
  auto result = cuda::std::format_to_n(limited, 2, "{}", inspect, value);
  assert(result.size == 4);
  assert((cuda::std::string_view{limited, result.out} == "tr"));
  assert(limited[2] == 0);
  return 0;
}
