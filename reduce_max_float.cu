// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

inline constexpr auto full_warp_mask = ~0u;

// The idea is to use signed integer reduction and handle special cases, so we don't produce invalid results but still
// can use the __reduce_max_sync operation.
__device__ float reduce_max_float(float v)
{
  auto bits = cuda::std::bit_cast<cuda::std::uint32_t>(v);

  if (cuda::std::isnan(v))
  {
    // When v is NaN, set bits to canonical NaN. For min operation, we can take advantage of the fact that the integer
    // value of NaNs is always greater than any other float number.
    bits = cuda::std::bit_cast<cuda::std::uint32_t>(cuda::std::numeric_limits<float>::quiet_NaN());
  }
  else if (cuda::std::signbit(v))
  {
    // If v is negative, we need to flip all bits except for the sign bit, because bit representation of negative floats
    // grows in different direction.
    bits = cuda::std::bit_cast<cuda::std::uint32_t>(v) ^ 0x7fff'ffffu;
  }

  // Calculate the reduction as 32-bit signed integers.
  const auto result = __reduce_min_sync(full_warp_mask, static_cast<cuda::std::int32_t>(bits));

  // When the result is negative, we need to correct the result.
  return cuda::std::bit_cast<float>((result < 0) ? result ^ 0x7fff'ffffu : result);
}
