// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile with both experimental macros and the early cudax group feature macro.
#include <cub/device/dispatch/kernels/kernel_scan_lookahead.cuh>
#include <cub/util_debug.cuh>

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <cstdio>

namespace
{
inline constexpr int block_count = 3;
inline constexpr int block_size  = 4 * 32;
inline constexpr int value_count = block_count * block_size;

struct affine
{
  cuda::std::uint32_t scale;
  cuda::std::uint32_t bias;
};

struct compose
{
  // Composition of affine functions is associative and preserves operand order.
  __host__ __device__ affine operator()(affine lhs, affine rhs) const
  {
    return {rhs.scale * lhs.scale, rhs.scale * lhs.bias + rhs.bias};
  }
};

struct counted_compose
{
  int* invalid_calls;
  int valid_items;

  __device__ affine operator()(affine lhs, affine rhs) const
  {
    if (static_cast<int>(threadIdx.x % 32) >= valid_items)
    {
      atomicAdd(invalid_calls, 1);
    }
    return compose{}(lhs, rhs);
  }
};

static_assert(!cuda::has_identity_element_v<compose, affine>);
static_assert(!cuda::has_identity_element_v<counted_compose, affine>);

__host__ __device__ int input_value(int, int warp, int lane)
{
  return warp * 100 + lane + 1;
}

__host__ __device__ affine input_value(affine, int warp, int lane)
{
  return {static_cast<cuda::std::uint32_t>(lane % 3 + 1), static_cast<cuda::std::uint32_t>(warp * 100 + lane + 1)};
}

[[nodiscard]] bool equals(int lhs, int rhs)
{
  return lhs == rhs;
}

[[nodiscard]] bool equals(affine lhs, affine rhs)
{
  return lhs.scale == rhs.scale && lhs.bias == rhs.bias;
}

template <class Tp>
struct scan_result
{
  Tp direct;
  Tp partial;
  Tp full;
};

template <class Tp, class PartialOp, class FullOp>
__global__ void evaluate(scan_result<Tp>* results, int valid_items, PartialOp partial_op, FullOp full_op)
{
  const int thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int lane   = thread % 32;
  const Tp input   = input_value(Tp{}, thread / 32, lane);
  results[thread]  = {
    cub::detail::scan::warp_scan_exclusive_cudax(input, partial_op, valid_items),
    cub::detail::scan::warpScanExclusivePartial(input, partial_op, valid_items, lane < valid_items),
    cub::detail::scan::warpScanExclusive(input, full_op),
  };
}

template <class Tp, class PartialOp, class FullOp>
[[nodiscard]] bool check_case(int valid_items, PartialOp partial_op, FullOp full_op)
{
  scan_result<Tp>* device_results{};
  CubDebugExit(cudaMalloc(&device_results, value_count * sizeof(scan_result<Tp>)));
  evaluate<Tp><<<block_count, block_size>>>(device_results, valid_items, partial_op, full_op);
  CubDebugExit(cudaGetLastError());
  cuda::std::array<scan_result<Tp>, value_count> results{};
  CubDebugExit(cudaMemcpy(results.data(), device_results, sizeof(results), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaFree(device_results));

  for (int warp = 0; warp < value_count / 32; ++warp)
  {
    Tp expected = input_value(Tp{}, warp, 0);
    for (int lane = 1; lane < 32; ++lane)
    {
      const auto& result = results[32 * warp + lane];
      if (!equals(result.full, expected)
          || (lane < valid_items && (!equals(result.direct, expected) || !equals(result.partial, expected))))
      {
        std::printf("Incorrect exclusive prefix: valid_items=%d, warp=%d, lane=%d\n", valid_items, warp, lane);
        return false;
      }
      expected = full_op(expected, input_value(Tp{}, warp, lane));
    }
  }
  return true;
}
} // namespace

int main()
{
  int* device_invalid_calls{};
  CubDebugExit(cudaMalloc(&device_invalid_calls, sizeof(int)));
  CubDebugExit(cudaMemset(device_invalid_calls, 0, sizeof(int)));
  for (int valid_items = 0; valid_items <= 32; ++valid_items)
  {
    if (!check_case<int>(valid_items, cuda::std::plus<>{}, cuda::std::plus<>{})
        || !check_case<affine>(valid_items, counted_compose{device_invalid_calls, valid_items}, compose{}))
    {
      CubDebugExit(cudaFree(device_invalid_calls));
      return 1;
    }
  }
  int invalid_calls{};
  CubDebugExit(cudaMemcpy(&invalid_calls, device_invalid_calls, sizeof(int), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaFree(device_invalid_calls));
  if (invalid_calls != 0)
  {
    std::printf("The scan operator ran %d times in invalid lanes\n", invalid_calls);
    return 1;
  }
  std::puts("PASS: 0..32 valid items, integer plus, noncommutative affine composition, and 12 physical warps");
}
