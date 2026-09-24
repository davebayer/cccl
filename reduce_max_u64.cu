//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <cuda_runtime.h>

inline constexpr auto full_warp_mask = ~0u;

__device__ uint64_t warp_reduce_max(uint64_t __v)
{
  const auto __lo = static_cast<uint32_t>(__v);
  const auto __hi = static_cast<uint32_t>(__v >> 32);

  const auto __ret_hi = __reduce_max_sync(full_warp_mask, __hi);
  const auto __ret_lo = __reduce_max_sync(full_warp_mask, (__ret_hi == __hi) ? __lo : 0u);

  return (uint64_t{__ret_hi} << 32) | __ret_lo;
}

__global__ void test_warp_reduce_max_kernel(const uint64_t* inputs, uint64_t* outputs)
{
  const auto lane = threadIdx.x;
  outputs[lane]   = warp_reduce_max(inputs[lane]);
}

void check_cuda(cudaError_t error)
{
  if (error != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

bool test_warp_reduce_max(const char* name, const std::array<uint64_t, 32>& inputs)
{
  uint64_t* device_inputs  = nullptr;
  uint64_t* device_outputs = nullptr;
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_inputs), sizeof(uint64_t) * inputs.size()));
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_outputs), sizeof(uint64_t) * inputs.size()));

  uint64_t expected = 0;
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    device_inputs[lane] = inputs[lane];
    if (inputs[lane] > expected)
    {
      expected = inputs[lane];
    }
  }

  test_warp_reduce_max_kernel<<<1, 32>>>(device_inputs, device_outputs);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());

  bool success = true;
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    if (device_outputs[lane] != expected)
    {
      if (success)
      {
        std::fprintf(
          stderr,
          "%s: lane %zu: expected 0x%016llx, got 0x%016llx\n",
          name,
          lane,
          static_cast<unsigned long long>(expected),
          static_cast<unsigned long long>(device_outputs[lane]));
      }
      success = false;
    }
  }

  check_cuda(cudaFree(device_outputs));
  check_cuda(cudaFree(device_inputs));
  return success;
}

uint64_t next_random(uint64_t& state)
{
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545'f491'4f6c'dd1dull;
}

int main()
{
  bool success = true;
  std::array<uint64_t, 32> inputs{};
  success &= test_warp_reduce_max("all zeros", inputs);

  inputs.fill(std::numeric_limits<uint64_t>::max());
  success &= test_warp_reduce_max("all maximum", inputs);

  inputs.fill((uint64_t{1} << 32) | 0xffff'ffffu);
  inputs[17] = uint64_t{2} << 32;
  success &= test_warp_reduce_max("larger high word wins", inputs);

  inputs[5]  = (uint64_t{2} << 32) | 100u;
  inputs[27] = (uint64_t{2} << 32) | 1000u;
  success &= test_warp_reduce_max("larger low word breaks tie", inputs);

  inputs[11] = inputs[27];
  success &= test_warp_reduce_max("multiple maximum lanes", inputs);

  char name[64];
  for (unsigned winner = 0; winner < inputs.size(); ++winner)
  {
    inputs.fill(0);
    inputs[winner] = 0x8000'0000'0000'0000ull + winner;
    std::snprintf(name, sizeof(name), "winner in lane %u", winner);
    success &= test_warp_reduce_max(name, inputs);
  }

  for (unsigned bit = 0; bit < 64; ++bit)
  {
    const auto value = uint64_t{1} << bit;
    inputs.fill(value);
    std::snprintf(name, sizeof(name), "bit %u in every lane", bit);
    success &= test_warp_reduce_max(name, inputs);

    inputs.fill(0);
    inputs[bit % 32] = value;
    std::snprintf(name, sizeof(name), "bit %u in one lane", bit);
    success &= test_warp_reduce_max(name, inputs);
  }

  uint64_t random_state = 0x8f72'3a65'd4b9'10ecull;
  for (unsigned case_index = 0; case_index < 128; ++case_index)
  {
    const auto common_hi = static_cast<uint32_t>(next_random(random_state));
    for (auto& value : inputs)
    {
      value = case_index % 2 == 0
              ? next_random(random_state)
              : (uint64_t{common_hi} << 32) | static_cast<uint32_t>(next_random(random_state));
    }
    std::snprintf(name, sizeof(name), "random case %u", case_index);
    success &= test_warp_reduce_max(name, inputs);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
