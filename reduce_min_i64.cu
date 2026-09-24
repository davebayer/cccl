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

__device__ int64_t warp_reduce_min(int64_t __v)
{
  // Flip the sign bit so unsigned min orders the high word as signed.
  const auto __lo = static_cast<uint32_t>(__v);
  const auto __hi = static_cast<uint32_t>(static_cast<uint64_t>(__v) >> 32);

  const auto __min_hi = static_cast<uint32_t>(__reduce_min_sync(full_warp_mask, static_cast<int32_t>(__hi)));
  const auto __min_lo = __reduce_min_sync(full_warp_mask, (__min_hi == __hi) ? __lo : ~0u);

  return static_cast<int64_t>((uint64_t{__min_hi} << 32) | __min_lo);
}

__global__ void test_warp_reduce_min_kernel(const int64_t* inputs, int64_t* outputs)
{
  const auto lane = threadIdx.x;
  outputs[lane]   = warp_reduce_min(inputs[lane]);
}

void check_cuda(cudaError_t error)
{
  if (error != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

bool test_warp_reduce_min(const char* name, const std::array<int64_t, 32>& inputs)
{
  int64_t* device_inputs  = nullptr;
  int64_t* device_outputs = nullptr;
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_inputs), sizeof(int64_t) * inputs.size()));
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_outputs), sizeof(int64_t) * inputs.size()));

  int64_t expected = std::numeric_limits<int64_t>::max();
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    device_inputs[lane] = inputs[lane];
    if (inputs[lane] < expected)
    {
      expected = inputs[lane];
    }
  }

  test_warp_reduce_min_kernel<<<1, 32>>>(device_inputs, device_outputs);
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
          static_cast<unsigned long long>(static_cast<uint64_t>(expected)),
          static_cast<unsigned long long>(static_cast<uint64_t>(device_outputs[lane])));
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
  std::array<int64_t, 32> inputs{};
  success &= test_warp_reduce_min("all zeros", inputs);

  inputs.fill(std::numeric_limits<int64_t>::max());
  success &= test_warp_reduce_min("all maximum", inputs);

  inputs.fill(std::numeric_limits<int64_t>::min());
  success &= test_warp_reduce_min("all minimum", inputs);

  inputs.fill(-1);
  success &= test_warp_reduce_min("all negative ones", inputs);

  inputs.fill(std::numeric_limits<int64_t>::max());
  inputs[17] = std::numeric_limits<int64_t>::min();
  success &= test_warp_reduce_min("negative high word wins", inputs);

  inputs.fill(0);
  inputs[9] = -1;
  success &= test_warp_reduce_min("negative versus zero", inputs);

  inputs.fill(static_cast<int64_t>(0x8000'0001'0000'0000ull));
  inputs[17] = static_cast<int64_t>(0x8000'0000'ffff'ffffull);
  success &= test_warp_reduce_min("signed high word dominates low word", inputs);

  inputs.fill(static_cast<int64_t>(0x8000'0001'ffff'ffffull));
  inputs[5]  = static_cast<int64_t>(0x8000'0001'0000'0064ull);
  inputs[27] = static_cast<int64_t>(0x8000'0001'0000'0007ull);
  success &= test_warp_reduce_min("negative high word tie", inputs);

  inputs[11] = inputs[27];
  success &= test_warp_reduce_min("multiple minimum lanes", inputs);

  inputs.fill(0x0000'0001'ffff'ffffll);
  inputs[5]  = 0x0000'0001'0000'0064ll;
  inputs[27] = 0x0000'0001'0000'0007ll;
  success &= test_warp_reduce_min("positive high word tie", inputs);

  char name[64];
  for (unsigned winner = 0; winner < inputs.size(); ++winner)
  {
    inputs.fill(std::numeric_limits<int64_t>::max());
    inputs[winner] = std::numeric_limits<int64_t>::min() + winner;
    std::snprintf(name, sizeof(name), "winner in lane %u", winner);
    success &= test_warp_reduce_min(name, inputs);
  }

  for (unsigned bit = 0; bit < 64; ++bit)
  {
    const auto value = static_cast<int64_t>(uint64_t{1} << bit);
    inputs.fill(value);
    std::snprintf(name, sizeof(name), "bit %u in every lane", bit);
    success &= test_warp_reduce_min(name, inputs);

    inputs.fill(std::numeric_limits<int64_t>::max());
    inputs[bit % 32] = value;
    std::snprintf(name, sizeof(name), "bit %u in one lane", bit);
    success &= test_warp_reduce_min(name, inputs);
  }

  uint64_t random_state = 0x8f72'3a65'd4b9'10ecull;
  for (unsigned case_index = 0; case_index < 128; ++case_index)
  {
    const auto common_hi = static_cast<uint32_t>(next_random(random_state));
    for (auto& value : inputs)
    {
      const auto bits = case_index % 2 == 0
                        ? next_random(random_state)
                        : (uint64_t{common_hi} << 32) | static_cast<uint32_t>(next_random(random_state));
      value           = static_cast<int64_t>(bits);
    }
    std::snprintf(name, sizeof(name), "random case %u", case_index);
    success &= test_warp_reduce_min(name, inputs);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
