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

#include <cuda_runtime.h>

inline constexpr auto full_warp_mask = ~0u;

__device__ __uint128_t warp_reduce_max(__uint128_t __v)
{
  const auto __w0 = static_cast<uint32_t>(__v);
  const auto __w1 = static_cast<uint32_t>(__v >> 32);
  const auto __w2 = static_cast<uint32_t>(__v >> 64);
  const auto __w3 = static_cast<uint32_t>(__v >> 96);

  const auto __r3 = __reduce_max_sync(full_warp_mask, __w3);
  bool __valid    = __r3 == __w3;
  const auto __r2 = __reduce_max_sync(full_warp_mask, (__valid) ? __w2 : 0u);
  __valid         = __valid && __r2 == __w2;
  const auto __r1 = __reduce_max_sync(full_warp_mask, (__valid) ? __w1 : 0u);
  __valid         = __valid && __r1 == __w1;
  const auto __r0 = __reduce_max_sync(full_warp_mask, (__valid) ? __w0 : 0u);

  return (__uint128_t{__r3} << 96) | (__uint128_t{__r2} << 64) | (__uint128_t{__r1} << 32) | __r0;
}

__global__ void test_warp_reduce_max_kernel(const __uint128_t* inputs, __uint128_t* outputs)
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

bool test_warp_reduce_max(const char* name, const std::array<__uint128_t, 32>& inputs)
{
  __uint128_t* device_inputs  = nullptr;
  __uint128_t* device_outputs = nullptr;
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_inputs), sizeof(__uint128_t) * inputs.size()));
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_outputs), sizeof(__uint128_t) * inputs.size()));

  __uint128_t expected = 0;
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
    const auto actual = device_outputs[lane];
    if (actual != expected)
    {
      if (success)
      {
        std::fprintf(
          stderr,
          "%s: lane %zu: expected 0x%016llx%016llx, got 0x%016llx%016llx\n",
          name,
          lane,
          static_cast<unsigned long long>(expected >> 64),
          static_cast<unsigned long long>(expected),
          static_cast<unsigned long long>(actual >> 64),
          static_cast<unsigned long long>(actual));
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

__uint128_t random_value(uint64_t& state)
{
  return (__uint128_t{next_random(state)} << 64) | next_random(state);
}

int main()
{
  bool success = true;
  std::array<__uint128_t, 32> inputs{};
  success &= test_warp_reduce_max("all zeros", inputs);

  const auto maximum = ~__uint128_t{0};
  inputs.fill(maximum);
  success &= test_warp_reduce_max("all maximum", inputs);

  const auto baseline = (__uint128_t{2} << 96) | (__uint128_t{2} << 64) | (__uint128_t{2} << 32) | 2u;
  char name[64];
  for (unsigned word = 0; word < 4; ++word)
  {
    auto winner     = baseline;
    const auto mask = __uint128_t{0xffff'ffffu} << (word * 32);
    winner          = (winner & ~mask) | (__uint128_t{3} << (word * 32));
    for (unsigned lower = 0; lower < word; ++lower)
    {
      winner &= ~(__uint128_t{0xffff'ffffu} << (lower * 32));
    }
    inputs.fill(baseline);
    inputs[17] = winner;
    std::snprintf(name, sizeof(name), "word %u decides maximum", word);
    success &= test_warp_reduce_max(name, inputs);
  }

  inputs.fill(0);
  inputs[3]  = baseline;
  inputs[23] = baseline;
  success &= test_warp_reduce_max("tied maximum lanes", inputs);

  for (unsigned winner = 0; winner < inputs.size(); ++winner)
  {
    inputs.fill(0);
    inputs[winner] = (__uint128_t{1} << 127) | winner;
    std::snprintf(name, sizeof(name), "winner in lane %u", winner);
    success &= test_warp_reduce_max(name, inputs);
  }

  for (unsigned bit = 0; bit < 128; ++bit)
  {
    const auto value = __uint128_t{1} << bit;
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
    const unsigned shared_words = case_index % 4;
    const auto common           = random_value(random_state);
    const auto shared_mask      = shared_words == 0 ? __uint128_t{0} : maximum << ((4 - shared_words) * 32);
    for (auto& value : inputs)
    {
      value = (random_value(random_state) & ~shared_mask) | (common & shared_mask);
    }
    std::snprintf(name, sizeof(name), "random case %u", case_index);
    success &= test_warp_reduce_max(name, inputs);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
