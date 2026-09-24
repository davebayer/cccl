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

// Thirty-two 27-bit values plus the incoming carry fit in a 32-bit sum.
__device__ uint32_t assemble_reduced_word(uint32_t __low_sum, uint32_t __high_sum, uint32_t& __carry)
{
  const auto __sum_lo = __low_sum + __carry;
  const auto __sum_hi = __high_sum + (__sum_lo >> 27);
  __carry             = __sum_hi >> 5;
  return (__sum_lo & 0x07ff'ffffu) | (__sum_hi << 27);
}

__device__ __uint128_t warp_reduce_add(__uint128_t __v)
{
  const auto __w0 = static_cast<uint32_t>(__v);
  const auto __w1 = static_cast<uint32_t>(__v >> 32);
  const auto __w2 = static_cast<uint32_t>(__v >> 64);
  const auto __w3 = static_cast<uint32_t>(__v >> 96);

  // Each five-bit field sums to at most 32 * 31 = 992, so three
  // ten-bit fields can be reduced together without cross-field carries.
  const auto __packed_hi = (__w0 >> 27) | ((__w1 >> 27) << 10) | ((__w2 >> 27) << 20);
  const auto __high_sums = __reduce_add_sync(full_warp_mask, __packed_hi);
  const auto __low0      = __reduce_add_sync(full_warp_mask, __w0 & 0x07ff'ffffu);
  const auto __low1      = __reduce_add_sync(full_warp_mask, __w1 & 0x07ff'ffffu);
  const auto __low2      = __reduce_add_sync(full_warp_mask, __w2 & 0x07ff'ffffu);

  uint32_t __carry = 0;
  const auto __r0  = assemble_reduced_word(__low0, __high_sums & 0x3ffu, __carry);
  const auto __r1  = assemble_reduced_word(__low1, (__high_sums >> 10) & 0x3ffu, __carry);
  const auto __r2  = assemble_reduced_word(__low2, __high_sums >> 20, __carry);
  const auto __r3  = __reduce_add_sync(full_warp_mask, __w3) + __carry;

  return (__uint128_t{__r3} << 96) | (__uint128_t{__r2} << 64) | (__uint128_t{__r1} << 32) | __r0;
}

__device__ __int128_t warp_reduce_add(__int128_t __v)
{
  return static_cast<__int128_t>(warp_reduce_add(static_cast<__uint128_t>(__v)));
}

template <class T>
__global__ void test_warp_reduce_add_kernel(const T* inputs, T* outputs)
{
  const auto lane = threadIdx.x;
  outputs[lane]   = warp_reduce_add(inputs[lane]);
}

void check_cuda(cudaError_t error)
{
  if (error != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
bool test_warp_reduce_add(const char* name, const std::array<T, 32>& inputs)
{
  T* device_inputs  = nullptr;
  T* device_outputs = nullptr;
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_inputs), sizeof(T) * inputs.size()));
  check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&device_outputs), sizeof(T) * inputs.size()));

  __uint128_t expected = 0;
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    device_inputs[lane] = inputs[lane];
    expected += static_cast<__uint128_t>(inputs[lane]);
  }

  test_warp_reduce_add_kernel<<<1, 32>>>(device_inputs, device_outputs);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());

  bool success = true;
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    const auto actual = static_cast<__uint128_t>(device_outputs[lane]);
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

int main()
{
  bool success = true;
  std::array<__uint128_t, 32> unsigned_inputs{};
  std::array<__int128_t, 32> signed_inputs{};

  success &= test_warp_reduce_add("u128 zeros", unsigned_inputs);
  success &= test_warp_reduce_add("i128 zeros", signed_inputs);

  unsigned_inputs.fill(~__uint128_t{0});
  signed_inputs.fill(-1);
  success &= test_warp_reduce_add("u128 wraparound", unsigned_inputs);
  success &= test_warp_reduce_add("i128 negative ones", signed_inputs);

  for (unsigned bit = 0; bit < 128; ++bit)
  {
    const auto value = __uint128_t{1} << bit;
    unsigned_inputs.fill(value);
    signed_inputs.fill(static_cast<__int128_t>(value));

    char name[64];
    std::snprintf(name, sizeof(name), "u128 bit %u in every lane", bit);
    success &= test_warp_reduce_add(name, unsigned_inputs);
    std::snprintf(name, sizeof(name), "i128 bit %u in every lane", bit);
    success &= test_warp_reduce_add(name, signed_inputs);

    unsigned_inputs.fill(0);
    signed_inputs.fill(0);
    unsigned_inputs[bit % 32] = value;
    signed_inputs[bit % 32]   = static_cast<__int128_t>(value);
    std::snprintf(name, sizeof(name), "u128 bit %u in one lane", bit);
    success &= test_warp_reduce_add(name, unsigned_inputs);
    std::snprintf(name, sizeof(name), "i128 bit %u in one lane", bit);
    success &= test_warp_reduce_add(name, signed_inputs);
  }

  uint64_t random_state = 0x8f72'3a65'd4b9'10ecull;
  for (unsigned case_index = 0; case_index < 128; ++case_index)
  {
    for (std::size_t lane = 0; lane < unsigned_inputs.size(); ++lane)
    {
      unsigned_inputs[lane] = (__uint128_t{next_random(random_state)} << 64) | next_random(random_state);
      signed_inputs[lane]   = static_cast<__int128_t>(unsigned_inputs[lane]);
    }

    char name[64];
    std::snprintf(name, sizeof(name), "u128 random case %u", case_index);
    success &= test_warp_reduce_add(name, unsigned_inputs);
    std::snprintf(name, sizeof(name), "i128 random case %u", case_index);
    success &= test_warp_reduce_add(name, signed_inputs);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
