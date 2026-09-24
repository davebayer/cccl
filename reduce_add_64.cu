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

__device__ uint64_t warp_reduce_add(uint64_t __v)
{
  const auto __lo = static_cast<uint32_t>(__v);
  const auto __hi = static_cast<uint32_t>(__v >> 32);

  const auto __lo0 = __lo & 0x07ff'ffffu;
  const auto __lo1 = (__lo & 0xf800'0000u) >> 27;

  const auto __tmp0 = __reduce_add_sync(full_warp_mask, __lo0);
  const auto __tmp1 = __reduce_add_sync(full_warp_mask, __lo1);
  const auto __tmp2 = __tmp1 + (__tmp0 >> 27);

  const auto __ret_lo = __tmp0 + (__tmp1 << 27);
  const auto __ret_hi = __reduce_add_sync(full_warp_mask, __hi) + (__tmp2 >> 5);

  return (uint64_t{__ret_hi} << 32) | __ret_lo;
}

__device__ int64_t warp_reduce_add(int64_t __v)
{
  return static_cast<int64_t>(warp_reduce_add(static_cast<uint64_t>(__v)));
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

  uint64_t expected = 0;
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    device_inputs[lane] = inputs[lane];
    expected += static_cast<uint64_t>(inputs[lane]);
  }

  test_warp_reduce_add_kernel<<<1, 32>>>(device_inputs, device_outputs);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());

  bool success = true;
  for (std::size_t lane = 0; lane < inputs.size(); ++lane)
  {
    if (static_cast<uint64_t>(device_outputs[lane]) != expected)
    {
      if (success)
      {
        std::fprintf(
          stderr,
          "%s: lane %zu: expected 0x%016llx, got 0x%016llx\n",
          name,
          lane,
          static_cast<unsigned long long>(expected),
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
  std::array<uint64_t, 32> unsigned_inputs{};
  success &= test_warp_reduce_add("u64 zeros", unsigned_inputs);

  unsigned_inputs.fill(std::numeric_limits<uint64_t>::max());
  success &= test_warp_reduce_add("u64 wraparound", unsigned_inputs);

  unsigned_inputs.fill(0);
  unsigned_inputs[17] = std::numeric_limits<uint64_t>::max();
  success &= test_warp_reduce_add("u64 single lane", unsigned_inputs);

  for (uint64_t lane = 0; lane < unsigned_inputs.size(); ++lane)
  {
    unsigned_inputs[lane] = (lane << 32) | (0xffff'fff0u + (lane & 15u));
  }
  success &= test_warp_reduce_add("u64 low word carries", unsigned_inputs);

  for (uint64_t lane = 0; lane < unsigned_inputs.size(); ++lane)
  {
    unsigned_inputs[lane] = (uint64_t{1} << (lane + 27)) | (uint64_t{1} << lane);
  }
  success &= test_warp_reduce_add("u64 varied bits", unsigned_inputs);

  std::array<int64_t, 32> signed_inputs{};
  success &= test_warp_reduce_add("i64 zeros", signed_inputs);

  signed_inputs.fill(-1);
  success &= test_warp_reduce_add("i64 negative ones", signed_inputs);

  for (std::size_t lane = 0; lane < signed_inputs.size(); ++lane)
  {
    signed_inputs[lane] = lane % 2 == 0 ? std::numeric_limits<int64_t>::min() : std::numeric_limits<int64_t>::max();
  }
  success &= test_warp_reduce_add("i64 alternating extremes", signed_inputs);

  for (std::size_t lane = 0; lane < signed_inputs.size(); ++lane)
  {
    signed_inputs[lane] =
      lane % 2 == 0 ? -static_cast<int64_t>(lane * 0x1000'0001ull) : static_cast<int64_t>(lane * 0x8000'0001ull);
  }
  success &= test_warp_reduce_add("i64 mixed magnitudes", signed_inputs);

  signed_inputs.fill(std::numeric_limits<int64_t>::max());
  success &= test_warp_reduce_add("i64 wraparound", signed_inputs);

  char name[64];
  for (unsigned bit = 0; bit < 64; ++bit)
  {
    const auto value = uint64_t{1} << bit;

    unsigned_inputs.fill(value);
    signed_inputs.fill(static_cast<int64_t>(value));
    std::snprintf(name, sizeof(name), "u64 bit %u in every lane", bit);
    success &= test_warp_reduce_add(name, unsigned_inputs);
    std::snprintf(name, sizeof(name), "i64 bit %u in every lane", bit);
    success &= test_warp_reduce_add(name, signed_inputs);

    unsigned_inputs.fill(0);
    signed_inputs.fill(0);
    unsigned_inputs[bit % 32] = value;
    signed_inputs[bit % 32]   = static_cast<int64_t>(value);
    std::snprintf(name, sizeof(name), "u64 bit %u in one lane", bit);
    success &= test_warp_reduce_add(name, unsigned_inputs);
    std::snprintf(name, sizeof(name), "i64 bit %u in one lane", bit);
    success &= test_warp_reduce_add(name, signed_inputs);
  }

  uint64_t random_state = 0x8f72'3a65'd4b9'10ecull;
  for (unsigned case_index = 0; case_index < 128; ++case_index)
  {
    for (std::size_t lane = 0; lane < unsigned_inputs.size(); ++lane)
    {
      unsigned_inputs[lane] = next_random(random_state);
      signed_inputs[lane]   = static_cast<int64_t>(unsigned_inputs[lane]);
    }
    std::snprintf(name, sizeof(name), "u64 random case %u", case_index);
    success &= test_warp_reduce_add(name, unsigned_inputs);
    std::snprintf(name, sizeof(name), "i64 random case %u", case_index);
    success &= test_warp_reduce_add(name, signed_inputs);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
