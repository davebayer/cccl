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
    bits = cuda::std::bit_cast<cuda::std::uint32_t>(-cuda::std::numeric_limits<float>::quiet_NaN()) ^ 0x7fff'ffffu;
  }
  else if (cuda::std::signbit(v))
  {
    // If v is negative, we need to flip all bits except for the sign bit, because bit representation of negative floats
    // grows in different direction.
    bits = cuda::std::bit_cast<cuda::std::uint32_t>(v) ^ 0x7fff'ffffu;
  }

  // Calculate the reduction as 32-bit signed integers.
  const auto result = __reduce_max_sync(full_warp_mask, static_cast<cuda::std::int32_t>(bits));

  // When the result is negative, we need to correct the result.
  const auto result_f = cuda::std::bit_cast<float>((result < 0) ? result ^ 0x7fff'ffffu : result);
  return (cuda::std::isnan(result_f)) ? cuda::std::numeric_limits<float>::quiet_NaN() : result_f;
}

// Standalone test:
// nvcc -std=c++17 -arch=sm_80 -Ilibcudacxx/include reduce_max_float.cu -o /tmp/reduce_max_float_test
// /tmp/reduce_max_float_test
#include <cuda/std/array>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

namespace
{
constexpr unsigned warp_size = 32;

struct test_case
{
  cuda::std::array<float, warp_size> values;
  const char* name;
  unsigned rotation;
};

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void test_reduce_max_float(const test_case* cases, unsigned* results)
{
  const auto& test                       = cases[blockIdx.x];
  const auto lane                        = threadIdx.x;
  const auto result                      = reduce_max_float(test.values[lane]);
  results[blockIdx.x * warp_size + lane] = cuda::std::bit_cast<unsigned>(result);
}

float reference_max(const test_case& test)
{
  auto result = cuda::std::numeric_limits<float>::quiet_NaN();
  for (unsigned lane = 0; lane < warp_size; ++lane)
  {
    const auto value = test.values[lane];
    if (cuda::std::isnan(value))
    {
      continue;
    }
    // Ignore NaNs unless every participating value is NaN; prefer +0 over -0.
    if (cuda::std::isnan(result) || value > result || (value == 0.0f && result == 0.0f && !cuda::std::signbit(value)))
    {
      result = value;
    }
  }
  return result;
}

void add_cases(std::vector<test_case>& cases, const char* name, const std::vector<float>& pattern)
{
  // Move each value through every lane of the full warp.
  for (unsigned rotation = 0; rotation < warp_size; ++rotation)
  {
    test_case test{{}, name, rotation};
    for (unsigned lane = 0; lane < warp_size; ++lane)
    {
      test.values[(lane + rotation) % warp_size] = pattern[lane % pattern.size()];
    }
    cases.push_back(test);
  }
}
} // namespace

int main()
{
  cudaDeviceProp properties{};
  int device = 0;
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 8)
  {
    std::fprintf(stderr, "Warp integer reductions require compute capability 8.0 or newer.\n");
    return EXIT_FAILURE;
  }

  const auto inf          = cuda::std::numeric_limits<float>::infinity();
  const auto nan          = cuda::std::numeric_limits<float>::quiet_NaN();
  const auto max          = cuda::std::numeric_limits<float>::max();
  const auto normal       = cuda::std::numeric_limits<float>::min();
  const auto subnormal    = cuda::std::numeric_limits<float>::denorm_min();
  const auto negative_nan = cuda::std::bit_cast<float>(0xffc12345u);
  std::vector<test_case> cases;
  add_cases(cases, "positive", {1.0f, 17.5f, 0.25f, 100.0f});
  add_cases(cases, "negative", {-1.0f, -17.5f, -0.25f, -100.0f});
  add_cases(cases, "mixed signs", {-4.5f, 2.0f, -0.125f, 100.0f, -19.0f});
  add_cases(cases, "equal", {3.25f});
  add_cases(cases, "positive zero", {0.0f});
  add_cases(cases, "negative zero", {-0.0f});
  add_cases(cases, "signed zeros", {0.0f, -0.0f});
  add_cases(cases, "zeros and finite", {0.0f, -0.0f, 1.0f, -1.0f});
  add_cases(cases, "positive infinity", {inf});
  add_cases(cases, "negative infinity", {-inf});
  add_cases(cases, "infinities and finite", {inf, -inf, max, -max, 1.0f, -1.0f});
  add_cases(cases, "finite limits", {max, -max, normal, -normal});
  add_cases(cases, "subnormals", {subnormal, -subnormal, 2 * subnormal, -2 * subnormal});
  add_cases(cases, "all NaN", {nan});
  add_cases(cases, "signed NaNs", {nan, negative_nan});
  add_cases(cases, "NaNs and finite", {nan, 7.0f, negative_nan, -3.5f, 0.0f});
  add_cases(cases, "NaNs and infinity", {nan, inf});
  add_cases(cases, "NaNs and negative infinity", {negative_nan, -inf});

  // Deterministic pseudo-random finite values without a library-dependent distribution.
  unsigned state = 0x12345678u;
  for (unsigned sample = 0; sample < 8; ++sample)
  {
    std::vector<float> pattern;
    for (unsigned lane = 0; lane < warp_size; ++lane)
    {
      state = state * 1664525u + 1013904223u;
      pattern.push_back(static_cast<float>(static_cast<int>(state % 200001u) - 100000) / 32.0f);
    }
    add_cases(cases, "random finite", pattern);
  }

  test_case* device_cases  = nullptr;
  unsigned* device_results = nullptr;
  std::vector<unsigned> results(cases.size() * warp_size);
  check_cuda(cudaMalloc(&device_cases, cases.size() * sizeof(test_case)));
  check_cuda(cudaMalloc(&device_results, results.size() * sizeof(unsigned)));
  check_cuda(cudaMemcpy(device_cases, cases.data(), cases.size() * sizeof(test_case), cudaMemcpyHostToDevice));
  test_reduce_max_float<<<static_cast<unsigned>(cases.size()), warp_size>>>(device_cases, device_results);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
  check_cuda(cudaMemcpy(results.data(), device_results, results.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  check_cuda(cudaFree(device_results));
  check_cuda(cudaFree(device_cases));

  std::size_t failures = 0;
  for (std::size_t index = 0; index < cases.size(); ++index)
  {
    const auto& test    = cases[index];
    const auto expected = reference_max(test);
    for (unsigned lane = 0; lane < warp_size; ++lane)
    {
      const auto actual_word = results[index * warp_size + lane];
      const auto actual      = cuda::std::bit_cast<float>(actual_word);
      const auto matches =
        cuda::std::isnan(expected) ? cuda::std::isnan(actual) : actual_word == cuda::std::bit_cast<unsigned>(expected);
      if (!matches)
      {
        if (failures < 20)
        {
          std::fprintf(
            stderr,
            "%s: rotation=%u lane=%u expected=0x%08x actual=0x%08x\n",
            test.name,
            test.rotation,
            lane,
            cuda::std::bit_cast<unsigned>(expected),
            actual_word);
        }
        ++failures;
      }
    }
  }
  std::printf("%zu cases, %zu lane checks, %zu failures\n", cases.size(), results.size(), failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
