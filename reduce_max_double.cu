// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

inline constexpr auto full_warp_mask = ~0u;

// Reduce ordered double bit patterns using two 32-bit integer reductions.
// All 32 lanes of the warp must participate.
__device__ double reduce_max_double(double v)
{
  constexpr cuda::std::uint64_t magnitude_mask = 0x7fff'ffff'ffff'ffffull;
  auto bits                                    = cuda::std::bit_cast<cuda::std::uint64_t>(v);
  if (cuda::std::isnan(v))
  {
    // Place every NaN below negative infinity in the signed integer ordering.
    bits = cuda::std::bit_cast<cuda::std::uint64_t>(-cuda::std::numeric_limits<double>::quiet_NaN()) ^ magnitude_mask;
  }
  else if (cuda::std::signbit(v))
  {
    // Reverse the ordering of negative doubles while retaining their sign bit.
    bits ^= magnitude_mask;
  }

  const auto high        = cuda::std::bit_cast<cuda::std::int32_t>(static_cast<cuda::std::uint32_t>(bits >> 32));
  const auto result_high = __reduce_max_sync(full_warp_mask, high);
  // Only lanes with the winning high word may contribute a low word.
  const auto low        = high == result_high ? static_cast<cuda::std::uint32_t>(bits) : 0u;
  const auto result_low = __reduce_max_sync(full_warp_mask, low);
  const auto result_bits =
    (static_cast<cuda::std::uint64_t>(cuda::std::bit_cast<cuda::std::uint32_t>(result_high)) << 32) | result_low;
  const auto result = cuda::std::bit_cast<double>(result_high < 0 ? result_bits ^ magnitude_mask : result_bits);
  return cuda::std::isnan(result) ? cuda::std::numeric_limits<double>::quiet_NaN() : result;
}

// Standalone test:
// nvcc -std=c++17 -arch=sm_80 -Ilibcudacxx/include reduce_max_double.cu -o /tmp/reduce_max_double_test
// /tmp/reduce_max_double_test
#include <cuda/std/array>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

namespace
{
constexpr unsigned warp_size = 32;

struct test_case
{
  cuda::std::array<double, warp_size> values;
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

__global__ void test_reduce_max_double(const test_case* cases, cuda::std::uint64_t* results)
{
  const auto& test                       = cases[blockIdx.x];
  const auto lane                        = threadIdx.x;
  const auto result                      = reduce_max_double(test.values[lane]);
  results[blockIdx.x * warp_size + lane] = cuda::std::bit_cast<cuda::std::uint64_t>(result);
}

double reference_max(const test_case& test)
{
  auto result = cuda::std::numeric_limits<double>::quiet_NaN();
  for (unsigned lane = 0; lane < warp_size; ++lane)
  {
    const auto value = test.values[lane];
    if (cuda::std::isnan(value))
    {
      continue;
    }
    // Ignore NaNs unless every participating value is NaN; prefer +0 over -0.
    if (cuda::std::isnan(result) || value > result || (value == 0.0 && result == 0.0 && !cuda::std::signbit(value)))
    {
      result = value;
    }
  }
  return result;
}

void add_cases(std::vector<test_case>& cases, const char* name, const std::vector<double>& pattern)
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

  const auto inf          = cuda::std::numeric_limits<double>::infinity();
  const auto nan          = cuda::std::numeric_limits<double>::quiet_NaN();
  const auto max          = cuda::std::numeric_limits<double>::max();
  const auto normal       = cuda::std::numeric_limits<double>::min();
  const auto subnormal    = cuda::std::numeric_limits<double>::denorm_min();
  const auto negative_nan = cuda::std::bit_cast<double>(cuda::std::uint64_t{0xfff8123456789abcull});
  std::vector<test_case> cases;
  add_cases(cases, "positive", {1.0, 17.5, 0.25, 100.0});
  add_cases(cases, "negative", {-1.0, -17.5, -0.25, -100.0});
  add_cases(cases, "mixed signs", {-4.5, 2.0, -0.125, 100.0, -19.0});
  add_cases(cases, "equal", {3.25});
  add_cases(cases, "positive zero", {0.0});
  add_cases(cases, "negative zero", {-0.0});
  add_cases(cases, "signed zeros", {0.0, -0.0});
  add_cases(cases, "zeros and finite", {0.0, -0.0, 1.0, -1.0});
  add_cases(cases, "positive infinity", {inf});
  add_cases(cases, "negative infinity", {-inf});
  add_cases(cases, "infinities and finite", {inf, -inf, max, -max, 1.0, -1.0});
  add_cases(cases, "finite limits", {max, -max, normal, -normal});
  add_cases(cases, "subnormals", {subnormal, -subnormal, 2 * subnormal, -2 * subnormal});
  add_cases(cases, "all NaN", {nan});
  add_cases(cases, "signed NaNs", {nan, negative_nan});
  add_cases(cases, "NaNs and finite", {nan, 7.0, negative_nan, -3.5, 0.0});
  add_cases(cases, "NaNs and infinity", {nan, inf});
  add_cases(cases, "NaNs and negative infinity", {negative_nan, -inf});

  // Exercise low-word ordering, including its unsigned sign boundary and high-word carries.
  const auto low_first = cuda::std::bit_cast<double>(cuda::std::uint64_t{0x3ff0000000000001ull});
  const auto low_mid   = cuda::std::bit_cast<double>(cuda::std::uint64_t{0x3ff000007fffffffull});
  const auto low_high  = cuda::std::bit_cast<double>(cuda::std::uint64_t{0x3ff0000080000000ull});
  const auto low_last  = cuda::std::bit_cast<double>(cuda::std::uint64_t{0x3ff00000ffffffffull});
  const auto next_high = cuda::std::bit_cast<double>(cuda::std::uint64_t{0x3ff0000100000000ull});
  add_cases(cases, "positive low words", {low_first, low_mid, low_high, low_last});
  add_cases(cases, "negative low words", {-low_first, -low_mid, -low_high, -low_last});
  add_cases(cases, "positive high-word carry", {low_last, next_high});
  add_cases(cases, "negative high-word carry", {-low_last, -next_high});
  add_cases(cases, "NaNs and low words", {nan, negative_nan, low_first, low_last});
  add_cases(cases, "NaNs and negative low words", {nan, negative_nan, -low_first, -low_last});

  // Deterministic pseudo-random finite values without a library-dependent distribution.
  unsigned state = 0x12345678u;
  for (unsigned sample = 0; sample < 8; ++sample)
  {
    std::vector<double> pattern;
    for (unsigned lane = 0; lane < warp_size; ++lane)
    {
      state = state * 1664525u + 1013904223u;
      pattern.push_back(static_cast<double>(static_cast<int>(state % 200001u) - 100000) / 32.0);
    }
    add_cases(cases, "random finite", pattern);
  }

  test_case* device_cases             = nullptr;
  cuda::std::uint64_t* device_results = nullptr;
  std::vector<cuda::std::uint64_t> results(cases.size() * warp_size);
  check_cuda(cudaMalloc(&device_cases, cases.size() * sizeof(test_case)));
  check_cuda(cudaMalloc(&device_results, results.size() * sizeof(cuda::std::uint64_t)));
  check_cuda(cudaMemcpy(device_cases, cases.data(), cases.size() * sizeof(test_case), cudaMemcpyHostToDevice));
  test_reduce_max_double<<<static_cast<unsigned>(cases.size()), warp_size>>>(device_cases, device_results);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
  check_cuda(
    cudaMemcpy(results.data(), device_results, results.size() * sizeof(cuda::std::uint64_t), cudaMemcpyDeviceToHost));
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
      const auto actual      = cuda::std::bit_cast<double>(actual_word);
      const auto matches     = cuda::std::isnan(expected)
                               ? cuda::std::isnan(actual)
                               : actual_word == cuda::std::bit_cast<cuda::std::uint64_t>(expected);
      if (!matches)
      {
        if (failures < 20)
        {
          std::fprintf(
            stderr,
            "%s: rotation=%u lane=%u expected=0x%016" PRIx64 " actual=0x%016" PRIx64 "\n",
            test.name,
            test.rotation,
            lane,
            cuda::std::bit_cast<cuda::std::uint64_t>(expected),
            actual_word);
        }
        ++failures;
      }
    }
  }
  std::printf("%zu cases, %zu lane checks, %zu failures\n", cases.size(), results.size(), failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
