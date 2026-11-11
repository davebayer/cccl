// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <cuda/std/functional>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifndef LOOKAHEAD_GROUPS_VARIANT
#  define LOOKAHEAD_GROUPS_VARIANT "unknown"
#endif

void check(cudaError_t status)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename T>
class device_array
{
  T* data_{};

public:
  explicit device_array(std::size_t size)
  {
    check(cudaMalloc(&data_, std::max(size, std::size_t{1}) * sizeof(T)));
  }
  ~device_array()
  {
    // Destructors cannot report CUDA failures while handling another exception.
    static_cast<void>(cudaFree(data_));
  }
  device_array(const device_array&)            = delete;
  device_array& operator=(const device_array&) = delete;
  device_array(device_array&&)                 = delete;
  device_array& operator=(device_array&&)      = delete;
  [[nodiscard]] T* data() const
  {
    return data_;
  }
};

template <typename T>
__global__ void fill(T* input, std::size_t count)
{
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
  {
    // Each seven-element period sums to zero, so floating-point results are exact.
    input[i] = static_cast<T>(static_cast<int>(i % 7) - 3);
  }
}

template <typename T>
__global__ void validate(const T* output, std::size_t count, bool exclusive, int* errors)
{
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
  {
    const auto remainder = static_cast<int>((i + 1 - static_cast<std::size_t>(exclusive)) % 7);
    const int prefix     = remainder * (remainder - 7) / 2 + 11 * static_cast<int>(exclusive);
    const T expected     = static_cast<T>(prefix);
    if (output[i] != expected)
    {
      atomicAdd(errors, 1);
    }
  }
}

template <typename T>
cudaError_t scan(
  void* storage, std::size_t& bytes, const T* input, T* output, std::size_t count, bool exclusive, cudaStream_t stream)
{
  if (exclusive)
  {
    return cub::DeviceScan::ExclusiveScan(storage, bytes, input, output, cuda::std::plus<>{}, T{11}, count, stream);
  }
  return cub::DeviceScan::InclusiveScan(storage, bytes, input, output, cuda::std::plus<>{}, count, stream);
}

template <typename T>
void run_case(const char* type, std::size_t count, bool exclusive, int shift, bool timed)
{
  const device_array<T> input(count + 8);
  const device_array<T> output(count + 8);
  const device_array<int> errors(1);
  cudaStream_t stream{};
  check(cudaStreamCreate(&stream));
  fill<<<256, 256, 0, stream>>>(input.data() + shift, count);
  check(cudaGetLastError());
  check(cudaMemsetAsync(output.data(), 0x5a, (count + 8) * sizeof(T), stream));
  std::size_t bytes{};
  check(scan(nullptr, bytes, input.data() + shift, output.data() + shift, count, exclusive, stream));
  const device_array<unsigned char> storage(bytes);
  const auto invoke = [&] {
    check(scan(storage.data(), bytes, input.data() + shift, output.data() + shift, count, exclusive, stream));
  };
  invoke();
  check(cudaMemsetAsync(errors.data(), 0, sizeof(int), stream));
  validate<<<256, 256, 0, stream>>>(output.data() + shift, count, exclusive, errors.data());
  check(cudaGetLastError());
  int error_count{};
  check(cudaMemcpyAsync(&error_count, errors.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  if (error_count != 0)
  {
    throw std::runtime_error(std::string{"incorrect "} + type + " scan of " + std::to_string(count));
  }
  // Check the untouched output allocation on both sides of a shifted scan.
  std::array<unsigned char, sizeof(T)> guard{};
  check(cudaMemcpy(guard.data(), output.data() + count + shift, sizeof(T), cudaMemcpyDeviceToHost));
  if (!std::all_of(guard.begin(), guard.end(), [](unsigned char value) {
        return value == 0x5a;
      }))
  {
    throw std::runtime_error("scan overwrote its trailing guard");
  }
  if (shift > 0)
  {
    check(cudaMemcpy(guard.data(), output.data(), sizeof(T), cudaMemcpyDeviceToHost));
    if (!std::all_of(guard.begin(), guard.end(), [](unsigned char value) {
          return value == 0x5a;
        }))
    {
      throw std::runtime_error("scan overwrote its leading guard");
    }
  }
  if (timed)
  {
    // NOLINTNEXTLINE(readability-redundant-parentheses): repository instructions require parentheses on ternary
    // conditions.
    const int iterations = (count <= (1u << 20)) ? 64 : 8;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i)
    {
      invoke();
    }
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    // Short cases need sustained work before measurement to settle unlocked GPU clocks.
    float warmup_ms{};
    while (warmup_ms < 50.0f)
    {
      check(cudaEventRecord(start, stream));
      for (int i = 0; i < 16; ++i)
      {
        check(cudaGraphLaunch(executable, stream));
      }
      check(cudaEventRecord(stop, stream));
      check(cudaEventSynchronize(stop));
      float elapsed_ms{};
      check(cudaEventElapsedTime(&elapsed_ms, start, stop));
      warmup_ms += elapsed_ms;
    }
    std::array<float, 9> samples{};
    for (auto& sample : samples)
    {
      check(cudaEventRecord(start, stream));
      check(cudaGraphLaunch(executable, stream));
      check(cudaEventRecord(stop, stream));
      check(cudaEventSynchronize(stop));
      check(cudaEventElapsedTime(&sample, start, stop));
      sample *= 1000.0f / static_cast<float>(iterations);
    }
    std::sort(samples.begin(), samples.end());
    const auto median_us      = samples[samples.size() / 2];
    const double bandwidth    = 2.0 * static_cast<double>(count) * sizeof(T) / (median_us * 1000.0);
    const auto operation_name = [&]() -> const char* {
      if (exclusive)
      {
        return "exclusive";
      }
      return "inclusive";
    };
    std::printf(
      "%s,%s,%s,%zu,%d,%.6f,%.6f,%.6f,%.3f\n",
      LOOKAHEAD_GROUPS_VARIANT,
      type,
      operation_name(),
      count,
      shift,
      median_us,
      samples.front(),
      samples.back(),
      bandwidth);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
  }
  check(cudaStreamDestroy(stream));
}

struct affine
{
  std::uint32_t a;
  std::uint32_t b;
  __host__ __device__ bool operator==(affine other) const
  {
    return a == other.a && b == other.b;
  }
  __host__ __device__ bool operator!=(affine other) const
  {
    return !(*this == other);
  }
};

struct compose
{
  __host__ __device__ affine operator()(affine left, affine right) const
  {
    // Composition modulo 2^32 is associative and noncommutative.
    return {left.a * right.a, left.b * right.a + right.b};
  }
};

struct affine_policy
{
  [[nodiscard]] __host__ __device__ constexpr cub::ScanPolicy operator()(cuda::compute_capability cc) const
  {
#if __cccl_ptx_isa >= 860
    // Three warps also exercise a squad whose size is not a power of two.
    static_cast<void>(cc);
    cub::ScanPolicy policy{};
    policy.algorithm                            = cub::ScanAlgorithm::lookahead;
    policy.lookahead.reduce_and_scan_warps      = 3;
    policy.lookahead.items_per_thread           = 31;
    policy.lookahead.lookahead_items_per_thread = 4;
    return policy;
#else
    return cub::detail::scan::policy_selector_from_types<const affine*, affine*, affine, std::uint64_t, compose>{}(cc);
#endif
  }
};

template <typename T>
void check_in_place()
{
  constexpr std::size_t count = 65537;
  const device_array<T> data(count);
  const device_array<int> errors(1);
  for (const bool exclusive : {false, true})
  {
    fill<<<256, 256>>>(data.data(), count);
    check(cudaGetLastError());
    std::size_t bytes{};
    check(scan(nullptr, bytes, data.data(), data.data(), count, exclusive, nullptr));
    const device_array<unsigned char> storage(bytes);
    check(scan(storage.data(), bytes, data.data(), data.data(), count, exclusive, nullptr));
    check(cudaMemset(errors.data(), 0, sizeof(int)));
    validate<<<256, 256>>>(data.data(), count, exclusive, errors.data());
    check(cudaGetLastError());
    int error_count{};
    check(cudaMemcpy(&error_count, errors.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (error_count != 0)
    {
      throw std::runtime_error("in-place scan failed");
    }
  }
}

void check_noncommutative()
{
  for (const std::size_t count : {std::size_t{1}, std::size_t{31}, std::size_t{8193}, std::size_t{65537}})
  {
    std::vector<affine> input(count);
    std::vector<affine> output(count);
    for (std::size_t i = 0; i < count; ++i)
    {
      input[i] = {static_cast<std::uint32_t>(2 * (i % 3) + 1), static_cast<std::uint32_t>(i % 7)};
    }
    const device_array<affine> device_input(count);
    const device_array<affine> device_output(count);
    check(cudaMemcpy(device_input.data(), input.data(), count * sizeof(affine), cudaMemcpyHostToDevice));
    std::size_t bytes{};
    check(cub::detail::scan::dispatch(
      nullptr,
      bytes,
      device_input.data(),
      device_output.data(),
      compose{},
      cub::NullType{},
      count,
      nullptr,
      affine_policy{}));
    const device_array<unsigned char> storage(bytes);
    check(cub::detail::scan::dispatch(
      storage.data(),
      bytes,
      device_input.data(),
      device_output.data(),
      compose{},
      cub::NullType{},
      count,
      nullptr,
      affine_policy{}));
    check(cudaMemcpy(output.data(), device_output.data(), count * sizeof(affine), cudaMemcpyDeviceToHost));
    affine expected{1, 0};
    for (std::size_t i = 0; i < count; ++i)
    {
      expected = compose{}(expected, input[i]);
      if (output[i] != expected)
      {
        throw std::runtime_error("noncommutative scan failed at " + std::to_string(i));
      }
    }
    check(cub::detail::scan::dispatch(
      storage.data(),
      bytes,
      device_input.data(),
      device_output.data(),
      compose{},
      cub::detail::InputValue<affine>{affine{3, 7}},
      count,
      nullptr,
      affine_policy{}));
    check(cudaMemcpy(output.data(), device_output.data(), count * sizeof(affine), cudaMemcpyDeviceToHost));
    expected = {3, 7};
    for (std::size_t i = 0; i < count; ++i)
    {
      if (output[i] != expected)
      {
        throw std::runtime_error("noncommutative exclusive scan failed at " + std::to_string(i));
      }
      expected = compose{}(expected, input[i]);
    }
  }
}

template <typename T>
void run_type(const char* type, bool correctness_only)
{
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  using selector    = cub::detail::scan::policy_selector_from_types<const T*, T*, T, std::uint64_t, cuda::std::plus<>>;
  const auto policy = selector{}(cuda::compute_capability{properties.major, properties.minor});
  std::fprintf(
    stderr,
    "%s sum policy: %s; reduce/scan warps=%d, items/thread=%d\n",
    type,
    cub::detail::to_string(policy.algorithm),
    policy.lookahead.reduce_and_scan_warps,
    policy.lookahead.items_per_thread);
  if (!correctness_only && policy.algorithm != cub::ScanAlgorithm::lookahead)
  {
    throw std::runtime_error("benchmark must exercise lookahead scan");
  }
  check_in_place<T>();
  for (const bool exclusive : {false, true})
  {
    for (const std::size_t count : {
           std::size_t{0},
           std::size_t{1},
           std::size_t{2},
           std::size_t{31},
           std::size_t{32},
           std::size_t{33},
           std::size_t{1023},
           std::size_t{1025},
           std::size_t{8191},
           std::size_t{8192},
           std::size_t{8193},
           std::size_t{65537},
           std::size_t{1048579},
         })
    {
      run_case<T>(type, count, exclusive, 1, false);
    }
    if (!correctness_only)
    {
      for (const std::size_t count :
           {std::size_t{1024}, std::size_t{65536}, std::size_t{1048576}, std::size_t{16777216}, std::size_t{67108864}})
      {
        run_case<T>(type, count, exclusive, 0, true);
      }
      run_case<T>(type, 1048579, exclusive, 1, true);
    }
  }
}

void run_selected_case(const char* type, std::size_t count, bool exclusive, int shift, bool timed)
{
  if (std::strcmp(type, "int32") == 0)
  {
    run_case<int>(type, count, exclusive, shift, timed);
  }
  else if (std::strcmp(type, "float32") == 0)
  {
    run_case<float>(type, count, exclusive, shift, timed);
  }
  else if (std::strcmp(type, "float64") == 0)
  {
    run_case<double>(type, count, exclusive, shift, timed);
  }
  else
  {
    throw std::runtime_error("type must be int32, float32, or float64");
  }
}

int main(int argc, char** argv) // NOLINT(misc-const-correctness): standard main signature.
try
{
  const bool correctness_only = argc > 1 && std::strcmp(argv[1], "--correctness-only") == 0;
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  std::fprintf(
    stderr,
    "%s: %s, SM %d.%d, CUDA %d, C++ %ld\n",
    LOOKAHEAD_GROUPS_VARIANT,
    properties.name,
    properties.major,
    properties.minor,
    CUDART_VERSION,
    static_cast<long>(__cplusplus));
  std::fprintf(
    stderr,
    "device_properties: l2_cache_bytes=%d multiprocessor_count=%d shared_memory_per_multiprocessor_bytes=%zu\n",
    properties.l2CacheSize,
    properties.multiProcessorCount,
    properties.sharedMemPerMultiprocessor);
  std::printf("variant,type,operation,elements,shift,median_us,min_us,max_us,GB_per_s\n");
  if (argc > 1 && (std::strcmp(argv[1], "--case") == 0 || std::strcmp(argv[1], "--profile-case") == 0))
  {
    if (argc != 5 && argc != 6)
    {
      throw std::runtime_error("usage: --case|--profile-case TYPE inclusive|exclusive ELEMENTS [SHIFT]");
    }
    const bool exclusive = std::strcmp(argv[3], "exclusive") == 0;
    if (!exclusive && std::strcmp(argv[3], "inclusive") != 0)
    {
      throw std::runtime_error("operation must be inclusive or exclusive");
    }
    const auto count = static_cast<std::size_t>(std::stoull(argv[4]));
    int shift        = 0;
    if (argc == 6)
    {
      shift = std::stoi(argv[5]);
    }
    if (shift < 0 || shift > 7)
    {
      throw std::runtime_error("shift must be between zero and seven");
    }
    run_selected_case(argv[2], count, exclusive, shift, std::strcmp(argv[1], "--case") == 0);
    std::fprintf(stderr, "selected case correctness checks passed\n");
    return 0;
  }
  run_type<int>("int32", correctness_only);
  run_type<float>("float32", correctness_only);
  run_type<double>("float64", correctness_only);
  const auto policy = affine_policy{}(cuda::compute_capability{properties.major, properties.minor});
  std::fprintf(stderr,
               "affine correctness policy: %s; reduce/scan warps=%d, items/thread=%d\n",
               cub::detail::to_string(policy.algorithm),
               policy.lookahead.reduce_and_scan_warps,
               policy.lookahead.items_per_thread);
  check_noncommutative();
  std::fprintf(stderr, "all correctness checks passed\n");
  return 0;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
