//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/warp>

struct alignas(4) OverAligned
{
  static constexpr auto N = 4;

  OverAligned() = default;

  __device__ OverAligned(unsigned v)
      : data_{(char) v, (char) v, (char) v}
  {}

  __device__ friend bool operator==(const OverAligned& lhs, const OverAligned& rhs)
  {
    for (int i = 0; i < N; ++i)
    {
      if (lhs.data_[i] != rhs.data_[i])
      {
        return false;
      }
    }
    return true;
  }

  char data_[N];
};

template <class T, int Width>
__device__ void test()
{
  const cuda::std::integral_constant<int, Width> width{};

  static_assert(noexcept(cuda::device::warp_shuffle_idx(cuda::std::declval<T>(), int{})));

  for (int src_idx = 0; src_idx < 32; ++src_idx)
  {
    const T data(threadIdx.x);
    {
      const auto result = cuda::device::warp_shuffle_idx(data, src_idx);
      assert(result.data == static_cast<T>(src_idx));
      assert(result.pred);
    }

    const T data_array[]{data, data, data};
    {
      const auto result = cuda::device::warp_shuffle_idx(data_array, src_idx);
      assert(result.data[0] == static_cast<T>(src_idx));
      assert(result.data[1] == static_cast<T>(src_idx));
      assert(result.data[2] == static_cast<T>(src_idx));
      assert(result.pred);
    }
  }
}

template <class T>
__device__ void test()
{
  static_assert(noexcept(cuda::device::warp_shuffle_idx(cuda::std::declval<T>(), int{})));

  const T data(threadIdx.x);
  const T data_array[]{data, data, data};

  // Test warp_shuffle_idx(T, src_idx).
  for (int src_idx = 0; src_idx < 32; ++src_idx)
  {
    const auto result = cuda::device::warp_shuffle_idx(data, src_idx);
    assert(result.data == static_cast<T>(src_idx));
    assert(result.pred);
  }

  // Test warp_shuffle_idx(T[N], src_idx).
  for (int src_idx = 0; src_idx < 32; ++src_idx)
  {
    const auto result = cuda::device::warp_shuffle_idx(data_array, src_idx);
    assert(result.data[0] == static_cast<T>(src_idx));
    assert(result.data[1] == static_cast<T>(src_idx));
    assert(result.data[2] == static_cast<T>(src_idx));
    assert(result.pred);
  }

  // Test warp_shuffle_idx(T, src_idx, lane_mask).
  const cuda::device::lane_mask lane_mask{0x0000'ffff};
  for (int src_idx = 0; src_idx < 32; ++src_idx)
  {
    const auto result = cuda::device::warp_shuffle_idx(data, src_idx, lane_mask);
    if (src_idx < 16 && (lane_mask & cuda::device::lane_mask::this_lane()) != cuda::device::lane_mask::none())
    {
      assert(result.data == static_cast<T>(src_idx));
      assert(result.pred);
    }
    else
    {
      assert(!result.pred);
    }
  }

  // test<T, 1>();
  // test<T, 2>();
  // test<T, 4>();
  // test<T, 8>();
  // test<T, 16>();
  // test<T, 32>();
}

__global__ void test_kernel()
{
  assert(blockDim.x == 32);

  test<char>();
  test<unsigned short>();
  test<int>();
  test<unsigned long long>();
  test<OverAligned>();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 test_kernel<<<1, 32>>>();
                 assert(cudaDeviceSynchronize() == cudaSuccess);
               }))
  return 0;
}
