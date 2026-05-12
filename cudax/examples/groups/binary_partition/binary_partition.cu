/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample illustrates basic usage of binary partition cooperative groups
 * within the thread block tile when divergent path exists.
 * 1.) Each thread loads a value from random array.
 * 2.) then checks if it is odd or even.
 * 3.) create binary partition group based on the above predicate
 * 4.) we count the number of odd/even in the group based on size of the binary
       groups
 * 5.) write it global counter of odd.
 * 6.) sum the values loaded by individual threads(using reduce) and write it to
       global even & odd elements sum.
 *
 * **NOTE** :
 *    binary_partition results in splitting warp into divergent thread groups
 *    this is not good from performance perspective, but in cases where warp
 *    divergence is inevitable one can use binary_partition group.
*/

#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include <cstdio>

namespace cudax = cuda::experimental;

template <class Group>
int sum(Group group, int value)
{
  const auto lane_mask = group.__synchronizer_instance().lane_mask();
  return __reduce_add_sync(lane_mask.get(), value);
}

void initOddEvenArr(int* inputArr, unsigned int size)
{
  for (int i = 0; i < size; i++)
  {
    inputArr[i] = rand() % 50;
  }
}

/**
 * CUDA kernel device code
 *
 * Creates cooperative groups and performs odd/even counting & summation.
 */
struct OddEvenCountAndSumKernel
{
  template <class Config>
  __device__ void operator()(Config config, int* inputArr, int* numOfOdds, int* sumOfOddAndEvens, unsigned int size)
  {
    cudax::this_grid grid{config};
    cudax::this_warp warp{config};

    for (auto i = cuda::gpu_thread.rank_as<unsigned>(grid); i < size; i += cuda::gpu_thread.count_as<unsigned>(grid))
    {
      int elem = inputArr[i];
      cudax::group sub_tile{
        cuda::gpu_thread,
        warp,
        cudax::binary_partition{[elem](auto) {
          return elem & 1;
        }},
        cudax::lane_synchronizer{}};
      if (sub_tile.rank(warp) == 0) // Odd numbers group
      {
        int oddGroupSum = sum(sub_tile, elem);

        if (cuda::gpu_thread.is_root_rank(sub_tile))
        {
          // Add number of odds present in this group of Odds.
          atomicAdd(numOfOdds, sub_tile.size());

          // Add local reduction of odds present in this group of Odds.
          atomicAdd(&sumOfOddAndEvens[0], oddGroupSum);
        }
      }
      else // Even numbers group
      {
        int evenGroupSum = sum(sub_tile, elem);

        if (cuda::gpu_thread.is_root_rank(sub_tile))
        {
          // Add local reduction of even present in this group of evens.
          atomicAdd(&sumOfOddAndEvens[1], evenGroupSum);
        }
      }
      // reconverge warp so for next loop iteration we ensure convergence of
      // above diverged threads to perform coalesced loads of inputArr.
      warp.sync();
    }
  }
};

/**
 * Host main routine
 */
int main(int argc, const char** argv)
{
  if (cuda::devices.size() == 0)
  {
    std::fprintf(stderr, "No CUDA devices found\n");
    return 1;
  }

  cuda::device_ref device = cuda::devices[0];
  cuda::stream stream{device};

  constexpr unsigned arrSize = 1024 * 100;

  auto h_inputArr          = cuda::make_pinned_buffer<int>(stream, arrSize, cuda::no_init);
  auto h_numOfOdds         = cuda::make_pinned_buffer<int>(stream, 1, cuda::no_init);
  auto h_sumOfOddEvenElems = cuda::make_pinned_buffer<int>(stream, 2, cuda::no_init);
  stream.sync();

  initOddEvenArr(h_inputArr.data(), arrSize);

  auto d_inputArr          = cuda::make_device_buffer<int>(stream, device, h_inputArr);
  auto d_numOfOdds         = cuda::make_device_buffer<int>(stream, device, 1, 0);
  auto d_sumOfOddEvenElems = cuda::make_device_buffer<int>(stream, device, 2, 0);

  // Launch the kernel
  const auto threadsPerBlock = 128u;
  const auto blocksPerGrid   = static_cast<unsigned>(
    cuda::device_attributes::multiprocessor_count(device)
    * cuda::device_attributes::max_blocks_per_multiprocessor(device));

  std::printf("\nLaunching %d blocks with %d threads...\n\n", blocksPerGrid, threadsPerBlock);

  const auto config = cuda::make_config(cuda::grid_dims(dim3{blocksPerGrid}), cuda::block_dims(blocksPerGrid));
  cuda::launch(
    stream,
    config,
    OddEvenCountAndSumKernel{},
    d_inputArr.data(),
    d_numOfOdds.data(),
    d_sumOfOddEvenElems.data(),
    arrSize);

  cuda::copy_bytes(stream, d_numOfOdds, h_numOfOdds);
  cuda::copy_bytes(stream, d_sumOfOddEvenElems, h_sumOfOddEvenElems);
  stream.sync();

  std::printf("Array size = %d Num of Odds = %d Sum of Odds = %d Sum of Evens %d\n",
              arrSize,
              h_numOfOdds.data()[0],
              h_sumOfOddEvenElems.data()[0],
              h_sumOfOddEvenElems.data()[1]);
  std::printf("\n...Done.\n\n");
}
