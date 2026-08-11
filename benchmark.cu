#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

constexpr auto N              = 1;
constexpr auto BLOCK_SIZE     = 128;
constexpr auto FFTS_PER_BLOCK = 2;

namespace cudax = cuda::experimental;

using Hierarchy = decltype(cuda::make_hierarchy(cuda::grid_dims<1>(), cuda::block_dims<BLOCK_SIZE>()));

extern "C" __device__ auto fn(const Hierarchy& h, int (&thread_data)[N], int& ret)
{
  cudax::this_block block{h};
  cudax::group_view g{cuda::warp, block, cudax::group_by<2>{}};

  auto result = *cudax::coop::reduce(g, thread_data, cuda::std::plus<>{});
  if (cuda::gpu_thread.is_root_rank(g))
  {
    ret = result;
  }
}

namespace
{
constexpr unsigned int WARP_SIZE       = 32;
constexpr unsigned int NUM_THREADS     = BLOCK_SIZE;
constexpr unsigned int NUM_WARPS       = NUM_THREADS / WARP_SIZE;
constexpr unsigned int THREADS_PER_FFT = NUM_THREADS / FFTS_PER_BLOCK;
constexpr unsigned int WARPS_PER_FFT   = NUM_WARPS / FFTS_PER_BLOCK;

static_assert(NUM_WARPS >= FFTS_PER_BLOCK);
// ensure FFTS_PER_BLOCK is power of 2
static_assert((FFTS_PER_BLOCK & (FFTS_PER_BLOCK - 1)) == 0);

using WarpReduceT = cub::WarpReduce<int>;
} // namespace

extern "C" __device__ void cub_fft_block_reduce_sum(int (&input_rmem)[N], int& output)
{
  union Scratch
  {
    WarpReduceT::TempStorage warp_red_scratch;
    int v;
  };

  __shared__ Scratch scratch[10];

  // Pre-sum the per-thread array into a scalar
  int thread_sum = 0;
#pragma unroll
  for (unsigned int i = 0; i < N; ++i)
  {
    thread_sum += input_rmem[i];
  }

  const unsigned int lane = threadIdx.x % WARP_SIZE;
  const unsigned int warp = threadIdx.x / WARP_SIZE;

  const unsigned int fft_group_lane = warp % WARPS_PER_FFT;
  const unsigned int fft_group_warp = warp / FFTS_PER_BLOCK;

  auto warp_sum = WarpReduceT(scratch[warp].warp_red_scratch).Sum(thread_sum);

  // first thread of each warp writes its warp sum back into shared memory
  if (lane == 0)
  {
    scratch[warp].v = warp_sum;
  }

  __syncthreads();

  // if there is more than 1 warp per FFT group, compute inter-FFT group reduction
  // first N threads of first warp of each fft_group load values from shared and perform a warp reduce to get final
  // reduced values
  auto cross_warp_val =
    (lane < WARPS_PER_FFT && fft_group_lane == 0) ? scratch[lane + fft_group_warp * FFTS_PER_BLOCK].v : 0;

  __syncthreads();

  warp_sum = WarpReduceT(scratch[warp].warp_red_scratch).Sum(cross_warp_val);

  if (lane == 0 && fft_group_lane == 0)
  {
    output = warp_sum;
  }
}
