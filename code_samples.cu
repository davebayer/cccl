#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

namespace cuda
{
using namespace cuda::experimental;

namespace coop
{
auto sum(auto, auto);
}
} // namespace cuda

////////////////////////////////////////////////////////////////////////////////

__global__ void this_block_example(auto config)
{
  // construction​
  cuda::this_block block{config};

  // synchronization​
  block.sync(); // equivalent to __barrier_sync(0)​
  block.sync_aligned(); // equivalent to __syncthreads()​

  // queries​
  cuda::gpu_thread.rank(block);
  block.count(cuda::cluster);
  cuda::warp.is_part_of(block);

  // generic this group construction​
  auto block2 = cuda::make_this_group(cuda::block, config);

  // interoperability with CG​
  auto block3 = cuda::make_this_group(cg::this_block());
}

////////////////////////////////////////////////////////////////////////////////

template <class Hier, class T, size_t N>
__device__ auto sum(const cuda::this_thread<Hier>& g, T (&data)[N]);

template <class Hier, class T, size_t N>
__device__ auto sum(const cuda::this_warp<Hier>& g, T (&data)[N]);

template <class Hier, class T, size_t N>
__device__ auto sum(const cuda::this_block<Hier>& g, T (&data)[N]);

template <class Hier, class T, size_t N>
__device__ auto sum(const cuda::this_cluster<Hier>& g, T (&data)[N]);

__global__ void kernel(auto config)
{
  int data[]{/*...*/};

  auto thread_sum  = sum(cuda::this_thread{config}, data);
  auto warp_sum    = sum(cuda::this_warp{config}, data);
  auto block_sum   = sum(cuda::this_block{config}, data);
  auto cluster_sum = sum(cuda::this_cluster{config}, data);
}

////////////////////////////////////////////////////////////////////////////////

template <class Hier, class T, size_t N>
__device__ auto sum(const cuda::this_block<Hier>& g, T (&data)[N])
{
  constexpr auto nthreads = cuda::gpu_thread.static_count(g);

  T result;
  if constexpr (nthreads != cuda::std::dynamic_extent)
  {
    __shared__ T scratch[cuda::warp.static_count(g)];
    result = __sum_impl_static(g, data, scratch);
  }
  else
  {
    constexpr auto max_nwarps = /*...*/;
    __shared__ T scratch[max_nwarps];
    result = __sum_impl_dynamic(g, data, scratch);
  }

  return (cuda::gpu_thread.is_root_rank(g)) ? cuda::std::optional{result} : cuda::std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////

template <class Hier, class T, size_t N>
__device__ auto sum(const cuda::this_block<Hier>& g, T (&data)[N]);

struct Kernel
{
  __device__ void operator()(auto config)
  {
    int data[]{/*...*/};

    auto result = sum(cuda::this_block{config}, data);
  }
};

void fn(cuda::stream_ref stream)
{
  using namespace cuda;

  auto config1 = make_config(grid_dims<2>(), block_dims<128>());
  launch(stream, config1, Kernel{});

  // oops, instantiates `sum` again even though block_dims are the same.​
  auto config2 = make_config(grid_dims<3>(), block_dims<128>());
  launch(stream, config2, Kernel{});
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
auto DeviceSegmentedTransform(T* in, T* out, int nsegments, int segment_size, auto fn)
{
  if (segment_size > /*bound*/)
  {
    // launch clusters with 1 segment per cluster​
  }
  else if (segment_size > /*bound*/)
  {
    // launch blocks with 1 segment per block​
  }
  else if (segment_size > /*bound*/)
  {
    // launch blocks with 1 segment per warp​
  }
  else
  {
    // launch blocks with 1 segment per thread​
  }
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
__global__ void device_segmented_transform_kernel(auto config, T* in, T* out, int nsegments, int segment_size, auto fn)
{
  T thread_data[/*nelems_per_thread*/];

  // load elements from `in` to `thread_data​`

  cuda::std::optional<T> result;
  if (segment_size > /*bound*/)
  {
    result = fn(cuda::this_cluster{config}, thread_data);
  }
  else if (segment_size > /*bound*/)
  {
    result = fn(cuda::this_block{config}, thread_data);
  }
  else if (segment_size > /*bound*/)
  {
    result = fn(cuda::this_warp{config}, thread_data);
  }
  else
  {
    result = fn(cuda::this_thread{config}, thread_data);
  }

  // store the valid `result` in `out​`
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
void demo_device_segmented_transform_call(T* in, T* out, int nsegments, int segment_size)
{
  // the function must be invocable with any group​
  DeviceSegmentedTransform(in, out, nsegments, segment_size, [] __device__(auto group, auto& thread_data) {
    // the right cuda::coop::sum overload is selected based on the group type​
    return cuda::coop::sum(group, thread_data);
  });
}

////////////////////////////////////////////////////////////////////////////////
