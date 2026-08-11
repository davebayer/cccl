#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

namespace cudax = cuda::experimental;

__device__ void print_group_info(auto group)
{
  printf("[%d/%d]: %d/%d\n", threadIdx.x, blockDim.x, cuda::gpu_thread.rank(group), cuda::gpu_thread.count(group));
  if (threadIdx.x == 0)
  {
    printf("\n");
  }
}

struct Kernel
{
  __device__ void operator()(auto config) const
  {
    cudax::group g{cuda::gpu_thread, cudax::this_warp{config}, cudax::identity_mapping{}, cudax::lane_synchronizer{}};

    {
      cudax::group_view gv{g};
      print_group_info(gv);
      gv.sync();
    }

    {
      cudax::group_view gv{cuda::gpu_thread, g};
      print_group_info(gv);
      gv.sync();
    }

    {
      cudax::group_view gv{
        cuda::gpu_thread,
        g,
        cudax::group_by<16>{} | cudax::group_as{cuda::std::integer_sequence<cuda::std::size_t, 7, 5, 4>{}}};
      print_group_info(gv);
      gv.sync();
    }
  }
};

int main()
{
  cuda::stream stream{cuda::device_ref{0}};

  const auto config = cuda::make_config(cuda::grid_dims(1), cuda::block_dims<32>());
  cuda::launch(stream, config, Kernel{});
  stream.sync();
}
