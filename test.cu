#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

namespace cudax = cuda::experimental;

struct group_view;

struct Kernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    int data[10]{};

    cudax::this_block block{config};

    {
      group_view view{block};
      cudax::coop::reduce(view, data, cuda::std::plus<>{});
    }

    {
      group_view g{cuda::warp, block, cudax::take<10>{}};
      cudax::coop::reduce(g, data, cuda::std::plus<>{});
    }

    {
      group_view g{cuda::warp, block, cudax::take<10>{}, cudax::unaligned};

      if (threadIdx.x % 2 == 1)
      {
        cudax::coop::reduce(g, data, cuda::std::plus<>{});
      }
      else
      {
        cudax::coop::reduce(g, data, cuda::std::plus<>{});
      }
    }
  }
};
