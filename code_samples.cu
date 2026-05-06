#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

namespace cuda
{
using namespace cuda::experimental;

template <int>
struct take
{};

namespace coop
{
auto sum(auto, auto);
}
} // namespace cuda

////////////////////////////////////////////////////////////////////////////////

__device__ int fn(int v)
{
  if (threadIdx.x < 8)
  {
    return __reduce_add_sync(0x000000ff, v);
  }
  if (threadIdx.x < 16)
  {
    return __reduce_min_sync(0x0000ff00, v);
  }
  if (threadIdx.x < 24)
  {
    return __reduce_and_sync(0x00ff0000, v);
  }
  return __reduce_or_sync(0xff000000, v);
}

__device__ int fn_groups(int v)
{
  auto g = /*group-by-8-in-warp-fn()*/;
  switch (g.group_rank())
  {
    case 0:
      return reduce_add(g, v);
    case 1:
      return reduce_min(g, v);
    case 2:
      return reduce_and(g, v);
    case 3:
      return reduce_or(g, v);
  }
}

////////////////////////////////////////////////////////////////////////////////

__device__ int sum(threads_in_warp_t g, int v)
{
  return __reduce_add_sync(g.mask(), v);
}

__device__ int sum(all_threads_in_block_t g, int v)
{
  __shared__ int scratch[n];

  scratch[warp_id] = sum(this_warp, v);
  g.sync();
  // sum values in smem​

  return scratch[0];
}

////////////////////////////////////////////////////////////////////////////////

namespace cg = cooperative_groups;

// 1. all threads in a block​
auto block = cg::this_thread_block();

// 2. all threads in a grid​
auto grid = cg::this_grid();

// 3. all threads in a warp​
auto warp = cg::coalesced_threads();

// 4. all threads in a cluster​
auto cluster = cg::this_cluster();

////////////////////////////////////////////////////////////////////////////////

namespace cg = cooperative_groups;

auto warp = cg::coalesced_threads();

// 1. tiled partition​
auto g1 = cg::tiled_partition<8>(warp);

// 2. labeled partition​
auto label = threadIdx.x % 4;
auto g2    = cg::labeled_partition(warp, label);

// 3. binary partition​
auto pred = (threadIdx.x % 2 == 0);
auto g3   = cg::binary_partition(warp, pred);

////////////////////////////////////////////////////////////////////////////////

namespace cg = cooperative_groups;

// 1. Queries​
{
  // thread index in grid​
  auto i1 = cg::this_grid().thread_index();

  // equivalent to (even when some of the dims are statically known)​
  auto i2 =
    (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
    + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  ​
}

// 2. Groups as API parameters​
constexpr auto dyn_n = static_cast<unsigned>(-1);

template <unsigned ntid_x = dyn_n, unsigned ntid_y = dyn_n, unsigned ntid_z = dyn_n>
auto my_algorithm(cg::thread_block block, ...);

template <unsigned ntid_x = dyn_n,
          unsigned ntid_y = dyn_n,
          unsigned ntid_z = dyn_n,
          unsigned nblk_x = dyn_n,
          unsigned nblk_y = dyn_n,
          unsigned nblk_z = dyn_n>
auto my_algorithm(cg::cluster_group cluster, ...);
​

  ////////////////////////////////////////////////////////////////////////////////

  __global__ void
  this_block_example(auto config)
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

__global__ void half_warp_demo_kernel(auto config)
{
  // construction
  cuda::group half_warp{cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<16>{}, cuda::lane_synchronizer{}};

  // sub-unit queries
  assert(cuda::gpu_thread.count(half_warp) == 16);
  assert(cuda::gpu_thread.rank(half_warp) == cuda::gpu_thread.rank(cuda::warp) % 16);

  // super-level queries
  assert(half_warp.count(cuda::warp) == 2);
  assert(half_warp.rank(cuda::warp) == cuda::gpu_thread.rank(cuda::warp) / 16);

  // synchronization
  half_warp.sync();
}

////////////////////////////////////////////////////////////////////////////////

__device__ auto make_quarter_warp(auto config)
{
  cuda::group half_warp{cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<16>{}, cuda::lane_synchronizer{}};

  cuda::group quarter_warp{cuda::gpu_thread, half_warp, cuda::group_by<8>{}, cuda::lane_synchronizer{}};

  // `half_warp` can be safely destroyed
  return quarter_warp;
}

////////////////////////////////////////////////////////////////////////////////

struct Kernel
{
  __device__ void operator()(auto config)
  {
    cuda::group half_warp{cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<16>{}, cuda::lane_synchronizer{}};

    static_assert(cuda::gpu_thread.static_count(half_warp) == 4);
    static_assert(half_warp.static_count(cuda::warp) == 2);
  }
};

void demo(cuda::stream_ref stream)
{
  using namespace cuda;

  auto config = make_config(grid_dims<2>(), block_dims<8, 8>());
  launch(stream, config, Kernel{});
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  cuda::group my_group{/*unit*/, /*parent-group*/, /*mapping*/, /*synchronizer*/};

  // What are we grouping?
  cuda::gpu_thread;
  cuda::warp;
  cuda::block;
  cuda::cluster;
  cuda::grid;

  // Within which group?
  cuda::this_block{...};
  cuda::group{...};

  // How should the original rank be mapped to the new group?
  cuda::group_by<10>{};
  cuda::group_as{2, 2, 8};
  cuda::take<16>{};
  cuda::take<16>{} | cuda::group_by<4> {}

  __shared__ int barriers[];
  // What synchronization mechanism should be used?
  cuda::barrier_synchronizer{barriers};
  cuda::lane_synchronizer{};
}

////////////////////////////////////////////////////////////////////////////////

struct MyMapping
{
  struct MyMappingResult
  {
    unsigned group_count;
    unsigned group_rank;
    unsigned count;
    unsigned rank;

    // ...
  };

  __device__ auto map(const auto& parent_group, auto prev_mapping_result)
  {
    MyMappingResult mapping_result;
    // compute `mapping_result` from `prev_mapping_result`
    return mapping_result;
  }
};

////////////////////////////////////////////////////////////////////////////////

struct MyMapping
{
  cuda::std::span<unsigned> smem;

  struct MyMappingResult
  {
    unsigned group_count;
    unsigned group_rank;
    unsigned count;
    unsigned rank;

    // ...
  };

  __device__ auto map(const auto& parent_group, auto prev_mapping_result)
  {
    // exchange mapping data via `smem`
    parent_group.sync();

    MyMappingResult mapping_result;
    // compute `mapping_result` from `prev_mapping_result` and `smem`
    return mapping_result;
  }
};

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  cuda::group half_warp_static{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<16>{},
    /*synchronizer*/
  };

  cuda::group half_warp_dynamic{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by{16},
    /*synchronizer*/
  };
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  cuda::group static_group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_as{cuda::std::integer_sequence<unsigned, 4, 4, 8, 16>{}},
    /*synchronizer*/
  };

  cuda::group dynamic_group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_as{4, 4, 8, 16},
    /*synchronizer*/
  };
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  cuda::group group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<4>{} | cuda::group_as{1, 3},
    /*synchronizer*/
  };
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept integer = true;

template <class T>
concept group_mapping_result = requires(const T& t) {
  // group count and rank
  { T::static_group_count() } -> integer;
  { t.group_count() } -> integer;
  { t.group_rank() } -> integer;

  // unit count and rank
  { T::static_count() } -> integer;
  { t.count() } -> integer;
  { t.rank() } -> integer;

  // properties
  { T::is_always_exhaustive() } -> std::same_as<bool>;
  { T::is_always_contiguous() } -> std::same_as<bool>;
  // ...
};

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  cuda::group group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<3>{},
    /*synchronizer*/
  };

  // oops, not true
  assert(cuda::gpu_thread.count(group) == 3);
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  // crashes
  cuda::group group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<3>{},
    /*synchronizer*/
  };
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  // doesn't crash
  cuda::group group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<3>{cuda::non_exhaustive},
    /*synchronizer*/
  };

  // oops, still not true
  assert(cuda::gpu_thread.count(group) == 3);
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  // doesn't crash
  cuda::group group{
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<3>{cuda::non_exhaustive},
    /*synchronizer*/
  };

  if (cuda::gpu_thread.is_part_of(group))
  {
    // passes
    assert(cuda::gpu_thread.count(group) == 3);
  }
  else
  {
    // last 2 threads are excluded
    assert(cuda::gpu_thread.rank(cuda::warp) > 30);
  }
}

////////////////////////////////////////////////////////////////////////////////

struct MySynchronizer
{
  struct MySynchronizerInstance
  { /*...*/
  };

  __device__ auto make_instance(const auto& unit, const auto& parent_group, const auto& mapping_result)
  {
    MySynchronizerInstance instance;
    // set up `instance`
    return instance;
  }
};

////////////////////////////////////////////////////////////////////////////////

using barrier_t = void;

struct MySynchronizer
{
  cuda::std::span<barrier_t> barriers;

  struct MySynchronizerInstance
  {
    // ...
  };

  __device__ auto make_instance(const auto& unit, const auto& parent_group, const auto& mapping_result)
  {
    if (mapping_result.rank() == 0)
    {
      // init `barrier[mapping_result.group_rank()]`
    }
    parent_group.sync();

    MySynchronizerInstance instance;
    // set up `instance`
    return instance;
  }
};

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  cuda::group group{cuda::gpu_thread,
                    cuda::this_warp{config},
                    /*mapping*/,
                    cuda::lane_synchronizer{}};

  // uses __syncwarp(mask) for synchronization
  group.sync();
}

////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(auto config)
{
  __shared__ cuda::barrier<cuda::thread_scope_block> barriers[/*N*/];

  cuda::group group{cuda::warp,
                    cuda::this_block{config},
                    /*mapping*/,
                    cuda::barrier_synchronizer{barriers}};

  // uses barrier.arrive_and_wait() for synchronization
  group.sync();
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class Group>
concept group_synchronizer_instance = requires(const T& t, const Group& g) {
  { t.do_sync(g) } -> std::same_as<void>;
  { t.do_sync_aligned(g) } -> std::same_as<void>;

  // todo: arrive + wait operations?
  // typename T::arrive_token_type;
  // { t.do_arrive(g) } -> std::same_as<typename T::arrive_token_type>;
  // { t.do_wait(g, typename T::arrive_token_type) } -> std::same_as<void>;
};

////////////////////////////////////////////////////////////////////////////////

template <class Group, class T, size_t N>
__device__ auto sum(const Group& g, T (&data)[N])
{
  using Unit  = typename Group::unit_type;
  using Level = typename Group::level_type;
  static_assert(cuda::std::is_same_v<Level, cuda::block_level>);

  constexpr Unit unit;
  auto partial = sum(cuda::make_this_group(unit, g.hierarchy()));

  T result;
  // sum partials among units

  return (unit.is_root_rank(g)) ? cuda::std::optional{result} : cuda::std::nullopt;
}

__global__ void kernel(auto config)
{
  cuda::group wgroup
  {
    cuda::gpu_thread, cuda::this_warp{config}, cuda::group_by<2>{}, cuda::lane_synchronizer {}
  }

  __shared__ cuda::barrier<cuda::thread_scope_block> barriers[16];
  cuda::group wgroup
  {
    cuda::warp, cuda::this_block{config}, cuda::group_by<2>{}, cuda::barrier_synchronizer{barriers};
  }

  int data[]{/*...*/};

  auto tg_sum = sum(wgroup, data);
  auto wg_sum = sum(wgroup, data);
}
