// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile with -D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX
// -D_CUB_WARPSPEED_USE_CUDAX_GROUPS and the CUB, libcu++, and cudax include paths.
#include <cub/detail/warpspeed/squad/squad.cuh>
#include <cub/util_debug.cuh>

#include <cuda/hierarchy>
#include <cuda/std/array>
#include <cuda/std/utility>

#include <cstdio>

namespace
{
inline constexpr int max_thread_count = 1024;
inline constexpr int rounds           = 8;

struct thread_result
{
  int squad;
  int thread_rank;
  int warp_rank;
  int lane_rank;
  int warp_leader;
  int squad_leader;
  int errors;
};

template <bool ReverseIds, cuda::std::size_t... Counts>
[[nodiscard]] __host__ __device__ constexpr cuda::std::array<cub::detail::warpspeed::SquadDesc, sizeof...(Counts)>
make_descriptors(cuda::std::index_sequence<Counts...>)
{
  constexpr cuda::std::array<int, sizeof...(Counts)> counts{static_cast<int>(Counts)...};
  cuda::std::array<cub::detail::warpspeed::SquadDesc, sizeof...(Counts)> squads{};
  for (cuda::std::size_t i = 0; i < counts.size(); ++i)
  {
    // Keep ternary conditions parenthesized consistently with the experiment.
    // NOLINTNEXTLINE(readability-redundant-parentheses)
    const int id = (ReverseIds) ? static_cast<int>(counts.size() - 1 - i) : static_cast<int>(i);
    squads[i]    = {id, counts[i]};
  }
  return squads;
}

template <bool StaticGroups, bool ReverseIds, cuda::std::size_t... Counts>
__global__ void test_squads(thread_result* results, int prefix_warps, cuda::std::index_sequence<Counts...> counts)
{
  __shared__ int values[max_thread_count];
  constexpr auto squads = make_descriptors<ReverseIds>(cuda::std::index_sequence<Counts...>{});
  const auto registers  = cub::detail::warpspeed::getSpecialRegisters();
  results[threadIdx.x]  = {-1, -1, -1, -1, 0, 0, 0};
  const auto test_squad = [&](auto squad) {
#if defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
    if constexpr (StaticGroups)
    {
      static_assert(decltype(squad)::warpCount() > 0);
      static_assert(decltype(squad)::threadCount() == 32 * decltype(squad)::warpCount());
      static_assert(cuda::warp.static_count(squad.mGroup) == decltype(squad)::warpCount());
    }
#endif // defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
    const int first_thread = static_cast<int>(threadIdx.x) - squad.threadRank();
    int errors             = 0;
    for (int round = 0; round < rounds; ++round)
    {
      values[threadIdx.x] = squad.threadRank() + round;
      squad.syncThreads();
      int sum = 0;
      for (int rank = 0; rank < squad.threadCount(); ++rank)
      {
        sum += values[first_thread + rank];
      }
      const int count = squad.threadCount();
      errors += (sum != count * (count - 1) / 2 + count * round) ? 1 : 0;
      squad.syncThreads();
    }
    results[threadIdx.x] = {
      squad.mSquadIdx,
      squad.threadRank(),
      squad.warpRank(),
      squad.laneRank(),
      static_cast<int>(squad.isLeaderThreadOfWarp()),
      static_cast<int>(squad.isLeaderThread()),
      errors,
    };
  };
#if defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
  if constexpr (StaticGroups)
  {
    cub::detail::warpspeed::squad_dispatch_static(registers, squads, counts, test_squad);
  }
  else
#endif // defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
  {
    cub::detail::warpspeed::squadDispatch(registers, squads, test_squad, prefix_warps);
  }
}
} // namespace

template <bool StaticGroups = false, bool ReverseIds = false, cuda::std::size_t... Counts>
[[nodiscard]] bool test_layout(cuda::std::index_sequence<Counts...> layout, int prefix_warps = 0, int suffix_warps = 0)
{
  constexpr int layout_warps = (static_cast<int>(Counts) + ...);
  const int thread_count     = 32 * (layout_warps + prefix_warps + suffix_warps);
  thread_result* device_results{};
  CubDebugExit(cudaMalloc(&device_results, thread_count * sizeof(thread_result)));
  test_squads<StaticGroups, ReverseIds><<<1, thread_count>>>(device_results, prefix_warps, layout);
  CubDebugExit(cudaGetLastError());
  cuda::std::array<thread_result, max_thread_count> results{};
  CubDebugExit(
    cudaMemcpy(results.data(), device_results, thread_count * sizeof(thread_result), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaFree(device_results));

  constexpr cuda::std::array<int, sizeof...(Counts)> counts{static_cast<int>(Counts)...};
  int start = 32 * prefix_warps;
  for (cuda::std::size_t squad = 0; squad < counts.size(); ++squad)
  {
    int squad_leaders = 0;
    for (int warp = 0; warp < counts[squad]; ++warp)
    {
      int warp_leaders = 0;
      for (int lane = 0; lane < 32; ++lane)
      {
        const int thread   = start + 32 * warp + lane;
        const auto& result = results[thread];
        // NOLINTNEXTLINE(readability-redundant-parentheses)
        const int expected_id = (ReverseIds) ? static_cast<int>(counts.size() - 1 - squad) : static_cast<int>(squad);
        if (result.squad != expected_id || result.thread_rank != 32 * warp + lane || result.warp_rank != warp
            || result.lane_rank != lane || result.errors != 0 || (result.squad_leader != 0 && warp != 0))
        {
          std::printf("Incorrect squad rank or synchronization at thread %d\n", thread);
          return false;
        }
        warp_leaders += result.warp_leader;
        squad_leaders += result.squad_leader;
      }
      if (warp_leaders != 1)
      {
        std::printf("Expected one elected leader in squad %zu, warp %d\n", squad, warp);
        return false;
      }
    }
    if (squad_leaders != 1)
    {
      std::printf("Expected one elected leader in squad %zu\n", squad);
      return false;
    }
    start += 32 * counts[squad];
  }
  for (int thread = 0; thread < thread_count; ++thread)
  {
    if ((thread < 32 * prefix_warps || thread >= 32 * (prefix_warps + layout_warps)) && results[thread].squad != -1)
    {
      std::printf("Thread %d outside dispatched subrange was modified\n", thread);
      return false;
    }
  }
  return true;
}

int main()
{
  using uneven_layout = cuda::std::index_sequence<1, 3, 2, 1>;
  if (!test_layout(uneven_layout{}) || !test_layout(uneven_layout{}, 1, 1)
      || !test_layout<false, true>(uneven_layout{}))
  {
    return 1;
  }
#if defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
  if (!test_layout<true>(uneven_layout{}) || !test_layout<true, true>(uneven_layout{})
      || !test_layout<true>(cuda::std::index_sequence<1, 1, 1, 1>{})
      || !test_layout<true>(cuda::std::index_sequence<2, 3, 1, 4>{})
      || !test_layout<true>(cuda::std::index_sequence<8, 8, 8, 8>{})
      || !test_layout<true>(cuda::std::index_sequence<1>{}))
  {
    return 1;
  }
  std::puts("PASS: static squads preserve compile-time sizes and synchronization for single, uniform, uneven, and "
            "1024-thread layouts");
#endif // defined(_CUB_WARPSPEED_USE_STATIC_CUDAX_GROUPS)
  std::puts("PASS: uneven squad ranks, elected leaders, repeated synchronization, and subrange dispatch");
}
