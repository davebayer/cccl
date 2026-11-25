//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/hierarchy>
#include <cuda/std/cstdint>
#include <cuda/std/span>

extern "C" __device__ ::cuda::std::uint32_t native_lane_index()
{
  return cuda::gpu_thread.index(cuda::warp).x;
}

extern "C" __device__ ::cuda::std::uint32_t lane_rank_in_1d_block()
{
  constexpr auto hier = cuda::make_hierarchy(cuda::grid_dims<1>(), cuda::block_dims<128>());
  return cuda::gpu_thread.rank(cuda::warp, hier);
}

extern "C" __device__ cuda::std::uint64_t thread_rank_in_block()
{
  return cuda::gpu_thread.rank_as<cuda::std::uint64_t>(cuda::block);
}

extern "C" __device__ cuda::std::uint64_t block_rank_in_grid()
{
  return cuda::block.rank(cuda::grid);
}

using mixed_grid = decltype(cuda::make_hierarchy(
  cuda::grid_dims<cuda::std::dynamic_extent, 3, 5>(), cuda::cluster_dims<2, 2, 1>(), cuda::block_dims<128>()));

extern "C" __device__ cuda::std::uint64_t block_rank_in_mixed_grid(const mixed_grid& hier)
{
  return cuda::block.rank(cuda::grid, hier);
}

extern "C" __device__ cuda::std::uint64_t thread_rank_in_1d_grid(::cuda::std::uint32_t grid_size, dim3 block_dims)
{
  const auto hier = cuda::make_hierarchy(cuda::grid_dims(grid_size), cuda::block_dims(block_dims));
  return cuda::gpu_thread.rank(cuda::grid, hier);
}

extern "C" __device__ cuda::std::uint64_t blocks_in_grid()
{
  return cuda::block.count(cuda::grid);
}

extern "C" __device__ ::cuda::std::uint32_t block_rank_in_1d_grid(::cuda::std::uint32_t grid_size)
{
  const auto hier = cuda::make_hierarchy(cuda::grid_dims(grid_size), cuda::block_dims<128>());
  return cuda::block.rank_as<::cuda::std::uint32_t>(cuda::grid, hier);
}

extern "C" __device__ ::cuda::std::uint32_t thread_index_y_in_1d_grid(::cuda::std::uint32_t grid_size)
{
  const auto hier = cuda::make_hierarchy(cuda::grid_dims(grid_size), cuda::block_dims<128>());
  return cuda::gpu_thread.index_as<::cuda::std::uint32_t>(cuda::grid, hier).y;
}

extern "C" __device__ ::cuda::std::uint32_t native_warps_in_block()
{
  return cuda::warp.count(cuda::block);
}

extern "C" __device__ ::cuda::std::uint32_t warps_in_1d_block(::cuda::std::uint32_t x)
{
  const auto hier = cuda::make_hierarchy(cuda::grid_dims<2>(), cuda::block_dims(x));
  return cuda::warp.count(cuda::block, hier);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : warps_in_1d_block
; SMXX: {{.*IADD.*-0x1.*}}
; SMXX: {{.*LEA.HI .*0x1b .*}}
; SMXX-NOT: {{.*MNMX.*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : native_warps_in_block
; SMXX: {{.*UIMAD .*}}
; SMXX: {{.*UIMAD .*0x1f .*}}
; SMXX: {{.*USHF.R.U32.HI .*0x5, .*}}
; SMXX-NOT: {{.*MNMX.*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : thread_index_y_in_1d_grid
; SMXX-NOT: {{.*SR_.*}}
; SMXX: {{.*(MOV|HFMA2[^ ]*) R[0-9]+, -?RZ.*}}
; SMXX-NEXT: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : block_rank_in_1d_grid
; SMXX-NOT: {{.*(SR_CTAID|IMAD|LDC).*}}
; SMXX: {{.*S2U?R .*SR_CTAID.X .*}}
; SMXX-NOT: {{.*(SR_CTAID|UIMAD|LDC).*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : blocks_in_grid
; SMXX-NOT: {{.*IMAD.WIDE.*}}
; SMXX: {{.*UIMAD .*}}
; SMXX: {{.*UIMAD.WIDE.U32 .*}}
; SMXX-NOT: {{.*(UIMAD|IADD).*}}
; SMXX: {{.*RET.*}}

; Two 32-bit operations each for the block size and local thread rank.
; SMXX-LABEL: {{[[:space:]]*}}Function : thread_rank_in_1d_grid
; SMXX-NOT: {{.*IMAD.WIDE.*}}
; SMXX-COUNT-4: {{.* IMAD .*}}
; SMXX-NOT: {{.*( IMAD |IADD).*}}
; SMXX: {{.* IMAD.WIDE.U32 .*}}
; SMXX-NEXT: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : block_rank_in_mixed_grid
; SMXX-NOT: {{.*( LD[G.]|IMAD.WIDE).*}}
; SMXX: {{.*UIMAD .*0x6, .*}}
; SMXX-NOT: {{.*( LD[G.]|IMAD.WIDE).*}}
; SMXX: {{.*UIMAD.WIDE.U32 .*}}
; SMXX-NOT: {{.*( LD[G.]|UIMAD|IADD).*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : block_rank_in_grid
; SMXX-NOT: {{.*IMAD.WIDE.*}}
; SMXX: {{.*UIMAD .*}}
; SMXX: {{.*UIMAD.WIDE.U32 .*}}
; SMXX-NOT: {{.*(UIMAD|IADD).*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : thread_rank_in_block
; SMXX-COUNT-2: {{.* IMAD .*}}
; SMXX-NOT: {{.*( IMAD |IADD).*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : lane_rank_in_1d_block
; SMXX: {{.*S2R .*SR_TID.X .*}}
; SMXX-NEXT: {{.*LOP3.LUT .*0x1f, .*}}
; SMXX-NEXT: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : native_lane_index
; SMXX: {{.*S2R .*SR_LANEID .*}}
; SMXX-NEXT: {{.*RET.*}}

*/
