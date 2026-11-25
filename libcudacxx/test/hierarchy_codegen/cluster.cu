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

extern "C" __device__ ::cuda::std::uint32_t threads_in_cluster()
{
  return cuda::gpu_thread.count(cuda::cluster);
}

extern "C" __device__ ::cuda::std::uint32_t block_rank_in_cluster()
{
  return cuda::block.rank(cuda::cluster);
}

// A preferred cluster can differ from the regular cluster dimensions in the descriptor.
extern "C" __device__ ::cuda::std::uint32_t cluster_index_difference(::cuda::std::uint32_t grid_size)
{
  const auto hier = cuda::make_hierarchy(cuda::grid_dims(grid_size), cuda::cluster_dims<4>(), cuda::block_dims<128>());
  return cuda::cluster.index_as<::cuda::std::uint32_t>(cuda::grid, hier).x
       - cuda::cluster.index_as<::cuda::std::uint32_t>(cuda::grid).x;
}

extern "C" __device__ ::cuda::std::uint32_t block_index_difference(::cuda::std::uint32_t grid_size)
{
  const auto hier = cuda::make_hierarchy(cuda::grid_dims(grid_size), cuda::cluster_dims<4>(), cuda::block_dims<128>());
  return cuda::block.index(cuda::cluster, hier).x - cuda::block.index(cuda::cluster).x;
}

extern "C" __device__ ::cuda::std::uint32_t cluster_index_y_in_partial_grid(::cuda::std::uint32_t grid_size)
{
  const auto hier = cuda::make_hierarchy<cuda::cluster_level>(cuda::grid_dims(grid_size));
  return cuda::cluster.index_as<::cuda::std::uint32_t>(cuda::grid, hier).y;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : cluster_index_y_in_partial_grid
; SMXX-NOT: {{.*(SR_|LDC).*}}
; SMXX: {{.*(MOV|HFMA2[^ ]*) R[0-9]+, -?RZ.*}}
; SMXX-NEXT: {{.*RET.*}}

; Blackwell folds the two identical native coordinate queries into a constant.
; SM1XX-LABEL: {{[[:space:]]*}}Function : block_index_difference
; SM1XX-NOT: {{.*(SR_|LDC).*}}
; SM1XX: {{.*(MOV|HFMA2[^ ]*) R[0-9]+, -?RZ.*}}
; SM1XX-NEXT: {{.*RET.*}}

; SM1XX-LABEL: {{[[:space:]]*}}Function : cluster_index_difference
; SM1XX-NOT: {{.*(SR_|LDC).*}}
; SM1XX: {{.*(MOV|HFMA2[^ ]*) R[0-9]+, -?RZ.*}}
; SM1XX-NEXT: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : block_rank_in_cluster
; SMXX: {{.*S2U?R .*SR_CgaCtaId .*}}
; SMXX: {{.*RET.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : threads_in_cluster
; SM90: {{.*ULDC .*}}
; SM90a: {{.*ULDC .*}}
; SM1XX: {{.*SR_CgaSize .*}}
; SM1XX-NOT: {{.*SR_CgaSize .*}}
; SMXX-COUNT-3: {{.*UIMAD UR[0-9]+, UR[0-9]+, UR[0-9]+, URZ .*}}
; SMXX-NOT: {{.*UIMAD .*}}
; SMXX: {{.*RET.*}}

*/
