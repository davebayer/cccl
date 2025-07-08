//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_CUDA_CAPABILITIES
#define __CCCL_CUDA_CAPABILITIES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/cuda_toolkit.h>

#include <nv/target>

// True, when programmatic dependent launch is available, otherwise false.
#define _CCCL_HAS_PDL _CCCL_CTK_AT_LEAST(12, 0)

#if _CCCL_HAS_PDL
// Waits for the previous kernel to complete (when it reaches its final membar). Should be put before the first global
// memory access in a kernel.
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC() NV_IF_TARGET(NV_PROVIDES_SM_90, cudaGridDependencySynchronize();)
// Allows the subsequent kernel in the same stream to launch. Can be put anywhere in a kernel.
// Heuristic(ahendriksen): put it after the last load.
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH() NV_IF_TARGET(NV_PROVIDES_SM_90, cudaTriggerProgrammaticLaunchCompletion();)
#else
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC()
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH()
#endif // _CCCL_HAS_PDL

#if _CCCL_CUDA_COMPILATION() && (defined(__CUDACC_EXTENDED_LAMBDA__) || _CCCL_CUDA_COMPILER(NVHPC))
#  define _CCCL_HAS_EXTENDED_LAMBDA() 1
#else // ^^^ has extended lambda ^^^ / vvv no extended lambda vvv
#  define _CCCL_HAS_EXTENDED_LAMBDA() 0
#endif // ^^^ no extended lambda ^^^

#endif // __CCCL_CUDA_CAPABILITIES
