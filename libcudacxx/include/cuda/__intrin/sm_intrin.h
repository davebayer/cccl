//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___INTRIN_SM_INTRIN_H
#define _CUDA___INTRIN_SM_INTRIN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME) \
    template <bool _False = false>                 \
    _CCCL_DEVICE typename __fail_unavailable_cuda_intrinsics<_False>::type _NAME(...)
#  define _CCCL_ADD_CUDA_INTRIN(_NAME) using ::_NAME

#  if __CUDA_ARCH__ >= 600
#    define _CCCL_ADD_CUDA_INTRIN_SM60(_NAME) _CCCL_ADD_CUDA_INTRIN(_NAME)
#  else // ^^^ __CUDA_ARCH__ >= 600 ^^^ / vvv __CUDA_ARCH__ < 600 vvv
#    define _CCCL_ADD_CUDA_INTRIN_SM60(_NAME) _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME)
#  endif // ^^^ __CUDA_ARCH__ < 600 ^^^

#  if __CUDA_ARCH__ >= 610
#    define _CCCL_ADD_CUDA_INTRIN_SM61(_NAME) _CCCL_ADD_CUDA_INTRIN(_NAME)
#  else // ^^^ __CUDA_ARCH__ >= 610 ^^^ / vvv __CUDA_ARCH__ < 610 vvv
#    define _CCCL_ADD_CUDA_INTRIN_SM61(_NAME) _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME)
#  endif // ^^^ __CUDA_ARCH__ < 610 ^^^

#  if __CUDA_ARCH__ >= 700
#    define _CCCL_ADD_CUDA_INTRIN_SM70(_NAME) _CCCL_ADD_CUDA_INTRIN(_NAME)
#  else // ^^^ __CUDA_ARCH__ >= 700 ^^^ / vvv __CUDA_ARCH__ < 700 vvv
#    define _CCCL_ADD_CUDA_INTRIN_SM70(_NAME) _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME)
#  endif // ^^^ __CUDA_ARCH__ < 700 ^^^

#  if __CUDA_ARCH__ >= 800
#    define _CCCL_ADD_CUDA_INTRIN_SM80(_NAME) _CCCL_ADD_CUDA_INTRIN(_NAME)
#  else // ^^^ __CUDA_ARCH__ >= 800 ^^^ / vvv __CUDA_ARCH__ < 800 vvv
#    define _CCCL_ADD_CUDA_INTRIN_SM80(_NAME) _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME)
#  endif // ^^^ __CUDA_ARCH__ < 800 ^^^

#  if __CUDA_ARCH__ >= 900
#    define _CCCL_ADD_CUDA_INTRIN_SM90(_NAME) _CCCL_ADD_CUDA_INTRIN(_NAME)
#  else // ^^^ __CUDA_ARCH__ >= 900 ^^^ / vvv __CUDA_ARCH__ < 900 vvv
#    define _CCCL_ADD_CUDA_INTRIN_SM90(_NAME) _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME)
#  endif // ^^^ __CUDA_ARCH__ < 900 ^^^

#  if __CUDA_ARCH__ >= 1000
#    define _CCCL_ADD_CUDA_INTRIN_SM100(_NAME) _CCCL_ADD_CUDA_INTRIN(_NAME)
#  else // ^^^ __CUDA_ARCH__ >= 1000 ^^^ / vvv __CUDA_ARCH__ < 1000 vvv
#    define _CCCL_ADD_CUDA_INTRIN_SM100(_NAME) _CCCL_ADD_UNAVAILABLE_CUDA_INTRIN(_NAME)
#  endif // ^^^ __CUDA_ARCH__ < 1000 ^^^
#else // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv _CCCL_CUDA_COMPILER(NVHPC) vvv
#  define _CCCL_ADD_CUDA_INTRIN(_NAME)       using ::_NAME
#  define _CCCL_ADD_CUDA_INTRIN_SM60(_NAME)  using ::_NAME
#  define _CCCL_ADD_CUDA_INTRIN_SM61(_NAME)  using ::_NAME
#  define _CCCL_ADD_CUDA_INTRIN_SM70(_NAME)  using ::_NAME
#  define _CCCL_ADD_CUDA_INTRIN_SM80(_NAME)  using ::_NAME
#  define _CCCL_ADD_CUDA_INTRIN_SM90(_NAME)  using ::_NAME
#  define _CCCL_ADD_CUDA_INTRIN_SM100(_NAME) using ::_NAME
#endif // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^

#if _CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_IF_TARGET(_COND) if target (_COND)
#else
#  define _CCCL_IF_TARGET(_COND)                                                                       \
    if constexpr (::nv::target::detail::toint(                                                         \
                    ::nv::target::detail::bitrounddown(::nv::target::sm_selector{__CUDA_ARCH__ / 10})) \
                  & ::nv::target::detail::toint(_COND))
#endif

namespace cuda::__intrin
{
template <bool _False>
struct __fail_unavailable_cuda_intrinsics
{
  static_assert(_False, "this CUDA intrinsic is not available for the current device target");
  using type = void;
};

// sm_20_atomic_functions.h

_CCCL_ADD_CUDA_INTRIN(atomicAdd);

// sm_20_intrinsics.h

_CCCL_ADD_CUDA_INTRIN(__threadfence_system);
_CCCL_ADD_CUDA_INTRIN(__ddiv_rn);
_CCCL_ADD_CUDA_INTRIN(__ddiv_rz);
_CCCL_ADD_CUDA_INTRIN(__ddiv_ru);
_CCCL_ADD_CUDA_INTRIN(__ddiv_rd);
_CCCL_ADD_CUDA_INTRIN(__drcp_rn);
_CCCL_ADD_CUDA_INTRIN(__drcp_rz);
_CCCL_ADD_CUDA_INTRIN(__drcp_ru);
_CCCL_ADD_CUDA_INTRIN(__drcp_rd);
_CCCL_ADD_CUDA_INTRIN(__dsqrt_rn);
_CCCL_ADD_CUDA_INTRIN(__dsqrt_rz);
_CCCL_ADD_CUDA_INTRIN(__dsqrt_ru);
_CCCL_ADD_CUDA_INTRIN(__dsqrt_rd);
_CCCL_ADD_CUDA_INTRIN(__ballot);
_CCCL_ADD_CUDA_INTRIN(__syncthreads_count);
_CCCL_ADD_CUDA_INTRIN(__syncthreads_and);
_CCCL_ADD_CUDA_INTRIN(__syncthreads_or);
_CCCL_ADD_CUDA_INTRIN(__fmaf_ieee_rn);
_CCCL_ADD_CUDA_INTRIN(__fmaf_ieee_rd);
_CCCL_ADD_CUDA_INTRIN(__fmaf_ieee_ru);
_CCCL_ADD_CUDA_INTRIN(__fmaf_ieee_rz);
_CCCL_ADD_CUDA_INTRIN(__double_as_longlong);
_CCCL_ADD_CUDA_INTRIN(__longlong_as_double);
_CCCL_ADD_CUDA_INTRIN(__fma_rn);
_CCCL_ADD_CUDA_INTRIN(__fma_rz);
_CCCL_ADD_CUDA_INTRIN(__fma_ru);
_CCCL_ADD_CUDA_INTRIN(__fma_rd);
_CCCL_ADD_CUDA_INTRIN(__dadd_rn);
_CCCL_ADD_CUDA_INTRIN(__dadd_rz);
_CCCL_ADD_CUDA_INTRIN(__dadd_ru);
_CCCL_ADD_CUDA_INTRIN(__dadd_rd);
_CCCL_ADD_CUDA_INTRIN(__dsub_rn);
_CCCL_ADD_CUDA_INTRIN(__dsub_rz);
_CCCL_ADD_CUDA_INTRIN(__dsub_ru);
_CCCL_ADD_CUDA_INTRIN(__dsub_rd);
_CCCL_ADD_CUDA_INTRIN(__dmul_rn);
_CCCL_ADD_CUDA_INTRIN(__dmul_rz);
_CCCL_ADD_CUDA_INTRIN(__dmul_ru);
_CCCL_ADD_CUDA_INTRIN(__dmul_rd);
_CCCL_ADD_CUDA_INTRIN(__double2float_rn);
_CCCL_ADD_CUDA_INTRIN(__double2float_rz);
_CCCL_ADD_CUDA_INTRIN(__double2float_ru);
_CCCL_ADD_CUDA_INTRIN(__double2float_rd);
_CCCL_ADD_CUDA_INTRIN(__double2int_rn);
_CCCL_ADD_CUDA_INTRIN(__double2int_ru);
_CCCL_ADD_CUDA_INTRIN(__double2int_rd);
_CCCL_ADD_CUDA_INTRIN(__double2uint_rn);
_CCCL_ADD_CUDA_INTRIN(__double2uint_ru);
_CCCL_ADD_CUDA_INTRIN(__double2uint_rd);
_CCCL_ADD_CUDA_INTRIN(__double2ll_rn);
_CCCL_ADD_CUDA_INTRIN(__double2ll_ru);
_CCCL_ADD_CUDA_INTRIN(__double2ll_rd);
_CCCL_ADD_CUDA_INTRIN(__double2ull_rn);
_CCCL_ADD_CUDA_INTRIN(__double2ull_ru);
_CCCL_ADD_CUDA_INTRIN(__double2ull_rd);
_CCCL_ADD_CUDA_INTRIN(__int2double_rn);
_CCCL_ADD_CUDA_INTRIN(__uint2double_rn);
_CCCL_ADD_CUDA_INTRIN(__ll2double_rn);
_CCCL_ADD_CUDA_INTRIN(__ll2double_rz);
_CCCL_ADD_CUDA_INTRIN(__ll2double_ru);
_CCCL_ADD_CUDA_INTRIN(__ll2double_rd);
_CCCL_ADD_CUDA_INTRIN(__ull2double_rn);
_CCCL_ADD_CUDA_INTRIN(__ull2double_rz);
_CCCL_ADD_CUDA_INTRIN(__ull2double_ru);
_CCCL_ADD_CUDA_INTRIN(__ull2double_rd);
_CCCL_ADD_CUDA_INTRIN(__double2hiint);
_CCCL_ADD_CUDA_INTRIN(__double2loint);
_CCCL_ADD_CUDA_INTRIN(__hiloint2double);
_CCCL_ADD_CUDA_INTRIN(ballot);
_CCCL_ADD_CUDA_INTRIN(syncthreads_count);
_CCCL_ADD_CUDA_INTRIN(syncthreads_and);
_CCCL_ADD_CUDA_INTRIN(syncthreads_or);
_CCCL_ADD_CUDA_INTRIN(__isGlobal);
_CCCL_ADD_CUDA_INTRIN(__isShared);
_CCCL_ADD_CUDA_INTRIN(__isConstant);
_CCCL_ADD_CUDA_INTRIN(__isLocal);
_CCCL_ADD_CUDA_INTRIN(__isGridConstant);
_CCCL_ADD_CUDA_INTRIN(__cvta_generic_to_global);
_CCCL_ADD_CUDA_INTRIN(__cvta_generic_to_shared);
_CCCL_ADD_CUDA_INTRIN(__cvta_generic_to_constant);
_CCCL_ADD_CUDA_INTRIN(__cvta_generic_to_local);
_CCCL_ADD_CUDA_INTRIN(__cvta_generic_to_grid_constant);
_CCCL_ADD_CUDA_INTRIN(__cvta_global_to_generic);
_CCCL_ADD_CUDA_INTRIN(__cvta_shared_to_generic);
_CCCL_ADD_CUDA_INTRIN(__cvta_constant_to_generic);
_CCCL_ADD_CUDA_INTRIN(__cvta_local_to_generic);
_CCCL_ADD_CUDA_INTRIN(__cvta_grid_constant_to_generic);
_CCCL_ADD_CUDA_INTRIN(__nv_bswap16);
_CCCL_ADD_CUDA_INTRIN(__nv_bswap32);
_CCCL_ADD_CUDA_INTRIN(__nv_bswap64);

// sm_30_intrinsics.
_CCCL_ADD_CUDA_INTRIN(__fns);
_CCCL_ADD_CUDA_INTRIN(__barrier_sync);
_CCCL_ADD_CUDA_INTRIN(__barrier_sync_count);
_CCCL_ADD_CUDA_INTRIN(__syncwarp);
_CCCL_ADD_CUDA_INTRIN(__all_sync);
_CCCL_ADD_CUDA_INTRIN(__any_sync);
_CCCL_ADD_CUDA_INTRIN(__uni_sync);
_CCCL_ADD_CUDA_INTRIN(__ballot_sync);
_CCCL_ADD_CUDA_INTRIN(__activemask);
_CCCL_ADD_CUDA_INTRIN(__shfl);
_CCCL_ADD_CUDA_INTRIN(__shfl_up);
_CCCL_ADD_CUDA_INTRIN(__shfl_down);
_CCCL_ADD_CUDA_INTRIN(__shfl_xor);
_CCCL_ADD_CUDA_INTRIN(__shfl_sync);
_CCCL_ADD_CUDA_INTRIN(__shfl_up_sync);
_CCCL_ADD_CUDA_INTRIN(__shfl_down_sync);
_CCCL_ADD_CUDA_INTRIN(__shfl_xor_sync);

// sm_32_atomic_functions.h

_CCCL_ADD_CUDA_INTRIN(atomicMin);
_CCCL_ADD_CUDA_INTRIN(atomicMax);
_CCCL_ADD_CUDA_INTRIN(atomicAnd);
_CCCL_ADD_CUDA_INTRIN(atomicOr);
_CCCL_ADD_CUDA_INTRIN(atomicXor);

// sm_32_intrinsics.h

_CCCL_ADD_CUDA_INTRIN(__ldg);
_CCCL_ADD_CUDA_INTRIN(__ldcg);
_CCCL_ADD_CUDA_INTRIN(__ldca);
_CCCL_ADD_CUDA_INTRIN(__ldcs);
_CCCL_ADD_CUDA_INTRIN(__ldlu);
_CCCL_ADD_CUDA_INTRIN(__ldcv);
_CCCL_ADD_CUDA_INTRIN(__stwb);
_CCCL_ADD_CUDA_INTRIN(__stcg);
_CCCL_ADD_CUDA_INTRIN(__stcs);
_CCCL_ADD_CUDA_INTRIN(__stwt);
_CCCL_ADD_CUDA_INTRIN(__funnelshift_l);
_CCCL_ADD_CUDA_INTRIN(__funnelshift_lc);
_CCCL_ADD_CUDA_INTRIN(__funnelshift_r);
_CCCL_ADD_CUDA_INTRIN(__funnelshift_rc);

// sm_35_atomic_functions.h

// sm_35_intrinsics.h

// sm_60_atomic_functions.h

_CCCL_ADD_CUDA_INTRIN_SM60(atomicAdd);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicAdd_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicAdd_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicSub_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicSub_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicExch_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicExch_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicMin_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicMin_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicMax_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicMax_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicInc_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicInc_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicDec_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicDec_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicCAS_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicCAS_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicAnd_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicAnd_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicOr_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicOr_system);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicXor_block);
_CCCL_ADD_CUDA_INTRIN_SM60(atomicXor_system);

// sm_61_intrinsics.h

_CCCL_ADD_CUDA_INTRIN_SM61(__dp2a_lo);
_CCCL_ADD_CUDA_INTRIN_SM61(__dp2a_hi);
_CCCL_ADD_CUDA_INTRIN_SM61(__dp4a);

// crt/sm_70_rt.h

_CCCL_ADD_CUDA_INTRIN_SM70(__match_any_sync);
_CCCL_ADD_CUDA_INTRIN_SM70(__match_all_sync);
_CCCL_ADD_CUDA_INTRIN_SM70(__nanosleep);
_CCCL_ADD_CUDA_INTRIN_SM70(atomicCAS);

// crt/sm_80_rt.h

_CCCL_ADD_CUDA_INTRIN_SM80(__reduce_add_sync);
_CCCL_ADD_CUDA_INTRIN_SM80(__reduce_min_sync);
_CCCL_ADD_CUDA_INTRIN_SM80(__reduce_max_sync);
_CCCL_ADD_CUDA_INTRIN_SM80(__reduce_and_sync);
_CCCL_ADD_CUDA_INTRIN_SM80(__reduce_or_sync);
_CCCL_ADD_CUDA_INTRIN_SM80(__reduce_xor_sync);
_CCCL_ADD_CUDA_INTRIN_SM80(__nv_associate_access_property);
_CCCL_ADD_CUDA_INTRIN_SM80(__nv_memcpy_async_shared_global_4);
_CCCL_ADD_CUDA_INTRIN_SM80(__nv_memcpy_async_shared_global_8);
_CCCL_ADD_CUDA_INTRIN_SM80(__nv_memcpy_async_shared_global_16);

// crt/sm_90_rt.h

_CCCL_ADD_CUDA_INTRIN_SM90(__isCtaShared);
_CCCL_ADD_CUDA_INTRIN_SM90(__isClusterShared);
_CCCL_ADD_CUDA_INTRIN_SM90(__cluster_map_shared_rank);
_CCCL_ADD_CUDA_INTRIN_SM90(__cluster_query_shared_rank);
_CCCL_ADD_CUDA_INTRIN_SM90(__cluster_map_shared_multicast);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterDimIsSpecified);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterDim);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterRelativeBlockIdx);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterGridDimInClusters);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterIdx);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterRelativeBlockRank);
_CCCL_ADD_CUDA_INTRIN_SM90(__clusterSizeInBlocks);
_CCCL_ADD_CUDA_INTRIN_SM90(__cluster_barrier_arrive);
_CCCL_ADD_CUDA_INTRIN_SM90(__cluster_barrier_arrive_relaxed);
_CCCL_ADD_CUDA_INTRIN_SM90(__cluster_barrier_wait);
_CCCL_ADD_CUDA_INTRIN_SM90(__threadfence_cluster);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicAdd);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicAdd_block);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicAdd_system);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicCAS);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicCAS_block);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicCAS_system);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicExch);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicExch_block);
_CCCL_ADD_CUDA_INTRIN_SM90(atomicExch_system);

// crt/sm_100_rt.h

_CCCL_ADD_CUDA_INTRIN_SM100(__ffma2_rn);
_CCCL_ADD_CUDA_INTRIN_SM100(__ffma2_rz);
_CCCL_ADD_CUDA_INTRIN_SM100(__ffma2_rd);
_CCCL_ADD_CUDA_INTRIN_SM100(__ffma2_ru);
_CCCL_ADD_CUDA_INTRIN_SM100(__fadd2_rn);
_CCCL_ADD_CUDA_INTRIN_SM100(__fadd2_rz);
_CCCL_ADD_CUDA_INTRIN_SM100(__fadd2_rd);
_CCCL_ADD_CUDA_INTRIN_SM100(__fadd2_ru);
_CCCL_ADD_CUDA_INTRIN_SM100(__fmul2_rn);
_CCCL_ADD_CUDA_INTRIN_SM100(__fmul2_rz);
_CCCL_ADD_CUDA_INTRIN_SM100(__fmul2_rd);
_CCCL_ADD_CUDA_INTRIN_SM100(__fmul2_ru);
} // namespace cuda::__intrin

#endif // _CUDA___INTRIN_SM_INTRIN_H
