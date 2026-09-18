//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_SYNCHRONIZER_WRAPPER_CUH
#define _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_SYNCHRONIZER_WRAPPER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _SynchronizerInstance, class _Wrapper>

template <class _Synchronizer, class _Wrapper>
class synchronizer_wrapper
{
  static_assert(::cuda::std::is_move_constructible_v<_Synchronizer>, "_Synchronizer must be move constructible");
  static_assert(::cuda::std::is_move_constructible_v<_Wrapper>, "_Wrapper must be move constructible");

  _Synchronizer __synchronizer_;
  _Wrapper __wrapper_;

public:
  _CCCL_DEVICE_API synchronizer_wrapper(_Synchronizer __synchronizer, _Wrapper __wrapper) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Synchronizer>
    && ::cuda::std::is_nothrow_move_constructible_v<_Wrapper>)
      : __synchronizer_(__synchronizer)
      , __wrapper_(__wrapper)
  {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result)
  {}
};
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_SYNCHRONIZER_WRAPPER_CUH
