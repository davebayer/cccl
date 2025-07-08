//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_LAMBDA_TRAITS_H
#define _CUDA___FUNCTIONAL_LAMBDA_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/cuda_capabilities.h>

#if _CCCL_HAS_EXTENDED_LAMBDA()

#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#  if _CCCL_CUDA_COMPILER(NVHPC)
template <class _Tp>
inline constexpr bool is_device_lambda_v = true;
template <class _Tp>
inline constexpr bool is_host_device_lambda_v = true;
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
template <class _Tp>
inline constexpr bool is_device_lambda_v = __nv_is_extended_device_lambda_closure_type(_Tp);
template <class _Tp>
inline constexpr bool is_host_device_lambda_v = __nv_is_extended_host_device_lambda_closure_type(_Tp);
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^

template <class _Tp>
struct is_device_lambda : _CUDA_VSTD::bool_constant<is_device_lambda_v<_Tp>>
{};
template <class _Tp>
struct is_host_device_lambda : _CUDA_VSTD::bool_constant<is_host_device_lambda_v<_Tp>>
{};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_EXTENDED_LAMBDA()

#endif // _CUDA___FUNCTIONAL_LAMBDA_TRAITS_H
