//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___TYPE_TRAITS_IS_BITMASK_H
#define _CUDA___TYPE_TRAITS_IS_BITMASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/lane_mask.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! Tells whether a type is a cv-qualified bitmask type.
//! Users are allowed to specialize this template for their own types.
template <class _Tp>
inline constexpr bool is_bitmask_v = false;
template <class _Tp>
inline constexpr bool is_bitmask_v<const _Tp> = is_bitmask_v<_Tp>;
template <class _Tp>
inline constexpr bool is_bitmask_v<volatile _Tp> = is_bitmask_v<_Tp>;
template <class _Tp>
inline constexpr bool is_bitmask_v<const volatile _Tp> = is_bitmask_v<_Tp>;

template <>
inline constexpr bool is_bitmask_v<_CUDA_DEVICE::lane_mask> = true;

// we define the trait as alias, so users cannot specialize it (they should specialize the variable template instead)
template <class _Tp>
using is_bitmask = _CUDA_VSTD::bool_constant<is_bitmask_v<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___TYPE_TRAITS_IS_BITMASK_H
