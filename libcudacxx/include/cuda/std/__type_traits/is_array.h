//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_ARRAY)
template <class _Tp>
inline constexpr bool is_array_v = _CCCL_BUILTIN_IS_ARRAY(_Tp);

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_array : bool_constant<_CCCL_BUILTIN_IS_ARRAY(_Tp)>
{};

// clang prior to clang-19 and nvcc return true for arrays of size 0
#  if _CCCL_COMPILER(CLANG, <, 19) || _CCCL_CUDA_COMPILER(NVCC)
template <class _Tp>
inline constexpr bool is_array_v<_Tp[0]> = false;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_array<_Tp[0]> : false_type
{};
#  endif // _CCCL_COMPILER(CLANG, <, 19)
#else // ^^^ _CCCL_BUILTIN_IS_ARRAY ^^^ / vvv !_CCCL_BUILTIN_IS_ARRAY vvv
template <class _Tp>
inline constexpr bool is_array_v = false;

template <class _Tp>
inline constexpr bool is_array_v<_Tp[]> = true;

template <class _Tp, size_t _Np>
inline constexpr bool is_array_v<_Tp[_Np]> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_array : bool_constant<is_array_v<_Tp>>
{};
#endif // ^^^ !_CCCL_BUILTIN_IS_ARRAY ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H
