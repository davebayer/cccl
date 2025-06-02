//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_ROTATE_H
#define _LIBCUDACXX___BIT_ROTATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/bitmask_value_type.h>
#include <cuda/__type_traits/is_bitmask.h>
#include <cuda/std/__bit/rotate.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotl(_Tp __v, int __cnt) noexcept
{
  return _CUDA_VSTD::rotl(__v, __cnt);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_bitmask_v<_Tp>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotl(_Tp __v, int __cnt) noexcept
{
  return _Tp(_CUDA_VSTD::rotl(_CUDA_VSTD::__to_unsigned_like(static_cast<bitmask_value_type_t<_Tp>>(__v), __cnt)));
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotr(_Tp __v, int __cnt) noexcept
{
  return _CUDA_VSTD::rotr(__v, __cnt);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_bitmask_v<_Tp>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotr(_Tp __v, int __cnt) noexcept
{
  return _Tp(_CUDA_VSTD::rotr(_CUDA_VSTD::__to_unsigned_like(static_cast<bitmask_value_type_t<_Tp>>(__v), __cnt)));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___BIT_ROTATE_H
