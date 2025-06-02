//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___TYPE_TRAITS_BITMASK_VALUE_TYPE_H
#define _CUDA___TYPE_TRAITS_BITMASK_VALUE_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_bitmask.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __cccl_bitmask_value_type_impl() noexcept
{
  if constexpr (_CUDA_VSTD::is_enum_v<_Tp>)
  {
    return _CUDA_VSTD::underlying_type_t<_Tp>{};
  }
  else
  {
    return typename _Tp::value_type{};
  }
}

template <class _Tp>
struct bitmask_value_type
{
  static_assert(is_bitmask_v<_Tp>, "bitmask_value_type requires a bitmask type");
  using type = decltype(::cuda::__cccl_bitmask_value_type_impl<_Tp>());
};

template <class _Tp>
using bitmask_value_type_t _CCCL_NODEBUG_ALIAS = typename bitmask_value_type<_Tp>::type;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___TYPE_TRAITS_BITMASK_VALUE_TYPE_H
