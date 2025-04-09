//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_RECONSTRUCT_H
#define _LIBCUDACXX___FLOATING_POINT_RECONSTRUCT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__floating_point/properties.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, size_t _UMantNBits, class _Up>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __fp_reconstruct(bool __sign, int __exp, _Up __mant)
{
  constexpr auto __fmt = __fp_format_of<_Tp>;
}

// template <class _Tp, size_t _UMantNBits, class _Up>
// [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __fp_reconstruct(int __exp, _Up __mant)
// {

// }

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FLOATING_POINT_RECONSTRUCT_H
