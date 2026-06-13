//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_TO_N_RESULT_H
#define _CUDA_STD___FORMAT_FORMAT_TO_N_RESULT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/format.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__utility/ctad_support.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _OutIt>
struct _CCCL_TYPE_VISIBILITY_DEFAULT format_to_n_result
{
  _OutIt out;
  iter_difference_t<_OutIt> size;
};
_CCCL_CTAD_SUPPORTED_FOR_TYPE(format_to_n_result);

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_TO_N_RESULT_H
