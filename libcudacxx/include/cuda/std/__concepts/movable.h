//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CONCEPTS_MOVABLE_H
#define _CUDA_STD___CONCEPTS_MOVABLE_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/concept_macros.h> // IWYU pragma: keep
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/swappable.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Tp>
concept movable = is_object_v<_Tp> && move_constructible<_Tp> && assignable_from<_Tp&, _Tp> && swappable<_Tp>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

// These requirements guarantee that ranges::swap has a move fallback whenever ADL finds no swap.
template <class _Tp, class = void>
inline constexpr bool __movable = false;

template <class _Tp>
inline constexpr bool __movable<_Tp, void_t<enable_if_t<is_object_v<_Tp>>, enable_if_t<move_constructible<_Tp>>>> =
  assignable_from<_Tp&, _Tp>;

template <class _Tp>
_CCCL_CONCEPT movable = __movable<_Tp>;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_MOVABLE_H
