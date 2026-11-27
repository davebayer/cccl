//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CONCEPTS_COMMON_REFERENCE_WITH_H
#define _CUDA_STD___CONCEPTS_COMMON_REFERENCE_WITH_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h> // IWYU pragma: keep
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.commonref]

template <class _Tp, class _Up>
concept common_reference_with =
  same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>>
  && convertible_to<_Tp, common_reference_t<_Tp, _Up>> && convertible_to<_Up, common_reference_t<_Tp, _Up>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp,
          class _Up,
          class _Common = common_reference_t<_Tp, _Up>,
          class         = enable_if_t<same_as<_Common, common_reference_t<_Up, _Tp>>>>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool __common_reference_with(int) noexcept
{
  // Reference conversions between cv-qualified forms of the same type bind directly.
  // Checking implicit convertibility also covers the explicit cast required by convertible_to.
  if constexpr (is_reference_v<_Tp> && is_reference_v<_Up> && is_reference_v<_Common>
                && is_same_v<remove_cvref_t<_Tp>, remove_cvref_t<_Common>>
                && is_same_v<remove_cvref_t<_Up>, remove_cvref_t<_Common>>)
  {
    return is_convertible_v<_Tp, _Common> && is_convertible_v<_Up, _Common>;
  }
  else if constexpr (convertible_to<_Tp, _Common>)
  {
    return convertible_to<_Up, _Common>;
  }
  else
  {
    return false;
  }
}

template <class, class>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool __common_reference_with(long) noexcept
{
  return false;
}

template <class _Tp, class _Up>
_CCCL_CONCEPT common_reference_with = ::cuda::std::__common_reference_with<_Tp, _Up>(0);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_COMMON_REFERENCE_WITH_H
