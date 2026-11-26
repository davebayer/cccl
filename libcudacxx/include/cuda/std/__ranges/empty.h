// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_EMPTY_H
#define _CUDA_STD___RANGES_EMPTY_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

// [range.prim.empty]

_CCCL_BEGIN_NAMESPACE_CPO(__empty)

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_empty = __workaround_52970<_Tp> && requires(_Tp&& __t) { bool(__t.empty()); };

template <class _Tp>
concept __can_invoke_size = !__member_empty<_Tp> && requires(_Tp&& __t) { ::cuda::std::ranges::size(__t); };

template <class _Tp>
concept __can_compare_begin_end = !__member_empty<_Tp> && !__can_invoke_size<_Tp> && requires(_Tp&& __t) {
  bool(::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t));
  { ::cuda::std::ranges::begin(__t) } -> forward_iterator;
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
// Check for a higher-priority customization before substituting the fallback expression.
template <class _Tp, bool = __workaround_52970<_Tp>, class = void>
inline constexpr bool __member_empty = false;

template <class _Tp>
inline constexpr bool __member_empty<_Tp, true, void_t<decltype(bool(::cuda::std::declval<_Tp&>().empty()))>> = true;

template <class _Tp, bool = !__member_empty<_Tp>, class = void>
inline constexpr bool __can_invoke_size = false;

template <class _Tp>
inline constexpr bool
  __can_invoke_size<_Tp, true, void_t<decltype(::cuda::std::ranges::size(::cuda::std::declval<_Tp&>()))>> = true;

template <class _Tp, bool = !__member_empty<_Tp> && !__can_invoke_size<_Tp>, class = void>
inline constexpr bool __can_compare_begin_end = false;

template <class _Tp>
inline constexpr bool __can_compare_begin_end<
  _Tp,
  true,
  void_t<decltype(bool(::cuda::std::ranges::begin(::cuda::std::declval<_Tp&>())
                       == ::cuda::std::ranges::end(::cuda::std::declval<_Tp&>()))),
         enable_if_t<forward_iterator<iterator_t<_Tp>>>>> = true;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_empty<_Tp>)
  [[nodiscard]] _CCCL_API constexpr bool _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(noexcept(bool(__t.empty())))
  {
    return bool(__t.empty());
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_invoke_size<_Tp>)
  [[nodiscard]] _CCCL_API constexpr bool
  _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(noexcept(::cuda::std::ranges::size(__t)))
  {
    return ::cuda::std::ranges::size(__t) == 0;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_compare_begin_end<_Tp>)
  [[nodiscard]] _CCCL_API constexpr bool _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(
    noexcept(bool(::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t))))
  {
    return ::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t);
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto empty = __empty::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_EMPTY_H
