// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_ITER_MOVE_H
#define _CUDA_STD___ITERATOR_ITER_MOVE_H

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
#include <cuda/std/__concepts/referenceable.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wvoid-ptr-dereference")

// [iterator.cust.move]

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES
_CCCL_BEGIN_NAMESPACE_CPO(__iter_move)

_CCCL_API void iter_move();

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __unqualified_iter_move =
  __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) { iter_move(::cuda::std::forward<_Tp>(__t)); };

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, bool = __class_or_enum<remove_cvref_t<_Tp>>>
inline constexpr bool __unqualified_iter_move = false;

template <class _Tp>
inline constexpr bool __unqualified_iter_move<_Tp, true> =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp&& __t)(((void) iter_move(::cuda::std::forward<_Tp>(__t))));

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

template <class _Tp>
struct __deref_move_result
{
  using type _CCCL_NODEBUG = _Tp;
};

template <class _Tp>
struct __deref_move_result<_Tp&>
{
  using type _CCCL_NODEBUG = _Tp&&;
};

// [iterator.cust.move]

struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(__unqualified_iter_move<_Ip>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  _CCCL_STATIC_CALL_OPERATOR(_Ip&& __i) noexcept(noexcept(iter_move(::cuda::std::forward<_Ip>(__i))))
  {
    return iter_move(::cuda::std::forward<_Ip>(__i));
  }

  // Check for ADL before substituting the dereference expression.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Ip,
            class         = enable_if_t<!__unqualified_iter_move<_Ip>>,
            class _Result = __type<__deref_move_result<decltype(*::cuda::std::declval<_Ip>())>>>
  [[nodiscard]] _CCCL_API constexpr _Result
  _CCCL_STATIC_CALL_OPERATOR(_Ip&& __i) noexcept(noexcept(*::cuda::std::forward<_Ip>(__i)))
  {
    return static_cast<_Result>(*::cuda::std::forward<_Ip>(__i));
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto iter_move = __iter_move::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()
template <__dereferenceable _Tp>
  requires requires(_Tp& __t) {
    { ::cuda::std::ranges::iter_move(__t) } -> __referenceable;
  }
using iter_rvalue_reference_t = decltype(::cuda::std::ranges::iter_move(::cuda::std::declval<_Tp&>()));

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__can_iter_rvalue_reference_t_,
                       requires(_Tp& __t)(requires(__dereferenceable<_Tp>),
                                          requires(__referenceable<decltype(::cuda::std::ranges::iter_move(__t))>)));

template <class _Tp>
_CCCL_CONCEPT __can_iter_rvalue_reference_t = _CCCL_FRAGMENT(__can_iter_rvalue_reference_t_, _Tp);

template <class _Tp>
using __iter_rvalue_reference_t _CCCL_NODEBUG = decltype(::cuda::std::ranges::iter_move(::cuda::std::declval<_Tp&>()));

template <class _Tp>
using iter_rvalue_reference_t = enable_if_t<__can_iter_rvalue_reference_t<_Tp>, __iter_rvalue_reference_t<_Tp>>;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_ITER_MOVE_H
