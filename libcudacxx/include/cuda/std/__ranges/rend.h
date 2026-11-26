// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_REND_H
#define _CUDA_STD___RANGES_REND_H

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
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/rbegin.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

// [range.access.rend]

_CCCL_BEGIN_NAMESPACE_CPO(__rend)

#if _CCCL_COMPILER(GCC, <, 12) || _CCCL_COMPILER(MSVC)
// A deleted declaration suppresses ADL on these compilers.
_CCCL_HOST_DEVICE void rend();
#else // ^^^ _CCCL_COMPILER(GCC, <, 12) || _CCCL_COMPILER(MSVC) ^^^ / vvv deleted declaration vvv
_CCCL_HOST_DEVICE void rend() = delete;
#endif // _CCCL_COMPILER(GCC, <, 12) || _CCCL_COMPILER(MSVC)

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_rend = __can_borrow<_Tp> && __workaround_52970<_Tp> && requires(_Tp&& __t) {
  ::cuda::std::ranges::rbegin(__t);
  { _LIBCUDACXX_AUTO_CAST(__t.rend()) } -> sentinel_for<decltype(::cuda::std::ranges::rbegin(__t))>;
};

template <class _Tp>
concept __unqualified_rend =
  !__member_rend<_Tp> && __can_borrow<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    ::cuda::std::ranges::rbegin(__t);
    { _LIBCUDACXX_AUTO_CAST(rend(__t)) } -> sentinel_for<decltype(::cuda::std::ranges::rbegin(__t))>;
  };

template <class _Tp>
concept __can_reverse = __can_borrow<_Tp> && !__member_rend<_Tp> && !__unqualified_rend<_Tp> && requires(_Tp&& __t) {
  { ::cuda::std::ranges::begin(__t) } -> same_as<decltype(::cuda::std::ranges::end(__t))>;
  { ::cuda::std::ranges::begin(__t) } -> bidirectional_iterator;
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
// Check borrowing and customization priority before substituting the iterator expressions.
template <class _Tp>
using __rbegin_t _CCCL_NODEBUG = decltype(::cuda::std::ranges::rbegin(::cuda::std::declval<_Tp&>()));

template <class _Tp, bool = __can_borrow<_Tp> && __workaround_52970<_Tp>, class = void>
inline constexpr bool __member_rend = false;

template <class _Tp>
inline constexpr bool __member_rend<
  _Tp,
  true,
  void_t<
    __rbegin_t<_Tp>,
    enable_if_t<sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(::cuda::std::declval<_Tp&>().rend())), __rbegin_t<_Tp>>>>> =
  true;

template <class _Tp, bool = __can_borrow<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && !__member_rend<_Tp>, class = void>
inline constexpr bool __unqualified_rend = false;

template <class _Tp>
inline constexpr bool __unqualified_rend<
  _Tp,
  true,
  void_t<
    __rbegin_t<_Tp>,
    enable_if_t<sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(rend(::cuda::std::declval<_Tp&>()))), __rbegin_t<_Tp>>>>> =
  true;

template <class _Tp, bool = __can_borrow<_Tp> && !__member_rend<_Tp> && !__unqualified_rend<_Tp>, class = void>
inline constexpr bool __can_reverse = false;

template <class _Tp>
inline constexpr bool __can_reverse<
  _Tp,
  true,
  void_t<enable_if_t<same_as<iterator_t<_Tp>, decltype(::cuda::std::ranges::end(::cuda::std::declval<_Tp&>()))>>,
         enable_if_t<bidirectional_iterator<iterator_t<_Tp>>>>> = true;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

class __fn
{
public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_rend<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto
  _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.rend())))
  {
    return _LIBCUDACXX_AUTO_CAST(__t.rend());
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__unqualified_rend<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto
  _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(rend(__t))))
  {
    return _LIBCUDACXX_AUTO_CAST(rend(__t));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_reverse<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(
    noexcept(::cuda::std::make_reverse_iterator(::cuda::std::ranges::begin(__t))))
  {
    return ::cuda::std::make_reverse_iterator(::cuda::std::ranges::begin(__t));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((!__member_rend<_Tp> && !__unqualified_rend<_Tp> && !__can_reverse<_Tp>) )
  void _CCCL_STATIC_CALL_OPERATOR(_Tp&&) = delete;
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto rend = __rend::__fn{};
} // namespace __cpo

// [range.access.crend]

_CCCL_BEGIN_NAMESPACE_CPO(__crend)
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_lvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(
    noexcept(::cuda::std::ranges::rend(static_cast<const remove_reference_t<_Tp>&>(__t))))
    -> decltype(::cuda::std::ranges::rend(static_cast<const remove_reference_t<_Tp>&>(__t)))
  {
    return ::cuda::std::ranges::rend(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_rvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto
  _CCCL_STATIC_CALL_OPERATOR(_Tp&& __t) noexcept(noexcept(::cuda::std::ranges::rend(static_cast<const _Tp&&>(__t))))
    -> decltype(::cuda::std::ranges::rend(static_cast<const _Tp&&>(__t)))
  {
    return ::cuda::std::ranges::rend(static_cast<const _Tp&&>(__t));
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto crend = __crend::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_REND_H
