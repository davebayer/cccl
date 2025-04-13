//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDINT_INT128_H
#define _LIBCUDACXX___CSTDINT_INT128_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

class alignas(16) __uint128_impl
{
#if _CCCL_HAS_INT128()
  __uint128_t __value_;
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
  unsigned long long __lo_;
  unsigned long long __hi_;
#endif // _CCCL_HAS_INT128()

public:
  _CCCL_HIDE_FROM_ABI __uint128_impl() = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl(_Tp __v) noexcept
#if _CCCL_HAS_INT128()
      : __value_(__v){}
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      : __lo_(static_cast<unsigned long long>(__v))
      , __hi_(0ull)
  {
  }
#endif // _CCCL_HAS_INT128()

      _CCCL_HIDE_FROM_ABI __uint128_impl(const __uint128_impl&) = default;

  _CCCL_HIDE_FROM_ABI __uint128_impl& operator=(const __uint128_impl&) = default;

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl
  operator+(__uint128_impl __lhs, __uint128_impl __rhs) noexcept
  {
#if _CCCL_HAS_INT128()
    return __uint128_impl{__lhs.__value_ + __rhs.__value_};
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    __uint128_impl __result;

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // host: use __builtin_add_overflow or _addcarry_u64

      // device:
      // asm volatile("\tadd.cc.s64  %0, %2, %4;\n"
      //              "\taddc.cc.s64 %1, %3, %5;" : "=l"(__result.__lo_), "=l"(__result.__hi_) : "l"(__lhs.__lo_),
      //              "l"(__lhs.__hi_), "l"(__rhs.__lo_), "l"(__rhs.__hi_));
    }

    __result.__lo_ = __lhs.__lo_ + __rhs.__lo_;
    __result.__hi_ = __lhs.__hi_ + __rhs.__hi_ + (__result.__lo_ < __lhs.__lo_);
    return __result;
#endif // _CCCL_HAS_INT128()
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl
  operator-(__uint128_impl __lhs, __uint128_impl __rhs) noexcept
  {
#if _CCCL_HAS_INT128()
    return __uint128_impl{__lhs.__value_ - __rhs.__value_};
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    __uint128_impl __result;

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // host: use __builtin_sub_overflow or _subborrow_u64

      // device:
      // asm volatile("\tsub.cc.s64  %0, %2, %4;\n"
      //              "\tsubc.cc.s64 %1, %3, %5;" : "=l"(__result.__lo_), "=l"(__result.__hi_) : "l"(__lhs.__lo_),
      //              "l"(__lhs.__hi_), "l"(__rhs.__lo_), "l"(__rhs.__hi_));
    }

    __result.__lo_ = __lhs.__lo_ - __rhs.__lo_;
    __result.__hi_ = __lhs.__hi_ - __rhs.__hi_ - (__result.__lo_ > __lhs.__lo_);
    return __result;
#endif // _CCCL_HAS_INT128()
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl
  operator*(__uint128_impl __lhs, __uint128_impl __rhs) noexcept
  {
#if _CCCL_HAS_INT128()
    return __uint128_impl{__lhs.__value_ * __rhs.__value_};
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    __uint128_impl __result;

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // host: use __builtin_mul_overflow or _umul128

      // device:
      // asm volatile("\t{\n"
      //              "\t.reg .u64 t1, t2, t3, t4;\n"
      //              "\tmul.lo.s64 t1, %2, %5;\n"
      //              "\tmul.hi.u64 t2, %2, %4;\n"
      //              "\tadd.s64    t3, t2, t1;\n"
      //              "\tmul.lo.s64 t4, %3, %4;\n"
      //              "\tadd.s64    %1, t3, t4;\n"
      //              "\tmul.lo.s64 %0, %2, %4;\n"
      //              "\t}" : "=l"(__result.__lo_), "=l"(__result.__hi_) : "l"(__lhs.__lo_), "l"(__lhs.__hi_),
      //              "l"(__rhs.__lo_), "l"(__rhs.__hi_));
    }

    __result.__lo_ = __lhs.__lo_ - __rhs.__lo_;
    __result.__hi_ = __lhs.__hi_ - __rhs.__hi_ - (__result.__lo_ > __lhs.__lo_);
    return __result;
#endif // _CCCL_HAS_INT128()
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl
  operator/(__uint128_impl __lhs, __uint128_impl __rhs) noexcept
  {
#if _CCCL_HAS_INT128()
    return __uint128_impl{__lhs.__value_ / __rhs.__value_};
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    __uint128_impl __result;

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // host: todo

      // device: todo
    }

    // todo
    return __result;
#endif // _CCCL_HAS_INT128()
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl
  operator%(__uint128_impl __lhs, __uint128_impl __rhs) noexcept
  {
#if _CCCL_HAS_INT128()
    return __uint128_impl{__lhs.__value_ % __rhs.__value_};
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    __uint128_impl __result;

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // host: todo

      // device: todo
    }

    // todo
    return __result;
#endif // _CCCL_HAS_INT128()
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr __uint128_impl
  operator<<(__uint128_impl __lhs, int __n) noexcept
  {
#if _CCCL_HAS_INT128()
    return __uint128_impl{__lhs.__value_ << __n};
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    __uint128_impl __result;

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // host: todo

      // device: todo
    }

    // todo
    return __result;
#endif // _CCCL_HAS_INT128()
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDINT_INT128_H
