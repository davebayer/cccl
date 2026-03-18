// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ATOMIC_ATOMIC_MACROS_H
#define _CUDA___ATOMIC_ATOMIC_MACROS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>

#include <cuda/std/__cccl/prologue.h>

enum class __cccl_atomic_order
{
  __relaxed,
  __consume,
  __acquire,
  __release,
  __acq_rel,
  __seq_cst,
};

enum class __cccl_atomic_scope
{
  __threads,
  __block,
  __cluster,
  __device,
  __system,
};

#if _CCCL_HAS_NV_ATOMIC_BUILTINS()
[[nodiscard]] _CCCL_API constexpr auto __cccl_atomic_order_to_nv(__cccl_atomic_order __order) noexcept
{
  switch (__order)
  {
    case __cccl_atomic_order::__relaxed:
      return __NV_ATOMIC_RELAXED;
    case __cccl_atomic_order::__consume:
      return __NV_ATOMIC_CONSUME;
    case __cccl_atomic_order::__acquire:
      return __NV_ATOMIC_ACQUIRE;
    case __cccl_atomic_order::__release:
      return __NV_ATOMIC_RELEASE;
    case __cccl_atomic_order::__acq_rel:
      return __NV_ATOMIC_ACQ_REL;
    case __cccl_atomic_order::__seq_cst:
      return __NV_ATOMIC_SEQ_CST;
  }
}

[[nodiscard]] constexpr auto __cccl_atomic_scope_to_nv(__cccl_atomic_scope __scope) noexcept
{
  switch (__scope)
  {
    case __cccl_atomic_scope::__threads:
      return __NV_THREAD_SCOPE_THREAD;
    case __cccl_atomic_scope::__block:
      return __NV_THREAD_SCOPE_BLOCK;
    case __cccl_atomic_scope::__cluster:
      return __NV_THREAD_SCOPE_CLUSTER;
    case __cccl_atomic_scope::__device:
      return __NV_THREAD_SCOPE_DEVICE;
    case __cccl_atomic_scope::__system:
      return __NV_THREAD_SCOPE_SYSTEM;
  }
}

#  define _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_add(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_FETCH_SUB(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_sub(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_FETCH_AND(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_and(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_FETCH_OR(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_or(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_FETCH_XOR(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_xor(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_FETCH_MIN(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_min(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_FETCH_MAX(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_fetch_max(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))

#  define _CCCL_NV_ATOMIC_ADD(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_add(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_SUB(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_sub(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_AND(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_and(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_OR(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_or(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_XOR(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_xor(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_MIN(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_min(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#  define _CCCL_NV_ATOMIC_MAX(_ADDR, _VAL, _ORDER, _SCOPE) \
    __nv_atomic_min(_ADDR, _VAL, __cccl_atomic_order_to_nv(_ORDER), __cccl_atomic_scope_to_nv(_SCOPE))
#else
enum class __cccl_atomic_op
{
  __add,
  __sub,
  __and,
  __or,
  __xor,
  __min,
  __max,
};

#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 5) || _CCCL_CUDA_COMPILER(NVRTC, >=, 12, 5)
template <__cccl_atomic_op _Op>
struct __cccl_atomic_op_name;
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__add>
{
  static constexpr char __name[]{".add"};
};
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__sub>
{
  static constexpr char __name[]{".sub"};
};
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__and>
{
  static constexpr char __name[]{".and"};
};
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__or>
{
  static constexpr char __name[]{".or"};
};
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__xor>
{
  static constexpr char __name[]{".xor"};
};
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__min>
{
  static constexpr char __name[]{".min"};
};
template <>
struct __cccl_atomic_op_name<__cccl_atomic_op::__max>
{
  static constexpr char __name[]{".max"};
};

template <__cccl_atomic_order _Ord>
struct __cccl_atomic_order_name;
template <>
struct __cccl_atomic_order_name<__cccl_atomic_order::__relaxed>
{
  static constexpr char __name[]{".relaxed"};
};
template <>
struct __cccl_atomic_order_name<__cccl_atomic_order::__consume>
{
  static constexpr char __name[]{".consume"};
};
template <>
struct __cccl_atomic_order_name<__cccl_atomic_order::__acquire>
{
  static constexpr char __name[]{".acquire"};
};
template <>
struct __cccl_atomic_order_name<__cccl_atomic_order::__release>
{
  static constexpr char __name[]{".release"};
};
template <>
struct __cccl_atomic_order_name<__cccl_atomic_order::__acq_rel>
{
  static constexpr char __name[]{".acq_rel"};
};

template <__cccl_atomic_scope _Ord>
struct __cccl_atomic_scope_name;
template <>
struct __cccl_atomic_scope_name<__cccl_atomic_scope::__threads>
{
  static constexpr char __name[]{".cta"};
};
template <>
struct __cccl_atomic_scope_name<__cccl_atomic_scope::__block>
{
  static constexpr char __name[]{".cta"};
};
template <>
struct __cccl_atomic_scope_name<__cccl_atomic_scope::__cluster>
{
  static constexpr char __name[]{".cluster"};
};
template <>
struct __cccl_atomic_scope_name<__cccl_atomic_scope::__device>
{
  static constexpr char __name[]{".gpu"};
};
template <>
struct __cccl_atomic_scope_name<__cccl_atomic_scope::__system>
{
  static constexpr char __name[]{".sys"};
};

template <__cccl_atomic_op _Op, __cccl_atomic_order _Ord, __cccl_atomic_scope _Sco, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __cccl_nv_atomic_fetch_impl(_Tp* __addr, _Tp __val) noexcept
{
  _Tp __ret;
  static constexpr char type[]{".s32"}; // todo: allow different types
  asm volatile(
    "atom%1%2%3%4 %0, [%5], %6;"
    : "=r"(__ret)
    : "C"(__cccl_atomic_op_name<_Op>::__name),
      "C"(__cccl_atomic_order_name<_Ord>::__name),
      "C"(__cccl_atomic_scope_name<_Sco>::__name),
      "C"(type),
      "l"(__addr),
      "r"(__val)
    : "memory");
  return __ret;
}
#  else

#  endif

#  define _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__add, _ORDER, _SCOPE>(_ADDR, _VAL)
#  define _CCCL_NV_ATOMIC_FETCH_SUB(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__sub, _ORDER, _SCOPE>(_ADDR, _VAL)
#  define _CCCL_NV_ATOMIC_FETCH_AND(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__and, _ORDER, _SCOPE>(_ADDR, _VAL)
#  define _CCCL_NV_ATOMIC_FETCH_OR(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__or, _ORDER, _SCOPE>(_ADDR, _VAL)
#  define _CCCL_NV_ATOMIC_FETCH_XOR(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__xor, _ORDER, _SCOPE>(_ADDR, _VAL)
#  define _CCCL_NV_ATOMIC_FETCH_MIN(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__min, _ORDER, _SCOPE>(_ADDR, _VAL)
#  define _CCCL_NV_ATOMIC_FETCH_MAX(_ADDR, _VAL, _ORDER, _SCOPE) \
    __cccl_nv_atomic_fetch_impl<__cccl_atomic_op::__max, _ORDER, _SCOPE>(_ADDR, _VAL)

#  define _CCCL_NV_ATOMIC_ADD(_ADDR, _VAL, _ORDER, _SCOPE) (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#  define _CCCL_NV_ATOMIC_SUB(_ADDR, _VAL, _ORDER, _SCOPE) (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#  define _CCCL_NV_ATOMIC_AND(_ADDR, _VAL, _ORDER, _SCOPE) (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#  define _CCCL_NV_ATOMIC_OR(_ADDR, _VAL, _ORDER, _SCOPE)  (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#  define _CCCL_NV_ATOMIC_XOR(_ADDR, _VAL, _ORDER, _SCOPE) (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#  define _CCCL_NV_ATOMIC_MIN(_ADDR, _VAL, _ORDER, _SCOPE) (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#  define _CCCL_NV_ATOMIC_MAX(_ADDR, _VAL, _ORDER, _SCOPE) (void) _CCCL_NV_ATOMIC_FETCH_ADD(_ADDR, _VAL, _ORDER, _SCOPE)
#endif

__global__ void kernel(int* ptr, int* old)
{
  __shared__ int v;
  *old = _CCCL_NV_ATOMIC_FETCH_ADD(&v, 10, __cccl_atomic_order::__relaxed, __cccl_atomic_scope::__device);
  _CCCL_NV_ATOMIC_ADD(&v, 10, __cccl_atomic_order::__relaxed, __cccl_atomic_scope::__device);
}

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ATOMIC_ATOMIC_MACROS_H
