//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_MUL_OVERFLOW_H
#define _CUDA___NUMERIC_MUL_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_arithmetic_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

struct __mul_overflow
{
  template <class _DoubleTp, class _Tp>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp> __impl_upcasted(_Tp __lhs, _Tp __rhs)
  {
    const _DoubleTp __mul_result = static_cast<_DoubleTp>(__lhs) * static_cast<_DoubleTp>(__rhs);

    overflow_result<_Tp> __result{};
    __result.value = static_cast<_Tp>(__mul_result);
    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      __result.overflow = __mul_result > static_cast<_DoubleTp>(_CUDA_VSTD::numeric_limits<_Tp>::max())
                       || __mul_result < static_cast<_DoubleTp>(_CUDA_VSTD::numeric_limits<_Tp>::min());
    }
    else
    {
      __result.overflow = __mul_result > static_cast<_DoubleTp>(_CUDA_VSTD::numeric_limits<_Tp>::max());
    }
    return __result;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((sizeof(_Tp) <= 4))
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp>
  __impl_generic(_Tp __lhs, _Tp __rhs) noexcept
  {
    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        return __impl_upcasted<_CUDA_VSTD::int16_t>(__lhs, __rhs);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        return __impl_upcasted<_CUDA_VSTD::int32_t>(__lhs, __rhs);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        return __impl_upcasted<_CUDA_VSTD::int64_t>(__lhs, __rhs);
      }
      else
      {
        _CCCL_UNREACHABLE();
      }
    }
    else
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        return __impl_upcasted<_CUDA_VSTD::uint16_t>(__lhs, __rhs);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        return __impl_upcasted<_CUDA_VSTD::uint32_t>(__lhs, __rhs);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        return __impl_upcasted<_CUDA_VSTD::uint64_t>(__lhs, __rhs);
      }
      else
      {
        _CCCL_UNREACHABLE();
      }
    }
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((sizeof(_Tp) > 4))
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp>
  __impl_generic(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};
    __result.value = static_cast<_Tp>(__lhs * __rhs);

    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      if (__lhs >= _Tp{0})
      {
        if (__rhs >= _Tp{0})
        {
          __result.overflow = __lhs > _CUDA_VSTD::numeric_limits<_Tp>::max() / __rhs;
        }
        else if (__lhs != _Tp{0})
        {
          __result.overflow = __rhs < _CUDA_VSTD::numeric_limits<_Tp>::min() / __lhs;
        }
        else
        {
          __result.overflow = false;
        }
      }
      else
      {
        if (__rhs >= _Tp{0})
        {
          __result.overflow = __lhs < _CUDA_VSTD::numeric_limits<_Tp>::min() / __rhs;
        }
        else if (__rhs != _Tp{0})
        {
          __result.overflow = __lhs < _CUDA_VSTD::numeric_limits<_Tp>::max() / __rhs;
        }
        else
        {
          __result.overflow = true;
        }
      }
    }
    else
    {
      __result.overflow = __lhs > _CUDA_VSTD::numeric_limits<_Tp>::max() / __rhs;
    }

    return __result;
  }

#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  template <class _Tp>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr overflow_result<_Tp>
  __impl_builtin(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};
    __result.overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__lhs, __rhs, &__result.value);

    return __result;
  }
#endif // _CCCL_BUILTIN_MUL_OVERFLOW
#if !_CCCL_COMPILER(NVRTC)
  template <class _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST static overflow_result<_Tp> __impl_host(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};

    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        _CUDA_VSTD::int16_t __mul_result;
        __result.overflow = _mul_full_overflow_i8(__lhs, __rhs, &__mul_result);
        __result.value    = static_cast<int8_t>(__mul_result);
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC, >=, 19, 37) ||
        // !_CCCL_ARCH(X86_64) vvv
        __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC, >=, 19, 37) || !_CCCL_ARCH(X86_64) ^^^
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        __result.overflow = _mul_overflow_i16(__lhs, __rhs, &__result.value);
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC, >=, 19, 37) ||
        // !_CCCL_ARCH(X86_64) vvv
        __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC, >=, 19, 37) || !_CCCL_ARCH(X86_64) ^^^
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        __result.overflow = _mul_overflow_i32(__lhs, __rhs, &__result.value);
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        const int64_t __mul_result = __emul(__lhs, __rhs);
        __result.value             = static_cast<int32_t>(__mul_result);
        __result.overflow          = (__mul_result > _CUDA_VSTD::numeric_limits<int32_t>::max())
                         || (__mul_result < _CUDA_VSTD::numeric_limits<int32_t>::min());
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
        __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        __result.overflow = _mul_overflow_i64(__lhs, __rhs, &__result.value);
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        __result.value     = __lhs * __rhs;
        const int64_t __hi = __mulh(__lhs, __rhs);
        __result.overflow  = __hi != _Tp{0} && __hi != _Tp{-1};
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
        __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
      }
      else
      {
        __result = __impl_generic(__lhs, __rhs);
      }
    }
    else
    {
      if constexpr (sizeof(_Tp) == 1)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        _CUDA_VSTD::uint16_t __mul_result;
        __result.overflow = _mul_full_overflow_u8(__lhs, __rhs, &__mul_result);
        __result.value    = static_cast<uint8_t>(__mul_result);
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC, >=, 19, 37) ||
        // !_CCCL_ARCH(X86_64) vvv
        __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC, >=, 19, 37) || !_CCCL_ARCH(X86_64) ^^^
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        _CUDA_VSTD::uint16_t __hi;
        __result.overflow = _mul_full_overflow_u16(__lhs, __rhs, &__result.value, &__hi);
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC, >=, 19, 37) ||
        // !_CCCL_ARCH(X86_64) vvv
        return __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC, >=, 19, 37) || !_CCCL_ARCH(X86_64) ^^^
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        _CUDA_VSTD::uint32_t __hi;
        __result.overflow = _mul_full_overflow_u32(__lhs, __rhs, &__result.value, &__hi);
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        const uint64_t __mul_result = __emulu(__lhs, __rhs);
        __result.value              = static_cast<uint32_t>(__mul_result);
        __result.overflow           = __mul_result > _CUDA_VSTD::numeric_limits<uint32_t>::max();
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
        return __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        _CUDA_VSTD::uint64_t __hi;
        __result.overflow = _mul_full_overflow_u64(__lhs, __rhs, &__result.value, &__hi);
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        __result.value    = __lhs * __rhs;
        __result.overflow = __umulh(__lhs, __rhs) != _Tp{0};
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
        __result = __impl_generic(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
      }
      else
      {
        __result = __impl_generic(__lhs, __rhs);
      }
    }

    return __result;
  }
#endif // !_CCCL_COMPILER(NVRTC)
#if _CCCL_HAS_CUDA_COMPILER
  template <class _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static overflow_result<_Tp> __impl_device(_Tp __lhs, _Tp __rhs) noexcept
  {
    overflow_result<_Tp> __result{};

    if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        _CUDA_VSTD::int16_t __mul_result;
        asm("mul.lo.s16 %0, %1, %2;"
            : "=h"(__mul_result)
            : "h"(static_cast<int16_t>(__lhs)), "h"(static_cast<int16_t>(__rhs)));
        __result.value    = static_cast<int8_t>(__mul_result);
        __result.overflow = (__mul_result < static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::min()))
                         || (__mul_result > static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::max()));
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        _CUDA_VSTD::int32_t __mul_result;
        asm("mul.wide.s16 %0, %1, %2;" : "=r"(__mul_result) : "h"(__lhs), "h"(__rhs));
        __result.value    = static_cast<int16_t>(__mul_result);
        __result.overflow = (__mul_result < static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::min()))
                         || (__mul_result > static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::max()));
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        _CUDA_VSTD::int64_t __mul_result;
        asm("mul.wide.s32 %0, %1, %2;" : "=l"(__mul_result) : "r"(__lhs), "r"(__rhs));
        __result.value    = static_cast<int32_t>(__mul_result);
        __result.overflow = (__mul_result < static_cast<int64_t>(_CUDA_VSTD::numeric_limits<int32_t>::min()))
                         || (__mul_result > static_cast<int64_t>(_CUDA_VSTD::numeric_limits<int32_t>::max()));
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.value                 = __lhs * __rhs;
        const _CUDA_VSTD::int64_t __hi = __mul64hi(__lhs, __rhs);
        __result.overflow              = __hi != int64_t{0} && __hi != int64_t{-1};
      }
      else
      {
        return __impl_generic(__lhs, __rhs);
      }
    }
    else
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        _CUDA_VSTD::uint16_t __mul_result;
        asm("mul.lo.u16 %0, %1, %2;"
            : "=h"(__mul_result)
            : "h"(static_cast<uint16_t>(__lhs)), "h"(static_cast<uint16_t>(__rhs)));
        __result.value    = static_cast<uint8_t>(__mul_result);
        __result.overflow = __mul_result > static_cast<uint16_t>(_CUDA_VSTD::numeric_limits<uint8_t>::max());
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        _CUDA_VSTD::uint32_t __mul_result;
        asm("mul.wide.u16 %0, %1, %2;" : "=r"(__mul_result) : "h"(__lhs), "h"(__rhs));
        __result.value    = static_cast<uint16_t>(__mul_result);
        __result.overflow = __mul_result > static_cast<uint32_t>(_CUDA_VSTD::numeric_limits<uint16_t>::max());
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        _CUDA_VSTD::uint64_t __mul_result;
        asm("mul.wide.u32 %0, %1, %2;" : "=l"(__mul_result) : "r"(__lhs), "r"(__rhs));
        __result.value    = static_cast<uint32_t>(__mul_result);
        __result.overflow = __mul_result > static_cast<uint64_t>(_CUDA_VSTD::numeric_limits<uint32_t>::max());
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.value    = __lhs * __rhs;
        __result.overflow = __umul64hi(__lhs, __rhs) != uint64_t{0};
      }
      else
      {
        return __impl_generic(__lhs, __rhs);
      }
    }

    return __result;
  }
#endif // _CCCL_HAS_CUDA_COMPILER
};

//! @brief Multiplies two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr overflow_result<_CUDA_VSTD::common_type_t<_Lhs, _Rhs>>
mul_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  using _Common = _CUDA_VSTD::common_type_t<_Lhs, _Rhs>;

  const auto __l = static_cast<_Common>(__lhs);
  const auto __r = static_cast<_Common>(__rhs);

  if constexpr (sizeof(_Common) >= 2 * sizeof(_Lhs) && sizeof(_Common) >= 2 * sizeof(_Rhs))
  {
    return {static_cast<_Common>(__l * __r), false};
  }
  else
  {
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
    return __mul_overflow::__impl_builtin(__l, __r);
#else // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_MUL_OVERFLOW vvv
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST, (return __mul_overflow::__impl_host(__l, __r);))
    }
    return __mul_overflow::__impl_generic(__l, __r);
#endif // !_CCCL_BUILTIN_MUL_OVERFLOW
  }
}

//! @brief Multiplies two numbers \p __lhs and \p __rhs with overflow detection
//! @param __lhs The left-hand side number
//! @param __rhs The right-hand side number
//! @param __result The result of the multiplication
_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
mul_overflow(const _Lhs __lhs, const _Rhs __rhs, _CUDA_VSTD::common_type_t<_Lhs, _Rhs>& __result) noexcept
{
  const auto __res = ::cuda::mul_overflow(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___NUMERIC_MUL_OVERFLOW_H
