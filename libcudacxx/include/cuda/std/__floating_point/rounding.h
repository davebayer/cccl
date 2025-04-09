//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_ROUNDING_H
#define _LIBCUDACXX___FLOATING_POINT_ROUNDING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_integer.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __fp_round_kind
{
  __to_nearest,
  __toward_zero,
  __toward_neg_inf,
  __toward_pos_inf,
};

template <size_t _ToNBits, size_t _FromNBits, class _FromMant>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _FromMant __fp_get_rounding_bits(const _FromMant& __v) noexcept
{
  static_assert(_FromNBits > _ToNBits);

  return static_cast<_FromMant>(__v & ((_FromMant(1) << (_FromNBits - _ToNBits)) - 1));
}

template <size_t _ToNBits, size_t _FromNBits, class _FromMant>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr unsigned
__fp_round_make_correction_to_nearest(const _FromMant& __from, bool __sign) noexcept
{
  if constexpr (_FromNBits > _ToNBits + 1)
  {
    constexpr auto __guard_round_bits_shift = _FromNBits - _ToNBits - 1;
    constexpr auto __guard_round_bits_mask  = (_ToNBits > 0) ? 1u : 3u;

    const auto __guard_round_bits = static_cast<unsigned>(__from >> __guard_round_bits_shift) & __guard_round_bits_mask;

    switch (__guard_round_bits)
    {
      case 0u:
      case 2u:
        return 0;
      case 1u:
        // check if any of the sticky bits are set
        return _CUDA_VSTD::countr_one(__from) < __guard_round_bits_shift;
      case 3u:
        return 1;
      default:
        _CCCL_UNREACHABLE();
    }
  }
  else
  {
    return 0;
  }
}

template <__fp_round_kind _RndKind, size_t _ToNBits, size_t _FromNBits, class _FromMant>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr unsigned
__fp_round_make_correction([[maybe_unused]] const _FromMant& __from, [[maybe_unused]] bool __sign) noexcept
{
  if constexpr (_RndKind == __fp_round_kind::__to_nearest)
  {
    return _CUDA_VSTD::__fp_round_make_correction_to_nearest<_ToNBits, _FromNBits>(__from, __sign);
  }
  else if constexpr (_RndKind == __fp_round_kind::__toward_zero)
  {
    return 0u;
  }
  else if constexpr (_RndKind == __fp_round_kind::__toward_neg_inf)
  {
    return _CUDA_VSTD::__fp_get_rounding_bits<_ToNBits, _FromNBits>(__from) != 0 && __sign;
  }
  else if constexpr (_RndKind == __fp_round_kind::__toward_pos_inf)
  {
    return _CUDA_VSTD::__fp_get_rounding_bits<_ToNBits, _FromNBits>(__from) != 0 && !__sign;
  }
  else
  {
    static_assert(__always_false_v<decltype(_RndKind)>, "Invalid rounding kind");
  }
}

template <__fp_round_kind _RndKind, size_t _ToNBits, class _ToMant, size_t _FromNBits, class _FromMant>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ::cuda::overflow_result<_ToMant>
__fp_round(const _FromMant& __from, bool __sign) noexcept
{
  ::cuda::overflow_result<_ToMant> __result{};

  // compute the result value
  if constexpr (_FromNBits != 0 && _ToNBits != 0)
  {
    if constexpr (_FromNBits > _ToNBits)
    {
      __result.value = static_cast<_ToMant>(__from >> (_FromNBits - _ToNBits));
    }
    else
    {
      __result.value = static_cast<_ToMant>(__from) << (_ToNBits - _FromNBits);
    }
  }

  // if we have bits to round, do the rounding
  if constexpr (_FromNBits > _ToNBits)
  {
    constexpr auto __to_mant_max = static_cast<_ToMant>((_ToMant(1) << _ToNBits) - 1);

    const auto __rnd_corr = _CUDA_VSTD::__fp_round_make_correction<_RndKind, _ToNBits, _FromNBits>(__from, __sign);

    if (__rnd_corr != 0)
    {
      if constexpr (_ToNBits > 0)
      {
        if (__result.value >= __to_mant_max)
        {
          __result.value    = _ToMant(0);
          __result.overflow = true;
        }
        else
        {
          __result.value += __rnd_corr;
        }
      }
      else
      {
        __result.value    = _ToMant(0);
        __result.overflow = true;
      }
    }
  }

  return __result;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FLOATING_POINT_ROUNDING_H
