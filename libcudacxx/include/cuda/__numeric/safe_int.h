//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_SAFE_INT_H
#define _CUDA___NUMERIC_SAFE_INT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__utility/cmp.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <_CUDA_VSTD::size_t _NBits, bool _IsSigned>
class __cccl_safe_int
{
  using __int_type = _CUDA_VSTD::__make_nbit_int_t<_NBits, _IsSigned>;

  __int_type __value_;

  static constexpr __int_type __min_value = _CUDA_VSTD::numeric_limits<__int_type>::min() + _IsSigned;
  static constexpr __int_type __max_value = _CUDA_VSTD::numeric_limits<__int_type>::max() - !_IsSigned;
  static constexpr __int_type __nan_value =
    (_IsSigned) ? _CUDA_VSTD::numeric_limits<__int_type>::min() : _CUDA_VSTD::numeric_limits<__int_type>::max();

public:
  _CCCL_HIDE_FROM_ABI constexpr __cccl_safe_int() noexcept = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp)) //_CCCL_AND _CUDA_VSTD::in_range<__cccl_safe_int>(
                                                            // _CUDA_VSTD::numeric_limits<_Tp>::min() _CCCL_AND
                                                            // _CUDA_VSTD::numeric_limits<_Tp>::max()))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __cccl_safe_int(const _Tp& __v) noexcept
      : __value_((_CUDA_VSTD::in_range<__cccl_safe_int>(__v)) ? static_cast<__int_type>(__v) : __nan_value)
  {}

  // _CCCL_TEMPLATE(class _Tp)
  // _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND !(_CUDA_VSTD::in_range<__cccl_safe_int>(
  //   _CUDA_VSTD::numeric_limits<_Tp>::min() || _CUDA_VSTD::numeric_limits<_Tp>::max())))
  // _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __cccl_safe_int(const _Tp& __v) noexcept
  //     : __value_((_CUDA_VSTD::in_range<__cccl_safe_int>(__v)) ? static_cast<__int_type>(__v) : __nan_value)
  // {}

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_floating_point, _Tp))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __cccl_safe_int(const _Tp& __v) noexcept
      : __cccl_safe_int((_CUDA_VSTD::isnan(__v) || __v < _CUDA_VSTD::numeric_limits<__int_type>::min()
                         || __v > _CUDA_VSTD::numeric_limits<__int_type>::max())
                          ? __nan_value
                          : static_cast<__int_type>(__v))
  {}

  _CCCL_HIDE_FROM_ABI constexpr __cccl_safe_int(const __cccl_safe_int&) noexcept = default;

  template <_CUDA_VSTD::size_t _ONBits, bool _OIsSigned>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __cccl_safe_int(const __cccl_safe_int<_ONBits, _OIsSigned>& __other) noexcept
      : __value_((__other.__is_nan() || !_CUDA_VSTD::in_range<__cccl_safe_int>(__other))
                   ? static_cast<__int_type>(__other.__value_)
                   : __nan_value)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __cccl_safe_int& operator=(const __cccl_safe_int&) noexcept = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_floating_point, _Tp) _CCCL_AND numeric_limits<_Tp>::has_quiet_NaN)
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator _Tp() noexcept
  {
    return _CUDA_VSTD::isnan(*this) ? _CUDA_VSTD::numeric_limits<_Tp>::quiet_NaN() : static_cast<_Tp>(__value_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit operator bool() const noexcept
  {
    return !_CUDA_VSTD::isnan(*this) && value != 0;
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __is_nan() const noexcept
  {
    return __value_ == __nan_value;
  }
};

template <_CUDA_VSTD::size_t _NBits>
using safe_int     = __cccl_safe_int<_NBits, true>;
using safe_int8_t  = safe_int<8>;
using safe_int16_t = safe_int<16>;
using safe_int32_t = safe_int<32>;
using safe_int64_t = safe_int<64>;
#if _CCCL_HAS_INT128()
using safe_int128_t = safe_int<128>;
#endif // _CCCL_HAS_INT128()

template <_CUDA_VSTD::size_t _NBits>
using safe_uint     = __cccl_safe_int<_NBits, false>;
using safe_uint8_t  = safe_uint<8>;
using safe_uint16_t = safe_uint<16>;
using safe_uint32_t = safe_uint<32>;
using safe_uint64_t = safe_uint<64>;
#if _CCCL_HAS_INT128()
using safe_uint128_t = safe_uint<128>;
#endif // _CCCL_HAS_INT128()

namespace safe_int_literals
{
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_int8_t operator"" _si8(unsigned long long __v) noexcept
{
  return safe_int8_t(__v);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_int16_t operator"" _si16(unsigned long long __v) noexcept
{
  return safe_int16_t(__v);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_int32_t operator"" _si32(unsigned long long __v) noexcept
{
  return safe_int32_t(__v);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_int64_t operator"" _si64(unsigned long long __v) noexcept
{
  return safe_int64_t(__v);
}
#if _CCCL_HAS_INT128()
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_int128_t operator"" _si128(unsigned long long __v) noexcept
{
  return safe_int128_t(__v);
}
#endif // _CCCL_HAS_INT128()
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_uint8_t operator"" _su8(unsigned long long __v) noexcept
{
  return safe_uint8_t(__v);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_uint16_t operator"" _su16(unsigned long long __v) noexcept
{
  return safe_uint16_t(__v);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_uint32_t operator"" _su32(unsigned long long __v) noexcept
{
  return safe_uint32_t(__v);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_uint64_t operator"" _su64(unsigned long long __v) noexcept
{
  return safe_uint64_t(__v);
}
#if _CCCL_HAS_INT128()
_LIBCUDACXX_HIDE_FROM_ABI constexpr safe_uint128_t operator"" _su128(unsigned long long __v) noexcept
{
  return safe_uint128_t(__v);
}
#endif // _CCCL_HAS_INT128()
} // namespace safe_int_literals

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___NUMERIC_SAFE_INT_H
