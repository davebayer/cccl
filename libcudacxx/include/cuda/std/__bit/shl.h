//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_SHL_H
#define _CUDA_STD___BIT_SHL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/uabs.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
#  include <cuda/__ptx/instructions/shl.h>
#  include <cuda/__ptx/instructions/shr.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_INT128()
[[nodiscard]] _CCCL_API constexpr __uint128_t
__shl_combine_words(const uint32_t __w0, const uint32_t __w1, const uint32_t __w2, const uint32_t __w3) noexcept
{
  return (__uint128_t{__w3} << 96) | (__uint128_t{__w2} << 64) | (__uint128_t{__w1} << 32) | __uint128_t{__w0};
}
#endif // _CCCL_HAS_INT128()

_CCCL_TEMPLATE(class _Tp, class _Shift)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Shift>)
[[nodiscard]] _CCCL_API constexpr _Tp shl(const _Tp __v, const _Shift __shift) noexcept
{
  constexpr auto __width = uint32_t{__num_bits_v<_Tp>};
  const auto __ushift    = ::cuda::uabs(__shift);

  if constexpr (is_signed_v<_Shift>)
  {
    if (__shift < 0)
    {
#if !_CCCL_TILE_COMPILATION()
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
        // On device, shr PTX instruction clamps the shift to width, however only 32-bit shifts are supported.
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if constexpr (sizeof(_Shift) <= sizeof(uint32_t) && sizeof(_Tp) <= sizeof(int64_t))
                       {
                         using _Up = __make_nbit_int_t<sizeof(_Tp) < sizeof(int64_t) ? 32 : 64, is_signed_v<_Tp>>;
                         return static_cast<_Tp>(::cuda::ptx::shr(_Up{__v}, static_cast<uint32_t>(__ushift)));
                       }
#  if _CCCL_HAS_INT128()
                       else if constexpr (sizeof(_Tp) == sizeof(__int128_t))
                       {
                         const auto __cnt  = static_cast<uint32_t>(__ushift);
                         const auto __uv   = static_cast<__uint128_t>(__v);
                         const auto __fill = (is_signed_v<_Tp> && __v < _Tp{0}) ? ~uint32_t{0} : uint32_t{0};
                         const auto __w0   = static_cast<uint32_t>(__uv);
                         const auto __w1   = static_cast<uint32_t>(__uv >> 32);
                         const auto __w2   = static_cast<uint32_t>(__uv >> 64);
                         const auto __w3   = static_cast<uint32_t>(__uv >> 96);

                         const auto __res_0 = ::__funnelshift_r(__w0, __w1, __cnt);
                         const auto __res_1 = ::__funnelshift_r(__w1, __w2, __cnt);
                         const auto __res_2 = ::__funnelshift_r(__w2, __w3, __cnt);
                         const auto __res_3 = ::__funnelshift_r(__w3, __fill, __cnt);

                         const auto __word_shift1 = (__cnt & __word_bits) != 0;
                         const auto __word_shift2 = (__cnt & (2 * __word_bits)) != 0;
                         const auto __fill_result = __ushift >= __width;

                         const auto __tmp_0 = __word_shift1 ? __res_1 : __res_0;
                         const auto __tmp_1 = __word_shift1 ? __res_2 : __res_1;
                         const auto __tmp_2 = __word_shift1 ? __res_3 : __res_2;
                         const auto __tmp_3 = __word_shift1 ? __fill : __res_3;

                         const auto __out_0 = __fill_result ? __fill : (__word_shift2 ? __tmp_2 : __tmp_0);
                         const auto __out_1 = __fill_result ? __fill : (__word_shift2 ? __tmp_3 : __tmp_1);
                         const auto __out_2 = (__fill_result || __word_shift2) ? __fill : __tmp_2;
                         const auto __out_3 = (__fill_result || __word_shift2) ? __fill : __tmp_3;
                         return static_cast<_Tp>(::cuda::std::__shl_combine_words(__out_0, __out_1, __out_2, __out_3));
                       }
#  endif // _CCCL_HAS_INT128()
                     }))
      }
#endif // !_CCCL_TILE_COMPILATION()
      return (__ushift < __width) ? (__v >> __ushift) : static_cast<_Tp>(::cuda::std::cmp_less(__v, 0) ? -1 : 0);
    }
  }

#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    // On device, shl PTX instruction clamps the shift to width, however only 32-bit shifts are supported.
    NV_IF_TARGET(NV_IS_DEVICE, ({
                   if constexpr (sizeof(_Shift) <= sizeof(uint32_t) && sizeof(_Tp) <= sizeof(int64_t))
                   {
                     using _Up = __make_nbit_int_t<sizeof(_Tp) < sizeof(int64_t) ? 32 : 64, is_signed_v<_Tp>>;
                     return static_cast<_Tp>(::cuda::ptx::shl(_Up{__v}, static_cast<uint32_t>(__ushift)));
                   }
#  if _CCCL_HAS_INT128()
                   else if constexpr (is_same_v<_Tp, __uint128_t> || is_same_v<_Tp, __int128_t>)
                   {
                     constexpr auto __word_bits = uint32_t{32};
                     const auto __cnt           = static_cast<uint32_t>(__ushift);
                     const auto __uv            = ::cuda::std::__to_unsigned_like(__v);
                     const auto __w0            = static_cast<uint32_t>(__uv);
                     const auto __w1            = static_cast<uint32_t>(__uv >> 32);
                     const auto __w2            = static_cast<uint32_t>(__uv >> 64);
                     const auto __w3            = static_cast<uint32_t>(__uv >> 96);

                     const auto __res_0 = ::__funnelshift_l(uint32_t{0}, __w0, __cnt);
                     const auto __res_1 = ::__funnelshift_l(__w0, __w1, __cnt);
                     const auto __res_2 = ::__funnelshift_l(__w1, __w2, __cnt);
                     const auto __res_3 = ::__funnelshift_l(__w2, __w3, __cnt);

                     const auto __word_shift1 = (__cnt & __word_bits) != 0;
                     const auto __word_shift2 = (__cnt & (2 * __word_bits)) != 0;
                     const auto __zero        = __ushift >= __width;

                     const auto __tmp_0 = __word_shift1 ? uint32_t{0} : __res_0;
                     const auto __tmp_1 = __word_shift1 ? __res_0 : __res_1;
                     const auto __tmp_2 = __word_shift1 ? __res_1 : __res_2;
                     const auto __tmp_3 = __word_shift1 ? __res_2 : __res_3;

                     const auto __out_0 = (__zero || __word_shift2) ? uint32_t{0} : __tmp_0;
                     const auto __out_1 = (__zero || __word_shift2) ? uint32_t{0} : __tmp_1;
                     const auto __out_2 = __zero ? uint32_t{0} : (__word_shift2 ? __tmp_0 : __tmp_2);
                     const auto __out_3 = __zero ? uint32_t{0} : (__word_shift2 ? __tmp_1 : __tmp_3);
                     return static_cast<_Tp>(::cuda::std::__shl_combine_words(__out_0, __out_1, __out_2, __out_3));
                   }
#  endif // _CCCL_HAS_INT128()
                 }))
  }
#endif // !_CCCL_TILE_COMPILATION()
  return (__ushift < __width) ? static_cast<_Tp>(::cuda::std::__to_unsigned_like(__v) << __ushift) : _Tp{0};
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_SHL_H
