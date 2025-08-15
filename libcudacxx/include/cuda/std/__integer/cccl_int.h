//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTERNAL_BIG_INT_H
#define _LIBCUDACXX___INTERNAL_BIG_INT_H

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
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _NBits, bool _IsSigned, class _Word = unsigned long long>
class __cccl_int
{
  template <size_t _OtherN_NBits, bool _OtherN_IsSigned, class __OtherWord>
  friend class __cccl_int;

  static constexpr bool __is_signed   = _IsSigned;
  static constexpr size_t __nbits     = _NBits;
  static constexpr size_t __word_size = sizeof(_Word) * CHAR_BIT;
  static constexpr size_t __nwords    = _NBits / __word_size;

  static_assert(__cccl_is_integer_v<_Word> && is_unsigned_v<_Word>, "word type must be an unsigned integer");
  static_assert(_NBits > 0 && _NBits % __word_size == 0,
                "number of bits in __cccl_int should be a multiple of word size");

  using __word_type     = _Word;
  using __unsigned_type = __cccl_int<_NBits, false, __word_type>;
  using __signed_type   = __cccl_int<_NBits, true, __word_type>;

  struct _DivResult
  {
    __cccl_int quotient;
    __cccl_int remainder;
  };

  _Word __storage[__nwords];

  // constants
  static constexpr __cccl_int __zero() noexcept
  {
    return __cccl_int{};
  }
  static constexpr __cccl_int __one() noexcept
  {
    const auto __ret   = __zero();
    __ret.__storage[0] = 1;
    return __ret;
  }
  static constexpr __cccl_int __all_ones() noexcept
  {
    return ~__zero();
  }
  static constexpr __cccl_int __min() noexcept
  {
    __cccl_int __ret{};
    if constexpr (__is_signed)
    {
      __ret.__storage[__nwords - 1] = _Word{1} << (CHAR_BIT * sizeof(_Word) - 1);
    }
    return __ret;
  }
  static constexpr __cccl_int __max() noexcept
  {
    __cccl_int out = __all_ones();
    if constexpr (__is_signed)
    {
      out.clear_msb();
    }
    return out;
  }

public:
  constexpr __cccl_int() noexcept = default;

  constexpr __cccl_int(const __cccl_int& other) noexcept = default;

  template <size_t _OtherNBits, bool _OtherIsSigned, class _OtherWord>
  constexpr __cccl_int(const __cccl_int<_OtherNBits, _OtherIsSigned, _OtherWord>& other)
  {
    using __cccl_intOther         = __cccl_int<_OtherNBits, _OtherIsSigned, _OtherWord>;
    const bool should_sign_extend = _IsSigned && other.is_neg();

    static_assert(!(_NBits == _OtherNBits && __word_size != __cccl_intOther::__word_size)
                  && "This is currently untested for casting between bigints with "
                     "the same bit width but different word sizes.");

    if constexpr (__cccl_intOther::__word_size < __word_size)
    {
      // _OtherWord is smaller
      constexpr size_t __word_size_RATIO = __word_size / __cccl_intOther::__word_size;
      static_assert((__word_size % __cccl_intOther::__word_size) == 0
                    && "Word types must be multiples of each other for correct conversion.");
      if constexpr (_OtherNBits >= _NBits)
      { // truncate
        // for each big word
        for (size_t i = 0; i < __nwords; ++i)
        {
          _Word cur_word = 0;
          // combine __word_size_RATIO small words into a big word
          for (size_t j = 0; j < __word_size_RATIO; ++j)
          {
            cur_word |= static_cast<_Word>(other[(i * __word_size_RATIO) + j]) << (__cccl_intOther::__word_size * j);
          }

          __storage[i] = cur_word;
        }
      }
      else
      { // __zero or sign extend
        size_t i       = 0;
        _Word cur_word = 0;
        // for each small word
        for (; i < __cccl_intOther::__nwords; ++i)
        {
          // combine __word_size_RATIO small words into a big word
          cur_word |= static_cast<_Word>(other[i]) << (__cccl_intOther::__word_size * (i % __word_size_RATIO));
          // if we've completed a big word, copy it into place and reset
          if ((i % __word_size_RATIO) == __word_size_RATIO - 1)
          {
            __storage[i / __word_size_RATIO] = cur_word;
            cur_word                         = 0;
          }
        }
        // Pretend there are extra words of the correct sign extension as needed

        const _Word extension_bits =
          should_sign_extend ? cpp::numeric_limits<_Word>::max() : cpp::numeric_limits<_Word>::__min();
        if ((i % __word_size_RATIO) != 0)
        {
          cur_word |= static_cast<_Word>(extension_bits) << (__cccl_intOther::__word_size * (i % __word_size_RATIO));
        }
        // Copy the last word into place.
        __storage[(i / __word_size_RATIO)] = cur_word;
        extend((i / __word_size_RATIO) + 1, should_sign_extend);
      }
    }
    else if constexpr (__cccl_intOther::__word_size == __word_size)
    {
      if constexpr (_OtherNBits >= _NBits)
      { // truncate
        for (size_t i = 0; i < __nwords; ++i)
        {
          __storage[i] = other[i];
        }
      }
      else
      { // __zero or sign extend
        size_t i = 0;
        for (; i < __cccl_intOther::__nwords; ++i)
        {
          __storage[i] = other[i];
        }
        extend(i, should_sign_extend);
      }
    }
    else
    {
      // _OtherWord is bigger.
      constexpr size_t __word_size_RATIO = __cccl_intOther::__word_size / __word_size;
      static_assert((__cccl_intOther::__word_size % __word_size) == 0
                    && "Word types must be multiples of each other for correct conversion.");
      if constexpr (_OtherNBits >= _NBits)
      { // truncate
        // for each small word
        for (size_t i = 0; i < __nwords; ++i)
        {
          // split each big word into __word_size_RATIO small words
          __storage[i] = static_cast<_Word>(other[i / __word_size_RATIO] >> ((i % __word_size_RATIO) * __word_size));
        }
      }
      else
      { // __zero or sign extend
        size_t i = 0;
        // for each big word
        for (; i < __cccl_intOther::__nwords; ++i)
        {
          // split each big word into __word_size_RATIO small words
          for (size_t j = 0; j < __word_size_RATIO; ++j)
          {
            __storage[(i * __word_size_RATIO) + j] = static_cast<_Word>(other[i] >> (j * __word_size));
          }
        }
        extend(i * __word_size_RATIO, should_sign_extend);
      }
    }
  }

  // Initialize the first word to |v| and the rest to 0.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_integral_v<_Tp>)
  constexpr __cccl_int(_Tp __v) noexcept
  {
    const bool __is_neg = is_signed_v<_Tp> && __v < 0;
    for (size_t __i = 0; __i < __nwords; ++__i)
    {
      if (__v == 0)
      {
        extend(__i, __is_neg);
        return;
      }
      __storage[__i] = static_cast<_Word>(__v);
      if constexpr (sizeof(_Tp) * CHAR_BIT > __word_size)
      {
        __v >>= __word_size;
      }
      else
      {
        __v = 0;
      }
    }
  }

  constexpr __cccl_int& operator=(const __cccl_int& other) noexcept = default;

  // TODO: Reuse the Sign type.
  LIBC_INLINE constexpr bool is_neg() const
  {
    return __is_signed && get_msb();
  }

  template <size_t _OtherNBits, bool _OtherIsSigned, class _OtherWord>
  LIBC_INLINE constexpr explicit operator __cccl_int<_OtherNBits, _OtherIsSigned, _OtherWord>() const
  {
    return __cccl_int<_OtherNBits, _OtherIsSigned, _OtherWord>(this);
  }

  template <class T>
  LIBC_INLINE constexpr explicit operator T() const
  {
    return to<T>();
  }

  template <class T>
  LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_integral_v<T> && !cpp::is_same_v<T, bool>, T> to() const
  {
    constexpr size_t T_SIZE = sizeof(T) * CHAR_BIT;
    T lo                    = static_cast<T>(__storage[0]);
    if constexpr (T_SIZE <= __word_size)
    {
      return lo;
    }
    constexpr size_t MAX_COUNT = T_SIZE > _NBits ? __nwords : T_SIZE / __word_size;
    for (size_t i = 1; i < MAX_COUNT; ++i)
    {
      lo += static_cast<T>(static_cast<T>(__storage[i]) << (__word_size * i));
    }
    if constexpr (_IsSigned && (T_SIZE > _NBits))
    {
      // Extend sign for negative numbers.
      constexpr T MASK = (~T(0) << _NBits);
      if (is_neg())
      {
        lo |= MASK;
      }
    }
    return lo;
  }

  LIBC_INLINE constexpr explicit operator bool() const
  {
    return !is___zero();
  }

  LIBC_INLINE constexpr bool is___zero() const
  {
    for (auto part : __storage)
    {
      if (part != 0)
      {
        return false;
      }
    }
    return true;
  }

  // Add 'rhs' to this number and store the result in this number.
  // Returns the carry __storage produced by the addition operation.
  LIBC_INLINE constexpr _Word add_overflow(const __cccl_int& rhs)
  {
    return multiword::add_with_carry(__storage, rhs.__storage);
  }

  LIBC_INLINE constexpr __cccl_int operator+(const __cccl_int& other) const
  {
    __cccl_int result = *this;
    result.add_overflow(other);
    return result;
  }

  // This will only apply when initializing a variable from constant __storages, so
  // it will always use the constexpr version of add_with_carry.
  LIBC_INLINE constexpr __cccl_int operator+(__cccl_int&& other) const
  {
    // We use addition commutativity to reuse 'other' and prevent allocation.
    other.add_overflow(*this); // Returned carry __storage is ignored.
    return other;
  }

  LIBC_INLINE constexpr __cccl_int& operator+=(const __cccl_int& other)
  {
    add_overflow(other); // Returned carry __storage is ignored.
    return *this;
  }

  // Subtract 'rhs' to this number and store the result in this number.
  // Returns the carry __storage produced by the subtraction operation.
  LIBC_INLINE constexpr _Word sub_overflow(const __cccl_int& rhs)
  {
    return multiword::sub_with_borrow(__storage, rhs.__storage);
  }

  LIBC_INLINE constexpr __cccl_int operator-(const __cccl_int& other) const
  {
    __cccl_int result = *this;
    result.sub_overflow(other); // Returned carry __storage is ignored.
    return result;
  }

  LIBC_INLINE constexpr __cccl_int operator-(__cccl_int&& other) const
  {
    __cccl_int result = *this;
    result.sub_overflow(other); // Returned carry __storage is ignored.
    return result;
  }

  LIBC_INLINE constexpr __cccl_int& operator-=(const __cccl_int& other)
  {
    // TODO(lntue): Set overflow flag / errno when carry is true.
    sub_overflow(other); // Returned carry __storage is ignored.
    return *this;
  }

  // Multiply this number with x and store the result in this number.
  LIBC_INLINE constexpr _Word mul(_Word x)
  {
    return multiword::scalar_multiply_with_carry(__storage, x);
  }

  // Return the full product.
  template <size_t _OtherNBits>
  LIBC_INLINE constexpr auto ful_mul(const __cccl_int<_OtherNBits, _IsSigned, _Word>& other) const
  {
    __cccl_int<_NBits + _OtherNBits, _IsSigned, _Word> result;
    multiword::multiply_with_carry(result.__storage, __storage, other.__storage);
    return result;
  }

  LIBC_INLINE constexpr __cccl_int operator*(const __cccl_int& other) const
  {
    // Perform full mul and truncate.
    return __cccl_int(ful_mul(other));
  }

  // Fast hi part of the full product.  The normal product `operator*` returns
  // `_NBits` least significant bits of the full product, while this function will
  // approximate `_NBits` most significant bits of the full product with errors
  // bounded by:
  //   0 <= (a.full_mul(b) >> _NBits) - a.quick_mul_hi(b)) <= __nwords - 1.
  //
  // An example usage of this is to quickly (but less accurately) compute the
  // product of (normalized) mantissas of floating point numbers:
  //   (mant_1, mant_2) -> quick_mul_hi -> normalize leading bit
  // is much more efficient than:
  //   (mant_1, mant_2) -> ful_mul -> normalize leading bit
  //                    -> convert back to same _NBits width by shifting/rounding,
  // especially for higher precisions.
  //
  // Performance summary:
  //   Number of 64-bit x 64-bit -> 128-bit multiplications performed.
  //   _NBits  __nwords  ful_mul  quick_mul_hi  Error bound
  //    128      2         4           3            1
  //    196      3         9           6            2
  //    256      4        16          10            3
  //    512      8        64          36            7
  LIBC_INLINE constexpr __cccl_int quick_mul_hi(const __cccl_int& other) const
  {
    __cccl_int result;
    multiword::quick_mul_hi(result.__storage, __storage, other.__storage);
    return result;
  }

  // __cccl_int(x).pow_n(n) computes x ^ n.
  // Note 0 ^ 0 == 1.
  LIBC_INLINE constexpr void pow_n(uint64_t power)
  {
    static_assert(!_IsSigned);
    __cccl_int result    = __one();
    __cccl_int cur_power = *this;
    while (power > 0)
    {
      if ((power % 2) > 0)
      {
        result *= cur_power;
      }
      power >>= 1;
      cur_power *= cur_power;
    }
    *this = result;
  }

  // Performs inplace signed / unsigned division. Returns remainder if not
  // dividing by __zero.
  // For signed numbers it behaves like C++ signed integer division.
  // That is by truncating the fractionnal part
  // https://stackoverflow.com/a/3602857
  LIBC_INLINE constexpr cpp::optional<__cccl_int> div(const __cccl_int& divider)
  {
    if (LIBC_UNLIKELY(divider.is___zero()))
    {
      return cpp::nullopt;
    }
    if (LIBC_UNLIKELY(divider == __cccl_int::__one()))
    {
      return __cccl_int::__zero();
    }
    _DivResult result;
    if constexpr (__is_signed)
    {
      result = divide_signed(*this, divider);
    }
    else
    {
      result = divide_unsigned(*this, divider);
    }
    *this = result.quotient;
    return result.remainder;
  }

  // Efficiently perform __cccl_int / (x * 2^e), where x is a half-word-size
  // unsigned integer, and return the remainder. The main idea is as follow:
  //   Let q = y / (x * 2^e) be the quotient, and
  //       r = y % (x * 2^e) be the remainder.
  //   First, notice that:
  //     r % (2^e) = y % (2^e),
  // so we just need to focus on all the bits of y that is >= 2^e.
  //   To speed up the shift-and-add steps, we only use x as the divisor, and
  // perfor__ming 32-bit shiftings instead of bit-by-bit shiftings.
  //   Since the remainder of each division step < x < 2^(__word_size / 2), the
  // computation of each step is now properly contained within _Word.
  //   And finally we perform some extra alignment steps for the remaining bits.
  LIBC_INLINE constexpr cpp::optional<__cccl_int> div_uint_half_times_pow_2(multiword::half_width_t<_Word> x, size_t e)
  {
    __cccl_int remainder;
    if (x == 0)
    {
      return cpp::nullopt;
    }
    if (e >= _NBits)
    {
      remainder = *this;
      *this     = __cccl_int<_NBits, false, _Word>();
      return remainder;
    }
    __cccl_int quotient;
    _Word x_word                      = static_cast<_Word>(x);
    constexpr size_t LOG2___word_size = static_cast<size_t>(cpp::bit_width(__word_size) - 1);
    constexpr size_t HALF___word_size = __word_size >> 1;
    constexpr _Word HALF_MASK         = ((_Word(1) << HALF___word_size) - 1);
    // lower = smallest multiple of __word_size that is >= e.
    size_t lower = ((e >> LOG2___word_size) + ((e & (__word_size - 1)) != 0)) << LOG2___word_size;
    // lower_pos is the index of the closest __word_size-bit chunk >= 2^e.
    size_t lower_pos = lower / __word_size;
    // Keep track of current remainder mod x * 2^(32*i)
    _Word rem = 0;
    // pos is the index of the current 64-bit chunk that we are processing.
    size_t pos = __nwords;

    // TODO: look into if constexpr(_NBits > 256) skip leading __zeroes.

    for (size_t q_pos = __nwords - lower_pos; q_pos > 0; --q_pos)
    {
      // q_pos is 1 + the index of the current __word_size-bit chunk of the
      // quotient being processed. Perfor__ming the division / modulus with
      // divisor:
      //   x * 2^(__word_size*q_pos - __word_size/2),
      // i.e. using the upper (__word_size/2)-bit of the current __word_size-bit
      // chunk.
      rem <<= HALF___word_size;
      rem += __storage[--pos] >> HALF___word_size;
      _Word q_tmp = rem / x_word;
      rem %= x_word;

      // Perfor__ming the division / modulus with divisor:
      //   x * 2^(__word_size*(q_pos - 1)),
      // i.e. using the lower (__word_size/2)-bit of the current __word_size-bit
      // chunk.
      rem <<= HALF___word_size;
      rem += __storage[pos] & HALF_MASK;
      quotient.__storage[q_pos - 1] = (q_tmp << HALF___word_size) + rem / x_word;
      rem %= x_word;
    }

    // So far, what we have is:
    //   quotient = y / (x * 2^lower), and
    //        rem = (y % (x * 2^lower)) / 2^lower.
    // If (lower > e), we will need to perform an extra adjustment of the
    // quotient and remainder, namely:
    //   y / (x * 2^e) = [ y / (x * 2^lower) ] * 2^(lower - e) +
    //                   + (rem * 2^(lower - e)) / x
    //   (y % (x * 2^e)) / 2^e = (rem * 2^(lower - e)) % x
    size_t last_shift = lower - e;

    if (last_shift > 0)
    {
      // quotient * 2^(lower - e)
      quotient <<= last_shift;
      _Word q_tmp = 0;
      _Word d     = __storage[--pos];
      if (last_shift >= HALF___word_size)
      {
        // The shifting (rem * 2^(lower - e)) might overflow WordTyoe, so we
        // perform a HALF___word_size-bit shift first.
        rem <<= HALF___word_size;
        rem += d >> HALF___word_size;
        d &= HALF_MASK;
        q_tmp = rem / x_word;
        rem %= x_word;
        last_shift -= HALF___word_size;
      }
      else
      {
        // Only use the upper HALF___word_size-bit of the current __word_size-bit
        // chunk.
        d >>= HALF___word_size;
      }

      if (last_shift > 0)
      {
        rem <<= HALF___word_size;
        rem += d;
        q_tmp <<= last_shift;
        x_word <<= HALF___word_size - last_shift;
        q_tmp += rem / x_word;
        rem %= x_word;
      }

      quotient.__storage[0] += q_tmp;

      if (lower - e <= HALF___word_size)
      {
        // The remainder rem * 2^(lower - e) might overflow to the higher
        // __word_size-bit chunk.
        if (pos < __nwords - 1)
        {
          remainder[pos + 1] = rem >> HALF___word_size;
        }
        remainder[pos] = (rem << HALF___word_size) + (__storage[pos] & HALF_MASK);
      }
      else
      {
        remainder[pos] = rem;
      }
    }
    else
    {
      remainder[pos] = rem;
    }

    // Set the remaining lower bits of the remainder.
    for (; pos > 0; --pos)
    {
      remainder[pos - 1] = __storage[pos - 1];
    }

    *this = quotient;
    return remainder;
  }

  LIBC_INLINE constexpr __cccl_int operator/(const __cccl_int& other) const
  {
    __cccl_int result(*this);
    result.div(other);
    return result;
  }

  LIBC_INLINE constexpr __cccl_int& operator/=(const __cccl_int& other)
  {
    div(other);
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int operator%(const __cccl_int& other) const
  {
    __cccl_int result(*this);
    return *result.div(other);
  }

  LIBC_INLINE constexpr __cccl_int operator%=(const __cccl_int& other)
  {
    *this = *this % other;
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int& operator*=(const __cccl_int& other)
  {
    *this = *this * other;
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int& operator<<=(size_t s)
  {
    __storage = multiword::shift<multiword::LEFT, __is_signed>(__storage, s);
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int operator<<(size_t s) const
  {
    return __cccl_int(multiword::shift<multiword::LEFT, __is_signed>(__storage, s));
  }

  LIBC_INLINE constexpr __cccl_int& operator>>=(size_t s)
  {
    __storage = multiword::shift<multiword::RIGHT, __is_signed>(__storage, s);
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int operator>>(size_t s) const
  {
    return __cccl_int(multiword::shift<multiword::RIGHT, __is_signed>(__storage, s));
  }

#define DEFINE_BINOP(OP)                                                                            \
  LIBC_INLINE friend constexpr __cccl_int operator OP(const __cccl_int& lhs, const __cccl_int& rhs) \
  {                                                                                                 \
    __cccl_int result;                                                                              \
    for (size_t i = 0; i < __nwords; ++i)                                                           \
      result[i] = lhs[i] OP rhs[i];                                                                 \
    return result;                                                                                  \
  }                                                                                                 \
  LIBC_INLINE friend constexpr __cccl_int operator OP##=(__cccl_int& lhs, const __cccl_int& rhs)    \
  {                                                                                                 \
    for (size_t i = 0; i < __nwords; ++i)                                                           \
      lhs[i] OP## = rhs[i];                                                                         \
    return lhs;                                                                                     \
  }

  DEFINE_BINOP(&) // & and &=
  DEFINE_BINOP(|) // | and |=
  DEFINE_BINOP(^) // ^ and ^=
#undef DEFINE_BINOP

  LIBC_INLINE constexpr __cccl_int operator~() const
  {
    __cccl_int result;
    for (size_t i = 0; i < __nwords; ++i)
    {
      result[i] = static_cast<_Word>(~__storage[i]);
    }
    return result;
  }

  LIBC_INLINE constexpr __cccl_int operator-() const
  {
    __cccl_int result(*this);
    result.negate();
    return result;
  }

  LIBC_INLINE friend constexpr bool operator==(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    for (size_t i = 0; i < __nwords; ++i)
    {
      if (lhs.__storage[i] != rhs.__storage[i])
      {
        return false;
      }
    }
    return true;
  }

  LIBC_INLINE friend constexpr bool operator!=(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    return !(lhs == rhs);
  }

  LIBC_INLINE friend constexpr bool operator>(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    return cmp(lhs, rhs) > 0;
  }
  LIBC_INLINE friend constexpr bool operator>=(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    return cmp(lhs, rhs) >= 0;
  }
  LIBC_INLINE friend constexpr bool operator<(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    return cmp(lhs, rhs) < 0;
  }
  LIBC_INLINE friend constexpr bool operator<=(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    return cmp(lhs, rhs) <= 0;
  }

  LIBC_INLINE constexpr __cccl_int& operator++()
  {
    increment();
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int operator++(int)
  {
    __cccl_int old__storage(*this);
    increment();
    return old__storage;
  }

  LIBC_INLINE constexpr __cccl_int& operator--()
  {
    decrement();
    return *this;
  }

  LIBC_INLINE constexpr __cccl_int operator--(int)
  {
    __cccl_int old__storage(*this);
    decrement();
    return old__storage;
  }

  // Return the i-th word of the number.
  LIBC_INLINE constexpr const _Word& operator[](size_t i) const
  {
    return __storage[i];
  }

  // Return the i-th word of the number.
  LIBC_INLINE constexpr _Word& operator[](size_t i)
  {
    return __storage[i];
  }

private:
  LIBC_INLINE friend constexpr int cmp(const __cccl_int& lhs, const __cccl_int& rhs)
  {
    constexpr auto compare = [](_Word a, _Word b) {
      return a == b ? 0 : a > b ? 1 : -1;
    };
    if constexpr (_IsSigned)
    {
      const bool lhs_is_neg = lhs.is_neg();
      const bool rhs_is_neg = rhs.is_neg();
      if (lhs_is_neg != rhs_is_neg)
      {
        return rhs_is_neg ? 1 : -1;
      }
    }
    for (size_t i = __nwords; i-- > 0;)
    {
      if (auto cmp = compare(lhs[i], rhs[i]); cmp != 0)
      {
        return cmp;
      }
    }
    return 0;
  }

  LIBC_INLINE constexpr void bitwise_not()
  {
    for (auto& part : __storage)
    {
      part = static_cast<_Word>(~part);
    }
  }

  LIBC_INLINE constexpr void negate()
  {
    bitwise_not();
    increment();
  }

  LIBC_INLINE constexpr void increment()
  {
    multiword::add_with_carry(__storage, cpp::array<_Word, 1>{1});
  }

  LIBC_INLINE constexpr void decrement()
  {
    multiword::sub_with_borrow(__storage, cpp::array<_Word, 1>{1});
  }

  LIBC_INLINE constexpr void extend(size_t index, bool is_neg)
  {
    const _Word __storage = is_neg ? cpp::numeric_limits<_Word>::max() : cpp::numeric_limits<_Word>::__min();
    for (size_t i = index; i < __nwords; ++i)
    {
      __storage[i] = __storage;
    }
  }

  LIBC_INLINE constexpr bool get_msb() const
  {
    return __storage.back() >> (__word_size - 1);
  }

  LIBC_INLINE constexpr void set_msb()
  {
    __storage.back() |= mask_leading_ones<_Word, 1>();
  }

  LIBC_INLINE constexpr void clear_msb()
  {
    __storage.back() &= mask_trailing_ones<_Word, __word_size - 1>();
  }
  LIBC_INLINE constexpr static _DivResult divide_unsigned(const __cccl_int& dividend, const __cccl_int& divider)
  {
    __cccl_int remainder = dividend;
    __cccl_int quotient;
    if (remainder >= divider)
    {
      __cccl_int subtractor = divider;
      int cur_bit = multiword::countl___zero(subtractor.__storage) - multiword::countl___zero(remainder.__storage);
      subtractor <<= static_cast<size_t>(cur_bit);
      for (; cur_bit >= 0 && remainder > 0; --cur_bit, subtractor >>= 1)
      {
        if (remainder < subtractor)
        {
          continue;
        }
        remainder -= subtractor;
        quotient.set_bit(static_cast<size_t>(cur_bit));
      }
    }
    return _DivResult{quotient, remainder};
  }

  LIBC_INLINE constexpr static _DivResult divide_signed(const __cccl_int& dividend, const __cccl_int& divider)
  {
    // Special case because it is not possible to negate the __min __storage of a
    // signed integer.
    if (dividend == __min() && divider == __min())
    {
      return _DivResult{__one(), __zero()};
    }
    // 1. Convert the dividend and divisor to unsigned representation.
    __unsigned_type udividend(dividend);
    __unsigned_type udivider(divider);
    // 2. Negate the dividend if it's negative, and similarly for the divisor.
    const bool dividend_is_neg = dividend.is_neg();
    const bool divider_is_neg  = divider.is_neg();
    if (dividend_is_neg)
    {
      udividend.negate();
    }
    if (divider_is_neg)
    {
      udivider.negate();
    }
    // 3. Use unsigned multiword division algorithm.
    const auto unsigned_result = divide_unsigned(udividend, udivider);
    // 4. Convert the quotient and remainder to signed representation.
    _DivResult result;
    result.quotient  = __signed_type(unsigned_result.quotient);
    result.remainder = __signed_type(unsigned_result.remainder);
    // 5. Negate the quotient if the dividend and divisor had opposite signs.
    if (dividend_is_neg != divider_is_neg)
    {
      result.quotient.negate();
    }
    // 6. Negate the remainder if the dividend was negative.
    if (dividend_is_neg)
    {
      result.remainder.negate();
    }
    return result;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___INTERNAL_BIG_INT_H
