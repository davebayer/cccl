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

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _NBits, bool _IsSigned, class _Word = unsigned long long>
class _BigInt
{
  template <size_t _OtherN_NBits, bool _OtherN_IsSigned, class __OtherWord>
  friend class _BigInt;

  static constexpr bool __is_signed   = _IsSigned;
  static constexpr size_t __nbits     = _NBits;
  static constexpr size_t __word_size = sizeof(_Word) * CHAR_BIT;
  static constexpr size_t __nwords    = _NBits / __word_size;

  static_assert(__cccl_is_integer_v<_Word> && is_unsigned_v<_Word>, "word type must be an unsigned integer");
  static_assert(_NBits > 0 && _NBits % __word_size == 0, "number of bits in _BigInt should be a multiple of word size");

  using __word_type   = _Word;
  using unsigned_type = _BigInt<_NBits, false, __word_type>;
  using signed_type   = _BigInt<_NBits, true, __word_type>;

  struct Division
  {
    _BigInt quotient;
    _BigInt remainder;
  };

  _Word val[__nwords];

public:
  constexpr _BigInt() noexcept = default;

  constexpr _BigInt(const _BigInt& other) noexcept = default;

  template <size_t _OtherNBits, bool _OtherIsSigned, class _OtherWord>
  constexpr _BigInt(const _BigInt<_OtherNBits, _OtherIsSigned, _OtherWord>& other)
  {
    using _BigIntOther            = _BigInt<_OtherNBits, _OtherIsSigned, _OtherWord>;
    const bool should_sign_extend = _IsSigned && other.is_neg();

    static_assert(!(_NBits == _OtherNBits && __word_size != _BigIntOther::__word_size)
                  && "This is currently untested for casting between bigints with "
                     "the same bit width but different word sizes.");

    if constexpr (_BigIntOther::__word_size < __word_size)
    {
      // _OtherWord is smaller
      constexpr size_t __word_size_RATIO = __word_size / _BigIntOther::__word_size;
      static_assert((__word_size % _BigIntOther::__word_size) == 0
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
            cur_word |= static_cast<_Word>(other[(i * __word_size_RATIO) + j]) << (_BigIntOther::__word_size * j);
          }

          val[i] = cur_word;
        }
      }
      else
      { // zero or sign extend
        size_t i       = 0;
        _Word cur_word = 0;
        // for each small word
        for (; i < _BigIntOther::__nwords; ++i)
        {
          // combine __word_size_RATIO small words into a big word
          cur_word |= static_cast<_Word>(other[i]) << (_BigIntOther::__word_size * (i % __word_size_RATIO));
          // if we've completed a big word, copy it into place and reset
          if ((i % __word_size_RATIO) == __word_size_RATIO - 1)
          {
            val[i / __word_size_RATIO] = cur_word;
            cur_word                   = 0;
          }
        }
        // Pretend there are extra words of the correct sign extension as needed

        const _Word extension_bits =
          should_sign_extend ? cpp::numeric_limits<_Word>::max() : cpp::numeric_limits<_Word>::min();
        if ((i % __word_size_RATIO) != 0)
        {
          cur_word |= static_cast<_Word>(extension_bits) << (_BigIntOther::__word_size * (i % __word_size_RATIO));
        }
        // Copy the last word into place.
        val[(i / __word_size_RATIO)] = cur_word;
        extend((i / __word_size_RATIO) + 1, should_sign_extend);
      }
    }
    else if constexpr (_BigIntOther::__word_size == __word_size)
    {
      if constexpr (_OtherNBits >= _NBits)
      { // truncate
        for (size_t i = 0; i < __nwords; ++i)
        {
          val[i] = other[i];
        }
      }
      else
      { // zero or sign extend
        size_t i = 0;
        for (; i < _BigIntOther::__nwords; ++i)
        {
          val[i] = other[i];
        }
        extend(i, should_sign_extend);
      }
    }
    else
    {
      // _OtherWord is bigger.
      constexpr size_t __word_size_RATIO = _BigIntOther::__word_size / __word_size;
      static_assert((_BigIntOther::__word_size % __word_size) == 0
                    && "Word types must be multiples of each other for correct conversion.");
      if constexpr (_OtherNBits >= _NBits)
      { // truncate
        // for each small word
        for (size_t i = 0; i < __nwords; ++i)
        {
          // split each big word into __word_size_RATIO small words
          val[i] = static_cast<_Word>(other[i / __word_size_RATIO] >> ((i % __word_size_RATIO) * __word_size));
        }
      }
      else
      { // zero or sign extend
        size_t i = 0;
        // for each big word
        for (; i < _BigIntOther::__nwords; ++i)
        {
          // split each big word into __word_size_RATIO small words
          for (size_t j = 0; j < __word_size_RATIO; ++j)
          {
            val[(i * __word_size_RATIO) + j] = static_cast<_Word>(other[i] >> (j * __word_size));
          }
        }
        extend(i * __word_size_RATIO, should_sign_extend);
      }
    }
  }

  // Construct a _BigInt from a C array.
  template <size_t N>
  LIBC_INLINE constexpr _BigInt(const _Word (&nums)[N])
  {
    static_assert(N == __nwords);
    for (size_t i = 0; i < __nwords; ++i)
    {
      val[i] = nums[i];
    }
  }

  LIBC_INLINE constexpr explicit _BigInt(const cpp::array<_Word, __nwords>& words)
  {
    val = words;
  }

  // Initialize the first word to |v| and the rest to 0.
  template <class T, class = cpp::enable_if_t<cpp::is_integral_v<T>>>
  LIBC_INLINE constexpr _BigInt(T v)
  {
    constexpr size_t T_SIZE = sizeof(T) * CHAR_BIT;
    const bool is_neg       = v < 0;
    for (size_t i = 0; i < __nwords; ++i)
    {
      if (v == 0)
      {
        extend(i, is_neg);
        return;
      }
      val[i] = static_cast<_Word>(v);
      if constexpr (T_SIZE > __word_size)
      {
        v >>= __word_size;
      }
      else
      {
        v = 0;
      }
    }
  }
  LIBC_INLINE constexpr _BigInt& operator=(const _BigInt& other) = default;

  // constants
  LIBC_INLINE static constexpr _BigInt zero()
  {
    return _BigInt();
  }
  LIBC_INLINE static constexpr _BigInt one()
  {
    return _BigInt(1);
  }
  LIBC_INLINE static constexpr _BigInt all_ones()
  {
    return ~zero();
  }
  LIBC_INLINE static constexpr _BigInt min()
  {
    _BigInt out;
    if constexpr (__is_signed)
    {
      out.set_msb();
    }
    return out;
  }
  LIBC_INLINE static constexpr _BigInt max()
  {
    _BigInt out = all_ones();
    if constexpr (__is_signed)
    {
      out.clear_msb();
    }
    return out;
  }

  // TODO: Reuse the Sign type.
  LIBC_INLINE constexpr bool is_neg() const
  {
    return __is_signed && get_msb();
  }

  template <size_t _OtherNBits, bool _OtherIsSigned, class _OtherWord>
  LIBC_INLINE constexpr explicit operator _BigInt<_OtherNBits, _OtherIsSigned, _OtherWord>() const
  {
    return _BigInt<_OtherNBits, _OtherIsSigned, _OtherWord>(this);
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
    T lo                    = static_cast<T>(val[0]);
    if constexpr (T_SIZE <= __word_size)
    {
      return lo;
    }
    constexpr size_t MAX_COUNT = T_SIZE > _NBits ? __nwords : T_SIZE / __word_size;
    for (size_t i = 1; i < MAX_COUNT; ++i)
    {
      lo += static_cast<T>(static_cast<T>(val[i]) << (__word_size * i));
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
    return !is_zero();
  }

  LIBC_INLINE constexpr bool is_zero() const
  {
    for (auto part : val)
    {
      if (part != 0)
      {
        return false;
      }
    }
    return true;
  }

  // Add 'rhs' to this number and store the result in this number.
  // Returns the carry value produced by the addition operation.
  LIBC_INLINE constexpr _Word add_overflow(const _BigInt& rhs)
  {
    return multiword::add_with_carry(val, rhs.val);
  }

  LIBC_INLINE constexpr _BigInt operator+(const _BigInt& other) const
  {
    _BigInt result = *this;
    result.add_overflow(other);
    return result;
  }

  // This will only apply when initializing a variable from constant values, so
  // it will always use the constexpr version of add_with_carry.
  LIBC_INLINE constexpr _BigInt operator+(_BigInt&& other) const
  {
    // We use addition commutativity to reuse 'other' and prevent allocation.
    other.add_overflow(*this); // Returned carry value is ignored.
    return other;
  }

  LIBC_INLINE constexpr _BigInt& operator+=(const _BigInt& other)
  {
    add_overflow(other); // Returned carry value is ignored.
    return *this;
  }

  // Subtract 'rhs' to this number and store the result in this number.
  // Returns the carry value produced by the subtraction operation.
  LIBC_INLINE constexpr _Word sub_overflow(const _BigInt& rhs)
  {
    return multiword::sub_with_borrow(val, rhs.val);
  }

  LIBC_INLINE constexpr _BigInt operator-(const _BigInt& other) const
  {
    _BigInt result = *this;
    result.sub_overflow(other); // Returned carry value is ignored.
    return result;
  }

  LIBC_INLINE constexpr _BigInt operator-(_BigInt&& other) const
  {
    _BigInt result = *this;
    result.sub_overflow(other); // Returned carry value is ignored.
    return result;
  }

  LIBC_INLINE constexpr _BigInt& operator-=(const _BigInt& other)
  {
    // TODO(lntue): Set overflow flag / errno when carry is true.
    sub_overflow(other); // Returned carry value is ignored.
    return *this;
  }

  // Multiply this number with x and store the result in this number.
  LIBC_INLINE constexpr _Word mul(_Word x)
  {
    return multiword::scalar_multiply_with_carry(val, x);
  }

  // Return the full product.
  template <size_t _OtherNBits>
  LIBC_INLINE constexpr auto ful_mul(const _BigInt<_OtherNBits, _IsSigned, _Word>& other) const
  {
    _BigInt<_NBits + _OtherNBits, _IsSigned, _Word> result;
    multiword::multiply_with_carry(result.val, val, other.val);
    return result;
  }

  LIBC_INLINE constexpr _BigInt operator*(const _BigInt& other) const
  {
    // Perform full mul and truncate.
    return _BigInt(ful_mul(other));
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
  LIBC_INLINE constexpr _BigInt quick_mul_hi(const _BigInt& other) const
  {
    _BigInt result;
    multiword::quick_mul_hi(result.val, val, other.val);
    return result;
  }

  // _BigInt(x).pow_n(n) computes x ^ n.
  // Note 0 ^ 0 == 1.
  LIBC_INLINE constexpr void pow_n(uint64_t power)
  {
    static_assert(!_IsSigned);
    _BigInt result    = one();
    _BigInt cur_power = *this;
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
  // dividing by zero.
  // For signed numbers it behaves like C++ signed integer division.
  // That is by truncating the fractionnal part
  // https://stackoverflow.com/a/3602857
  LIBC_INLINE constexpr cpp::optional<_BigInt> div(const _BigInt& divider)
  {
    if (LIBC_UNLIKELY(divider.is_zero()))
    {
      return cpp::nullopt;
    }
    if (LIBC_UNLIKELY(divider == _BigInt::one()))
    {
      return _BigInt::zero();
    }
    Division result;
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

  // Efficiently perform _BigInt / (x * 2^e), where x is a half-word-size
  // unsigned integer, and return the remainder. The main idea is as follow:
  //   Let q = y / (x * 2^e) be the quotient, and
  //       r = y % (x * 2^e) be the remainder.
  //   First, notice that:
  //     r % (2^e) = y % (2^e),
  // so we just need to focus on all the bits of y that is >= 2^e.
  //   To speed up the shift-and-add steps, we only use x as the divisor, and
  // performing 32-bit shiftings instead of bit-by-bit shiftings.
  //   Since the remainder of each division step < x < 2^(__word_size / 2), the
  // computation of each step is now properly contained within _Word.
  //   And finally we perform some extra alignment steps for the remaining bits.
  LIBC_INLINE constexpr cpp::optional<_BigInt> div_uint_half_times_pow_2(multiword::half_width_t<_Word> x, size_t e)
  {
    _BigInt remainder;
    if (x == 0)
    {
      return cpp::nullopt;
    }
    if (e >= _NBits)
    {
      remainder = *this;
      *this     = _BigInt<_NBits, false, _Word>();
      return remainder;
    }
    _BigInt quotient;
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

    // TODO: look into if constexpr(_NBits > 256) skip leading zeroes.

    for (size_t q_pos = __nwords - lower_pos; q_pos > 0; --q_pos)
    {
      // q_pos is 1 + the index of the current __word_size-bit chunk of the
      // quotient being processed. Performing the division / modulus with
      // divisor:
      //   x * 2^(__word_size*q_pos - __word_size/2),
      // i.e. using the upper (__word_size/2)-bit of the current __word_size-bit
      // chunk.
      rem <<= HALF___word_size;
      rem += val[--pos] >> HALF___word_size;
      _Word q_tmp = rem / x_word;
      rem %= x_word;

      // Performing the division / modulus with divisor:
      //   x * 2^(__word_size*(q_pos - 1)),
      // i.e. using the lower (__word_size/2)-bit of the current __word_size-bit
      // chunk.
      rem <<= HALF___word_size;
      rem += val[pos] & HALF_MASK;
      quotient.val[q_pos - 1] = (q_tmp << HALF___word_size) + rem / x_word;
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
      _Word d     = val[--pos];
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

      quotient.val[0] += q_tmp;

      if (lower - e <= HALF___word_size)
      {
        // The remainder rem * 2^(lower - e) might overflow to the higher
        // __word_size-bit chunk.
        if (pos < __nwords - 1)
        {
          remainder[pos + 1] = rem >> HALF___word_size;
        }
        remainder[pos] = (rem << HALF___word_size) + (val[pos] & HALF_MASK);
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
      remainder[pos - 1] = val[pos - 1];
    }

    *this = quotient;
    return remainder;
  }

  LIBC_INLINE constexpr _BigInt operator/(const _BigInt& other) const
  {
    _BigInt result(*this);
    result.div(other);
    return result;
  }

  LIBC_INLINE constexpr _BigInt& operator/=(const _BigInt& other)
  {
    div(other);
    return *this;
  }

  LIBC_INLINE constexpr _BigInt operator%(const _BigInt& other) const
  {
    _BigInt result(*this);
    return *result.div(other);
  }

  LIBC_INLINE constexpr _BigInt operator%=(const _BigInt& other)
  {
    *this = *this % other;
    return *this;
  }

  LIBC_INLINE constexpr _BigInt& operator*=(const _BigInt& other)
  {
    *this = *this * other;
    return *this;
  }

  LIBC_INLINE constexpr _BigInt& operator<<=(size_t s)
  {
    val = multiword::shift<multiword::LEFT, __is_signed>(val, s);
    return *this;
  }

  LIBC_INLINE constexpr _BigInt operator<<(size_t s) const
  {
    return _BigInt(multiword::shift<multiword::LEFT, __is_signed>(val, s));
  }

  LIBC_INLINE constexpr _BigInt& operator>>=(size_t s)
  {
    val = multiword::shift<multiword::RIGHT, __is_signed>(val, s);
    return *this;
  }

  LIBC_INLINE constexpr _BigInt operator>>(size_t s) const
  {
    return _BigInt(multiword::shift<multiword::RIGHT, __is_signed>(val, s));
  }

#define DEFINE_BINOP(OP)                                                                   \
  LIBC_INLINE friend constexpr _BigInt operator OP(const _BigInt& lhs, const _BigInt& rhs) \
  {                                                                                        \
    _BigInt result;                                                                        \
    for (size_t i = 0; i < __nwords; ++i)                                                  \
      result[i] = lhs[i] OP rhs[i];                                                        \
    return result;                                                                         \
  }                                                                                        \
  LIBC_INLINE friend constexpr _BigInt operator OP##=(_BigInt& lhs, const _BigInt& rhs)    \
  {                                                                                        \
    for (size_t i = 0; i < __nwords; ++i)                                                  \
      lhs[i] OP## = rhs[i];                                                                \
    return lhs;                                                                            \
  }

  DEFINE_BINOP(&) // & and &=
  DEFINE_BINOP(|) // | and |=
  DEFINE_BINOP(^) // ^ and ^=
#undef DEFINE_BINOP

  LIBC_INLINE constexpr _BigInt operator~() const
  {
    _BigInt result;
    for (size_t i = 0; i < __nwords; ++i)
    {
      result[i] = static_cast<_Word>(~val[i]);
    }
    return result;
  }

  LIBC_INLINE constexpr _BigInt operator-() const
  {
    _BigInt result(*this);
    result.negate();
    return result;
  }

  LIBC_INLINE friend constexpr bool operator==(const _BigInt& lhs, const _BigInt& rhs)
  {
    for (size_t i = 0; i < __nwords; ++i)
    {
      if (lhs.val[i] != rhs.val[i])
      {
        return false;
      }
    }
    return true;
  }

  LIBC_INLINE friend constexpr bool operator!=(const _BigInt& lhs, const _BigInt& rhs)
  {
    return !(lhs == rhs);
  }

  LIBC_INLINE friend constexpr bool operator>(const _BigInt& lhs, const _BigInt& rhs)
  {
    return cmp(lhs, rhs) > 0;
  }
  LIBC_INLINE friend constexpr bool operator>=(const _BigInt& lhs, const _BigInt& rhs)
  {
    return cmp(lhs, rhs) >= 0;
  }
  LIBC_INLINE friend constexpr bool operator<(const _BigInt& lhs, const _BigInt& rhs)
  {
    return cmp(lhs, rhs) < 0;
  }
  LIBC_INLINE friend constexpr bool operator<=(const _BigInt& lhs, const _BigInt& rhs)
  {
    return cmp(lhs, rhs) <= 0;
  }

  LIBC_INLINE constexpr _BigInt& operator++()
  {
    increment();
    return *this;
  }

  LIBC_INLINE constexpr _BigInt operator++(int)
  {
    _BigInt oldval(*this);
    increment();
    return oldval;
  }

  LIBC_INLINE constexpr _BigInt& operator--()
  {
    decrement();
    return *this;
  }

  LIBC_INLINE constexpr _BigInt operator--(int)
  {
    _BigInt oldval(*this);
    decrement();
    return oldval;
  }

  // Return the i-th word of the number.
  LIBC_INLINE constexpr const _Word& operator[](size_t i) const
  {
    return val[i];
  }

  // Return the i-th word of the number.
  LIBC_INLINE constexpr _Word& operator[](size_t i)
  {
    return val[i];
  }

  // Return the i-th bit of the number.
  LIBC_INLINE constexpr bool get_bit(size_t i) const
  {
    const size_t word_index = i / __word_size;
    return 1 & (val[word_index] >> (i % __word_size));
  }

  // Set the i-th bit of the number.
  LIBC_INLINE constexpr void set_bit(size_t i)
  {
    const size_t word_index = i / __word_size;
    val[word_index] |= _Word(1) << (i % __word_size);
  }

private:
  LIBC_INLINE friend constexpr int cmp(const _BigInt& lhs, const _BigInt& rhs)
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
    for (auto& part : val)
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
    multiword::add_with_carry(val, cpp::array<_Word, 1>{1});
  }

  LIBC_INLINE constexpr void decrement()
  {
    multiword::sub_with_borrow(val, cpp::array<_Word, 1>{1});
  }

  LIBC_INLINE constexpr void extend(size_t index, bool is_neg)
  {
    const _Word value = is_neg ? cpp::numeric_limits<_Word>::max() : cpp::numeric_limits<_Word>::min();
    for (size_t i = index; i < __nwords; ++i)
    {
      val[i] = value;
    }
  }

  LIBC_INLINE constexpr bool get_msb() const
  {
    return val.back() >> (__word_size - 1);
  }

  LIBC_INLINE constexpr void set_msb()
  {
    val.back() |= mask_leading_ones<_Word, 1>();
  }

  LIBC_INLINE constexpr void clear_msb()
  {
    val.back() &= mask_trailing_ones<_Word, __word_size - 1>();
  }
  LIBC_INLINE constexpr static Division divide_unsigned(const _BigInt& dividend, const _BigInt& divider)
  {
    _BigInt remainder = dividend;
    _BigInt quotient;
    if (remainder >= divider)
    {
      _BigInt subtractor = divider;
      int cur_bit        = multiword::countl_zero(subtractor.val) - multiword::countl_zero(remainder.val);
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
    return Division{quotient, remainder};
  }

  LIBC_INLINE constexpr static Division divide_signed(const _BigInt& dividend, const _BigInt& divider)
  {
    // Special case because it is not possible to negate the min value of a
    // signed integer.
    if (dividend == min() && divider == min())
    {
      return Division{one(), zero()};
    }
    // 1. Convert the dividend and divisor to unsigned representation.
    unsigned_type udividend(dividend);
    unsigned_type udivider(divider);
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
    Division result;
    result.quotient  = signed_type(unsigned_result.quotient);
    result.remainder = signed_type(unsigned_result.remainder);
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

  friend signed_type;
  friend unsigned_type;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___INTERNAL_BIG_INT_H
