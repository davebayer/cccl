//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/charconv>
#include <cuda/std/cstdint>
#include <cuda/std/initializer_list>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC constexpr void test_value(T value, int offset = 1)
{
  using U       = cuda::std::make_unsigned_t<T>;
  bool negative = value < T{0};
  U magnitude   = negative ? U{0} - static_cast<U>(value) : static_cast<U>(value);

  char expected[sizeof(T) * 8 + 1]{};
  int length = 0;
  do
  {
    expected[length++] = static_cast<char>('0' + magnitude % 2);
    magnitude /= 2;
  } while (magnitude != 0);
  if (negative)
  {
    expected[length++] = '-';
  }

  for (int capacity : {0, 1, length / 2, length - 1, length, length + 1})
  {
    alignas(cuda::std::uint32_t) char buffer[sizeof(T) * 8 + 8]{};
    char* first     = buffer + offset;
    first[-1]       = '#';
    first[capacity] = '#';
    auto result     = cuda::std::to_chars(first, first + capacity, value, 2);
    assert(first[-1] == '#');
    assert(first[capacity] == '#');
    int written = capacity < length ? capacity : length;
    for (int i = 0; i < written; ++i)
    {
      assert(first[i] == expected[length - i - 1]);
    }
    if (capacity < length)
    {
      assert(result.ec == cuda::std::errc::value_too_large);
      assert(result.ptr == first + capacity);
    }
    else
    {
      assert(result.ec == cuda::std::errc{});
      assert(result.ptr == first + length);
      if (capacity > length)
      {
        assert(first[length] == '\0');
      }
    }
  }
}

template <class T>
TEST_FUNC constexpr bool test_type()
{
  using U = cuda::std::make_unsigned_t<T>;
  test_value(cuda::std::numeric_limits<T>::min());
  test_value(cuda::std::numeric_limits<T>::max());
  if constexpr (sizeof(T) <= 4)
  {
    // Cover every output width of the unrolled conversion, including chunk boundaries.
    for (int bit = 0; bit < cuda::std::numeric_limits<T>::digits; ++bit)
    {
      auto power = static_cast<T>(U{1} << bit);
      test_value(power);
      test_value(static_cast<T>(power - 1));
      if constexpr (cuda::std::is_signed_v<T>)
      {
        test_value(static_cast<T>(-power));
      }
    }
  }
  for (int nibble = 0; nibble < 16; ++nibble)
  {
    test_value(static_cast<T>(nibble));
    // Repeat every nibble throughout the value, clearing the sign bit for signed types.
    auto pattern = static_cast<T>(
      ((cuda::std::numeric_limits<U>::max() / 15) * nibble) & static_cast<U>(cuda::std::numeric_limits<T>::max()));
    test_value(pattern);
    if constexpr (cuda::std::is_signed_v<T>)
    {
      test_value(static_cast<T>(-nibble));
      test_value(static_cast<T>(-pattern));
    }
  }
  return true;
}

TEST_FUNC constexpr bool test_nibbles(int prefix)
{
  for (int byte = 0; byte < 256; ++byte)
  {
    test_value(static_cast<cuda::std::uint16_t>((prefix << 8) | byte));
  }
  return true;
}

TEST_FUNC constexpr bool test_alignment()
{
  for (int offset = 1; offset <= 4; ++offset)
  {
    test_value(cuda::std::uint32_t{0xFFFFFFFF}, offset);
    test_value(cuda::std::uint32_t{0x12345678}, offset);
    test_value(cuda::std::int32_t{-0x1234567}, offset);
    test_value(cuda::std::numeric_limits<cuda::std::int64_t>::min(), offset);
  }
  return true;
}

int main(int, char**)
{
  test_alignment();
  static_assert(test_alignment());

  // Exercise every pair of nibbles with zero to three leading bits left over.
  // Keep the constant evaluations separate to stay within compiler step limits.
  for (int prefix : {0, 1, 2, 4})
  {
    test_nibbles(prefix);
  }
  static_assert(test_nibbles(0));
  static_assert(test_nibbles(1));
  static_assert(test_nibbles(2));
  static_assert(test_nibbles(4));

  test_type<cuda::std::int8_t>();
  test_type<cuda::std::int16_t>();
  test_type<cuda::std::int32_t>();
  test_type<cuda::std::int64_t>();
  test_type<cuda::std::uint8_t>();
  test_type<cuda::std::uint16_t>();
  test_type<cuda::std::uint32_t>();
  test_type<cuda::std::uint64_t>();
  static_assert(test_type<cuda::std::int8_t>());
  static_assert(test_type<cuda::std::int16_t>());
  static_assert(test_type<cuda::std::int32_t>());
  static_assert(test_type<cuda::std::int64_t>());
  static_assert(test_type<cuda::std::uint8_t>());
  static_assert(test_type<cuda::std::uint16_t>());
  static_assert(test_type<cuda::std::uint32_t>());
  static_assert(test_type<cuda::std::uint64_t>());
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
  static_assert(test_type<__int128_t>());
  static_assert(test_type<__uint128_t>());
#endif // _CCCL_HAS_INT128()

  return 0;
}
