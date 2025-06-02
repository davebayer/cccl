//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template <class T>
//   constexpr int countl_one(T x) noexcept;

// The number of consecutive 1 bits, starting from the most significant bit.
//   [ Note: Returns N if x == cuda::std::numeric_limits<T>::max(). ]
//
// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

enum EnumMask : int
{
};
enum class EnumClassMask : unsigned char
{
};

class ClassMask
{
  unsigned value;

public:
  using value_type = unsigned;
  constexpr ClassMask(value_type v) noexcept
      : value(v)
  {}
  constexpr explicit operator unsigned() const noexcept
  {
    return value;
  }
};

template <>
inline constexpr bool cuda::__is_bitmask_v<EnumMask> = true;
template <>
inline constexpr bool cuda::__is_bitmask_v<EnumClassMask> = true;
template <>
inline constexpr bool cuda::__is_bitmask_v<ClassMask> = true;

template <class T>
__host__ __device__ constexpr void test_countl_one(int zero_value)
{
  using VT = cuda::__bitmask_value_type_t<T>;

  static_assert(cuda::std::is_same_v<int, decltype(cuda::countl_one(cuda::std::declval<T>()))>);
  static_assert(noexcept(cuda::countl_one(cuda::std::declval<T>())));

  constexpr auto dig = cuda::std::numeric_limits<VT>::digits;

  assert(cuda::countl_one(T(VT(~121 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~122 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~123 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~124 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~125 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~126 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~127 + zero_value))) == dig - 7);
  assert(cuda::countl_one(T(VT(~128 + zero_value))) == dig - 8);
  assert(cuda::countl_one(T(VT(~129 + zero_value))) == dig - 8);
  assert(cuda::countl_one(T(VT(~130 + zero_value))) == dig - 8);
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_countl_one<EnumMask>(zero_value);
  test_countl_one<EnumClassMask>(zero_value);
  test_countl_one<ClassMask>(zero_value);
  return true;
}

int main(int, char**)
{
  volatile int zero_value = 0;
  test(zero_value);
  static_assert(test(0));
  return 0;
}
