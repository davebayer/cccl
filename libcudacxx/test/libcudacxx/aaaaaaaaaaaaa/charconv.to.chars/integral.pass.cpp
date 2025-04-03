//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12712420
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=50000000

#include <cuda/std/__charconv_>
#include <cuda/std/cstdint>

#include "charconv_test_helpers.h"
#include "test_macros.h"

[[nodiscard]] __host__ __device__ constexpr bool str_equal(const char* str1, const char* str2, size_t max_len)
{
  for (size_t i = 0; i < max_len; ++i)
  {
    if (str1[i] != str2[i])
    {
      return false;
    }
    if (str1[i] == '\0')
    {
      return true;
    }
  }
  return true;
}

[[nodiscard]] __host__ __device__ constexpr size_t str_len(const char* str)
{
  size_t i{};
  for (; str[i]; ++i)
  {
  }
  return i;
}

struct TestData
{
  int val;
  const char* str_signed;
  const char* str_unsigned = str_signed;
};

template <int Base>
struct TestBaseData
{};

template <>
struct TestBaseData<2>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "10"},
    {3, "11"},
    {4, "100"},
    {7, "111"},
    {8, "1000"},
    {14, "1110"},
    {16, "10000"},
    {23, "10111"},
    {32, "100000"},
    {36, "100100"},
    {127, "1111111"},
    {8158, "1111111011110"},
    {17875098, "1000100001100000010011010"},
    {1787987597, "1101010100100101000011010001101"},
    {-1, "-1", "11111111111111111111111111111111"},
    {-13, "-1101", "11111111111111111111111111110011"},
    {-12345, "-11000000111001", "11111111111111111100111111000111"},
    // {-2147483648, "-10000000000000000000000000000000", "10000000000000000000000000000000"}
  };
};

template <>
struct TestBaseData<3>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "10"},
    {4, "11"},
    {7, "21"},
    {8, "22"},
    {14, "112"},
    {16, "121"},
    {23, "212"},
    {32, "1012"},
    {36, "1100"},
    {127, "11201"},
    {8158, "102012011"},
    {17875098, "1020122011000200"},
    {1787987597, "11121121102011212212"},
    {-1, "-1", "102002022201221111210"},
    {-13, "-111", "102002022201221111100"},
    {-12345, "-121221020", "102002022201022120121"},
    // {-2147483648, "-12112122212110202102", "121121222121102021022"}
  };
};

template <>
struct TestBaseData<4>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "10"},
    {7, "13"},
    {8, "20"},
    {14, "32"},
    {16, "100"},
    {23, "113"},
    {32, "200"},
    {36, "210"},
    {127, "1333"},
    {8158, "1333132"},
    {17875098, "1010030002122"},
    {1787987597, "1222210220122031"},
    {-1, "-1", "3333333333333333"},
    {-13, "-31", "3333333333333303"},
    {-12345, "-3000321", "3333333330333013"},
    // {-2147483648, "-2000000000000000", "20000000000000000"}
  };
};

template <>
struct TestBaseData<7>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "10"},
    {8, "11"},
    {14, "20"},
    {16, "22"},
    {23, "32"},
    {32, "44"},
    {36, "51"},
    {127, "241"},
    {8158, "32533"},
    {17875098, "304635663"},
    {1787987597, "62210433554"},
    {-1, "-1", "211301422353"},
    {-13, "-16", "211301422335"},
    {-12345, "-50664", "211301341360"},
    // {-2147483648, "-104134211162", "1041342111622"}
  };
};

template <>
struct TestBaseData<8>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "7"},
    {8, "10"},
    {14, "16"},
    {16, "20"},
    {23, "27"},
    {32, "40"},
    {36, "44"},
    {127, "177"},
    {8158, "17736"},
    {17875098, "104140232"},
    {1787987597, "15244503215"},
    {-1, "-1", "37777777777"},
    {-13, "-15", "37777777763"},
    {-12345, "-30071", "37777747707"},
    // {-2147483648, "-20000000000", "200000000000"}
  };
};

template <>
struct TestBaseData<14>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "7"},
    {8, "8"},
    {14, "10"},
    {16, "12"},
    {23, "19"},
    {32, "24"},
    {36, "28"},
    {127, "91"},
    {8158, "2d8a"},
    {17875098, "253436a"},
    {1787987597, "12d66ad9b"},
    {-1, "-1", "2ca5b7463"},
    {-13, "-d", "2ca5b7455"},
    {-12345, "-46db", "2ca5b2b67"},
    // {-2147483648, "-1652ca932", "1652ca9322"}
  };
};

template <>
struct TestBaseData<16>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "7"},
    {8, "8"},
    {14, "e"},
    {16, "10"},
    {23, "17"},
    {32, "20"},
    {36, "24"},
    {127, "7f"},
    {8158, "1fde"},
    {17875098, "110c09a"},
    {1787987597, "6a92868d"},
    {-1, "-1", "ffffffff"},
    {-13, "-d", "fffffff3"},
    {-12345, "-3039", "ffffcfc7"},
    // {-2147483648, "-80000000", "800000000"}
  };
};

template <>
struct TestBaseData<23>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "7"},
    {8, "8"},
    {14, "e"},
    {16, "g"},
    {23, "10"},
    {32, "19"},
    {36, "1d"},
    {127, "5c"},
    {8158, "f9g"},
    {17875098, "2hk384"},
    {1787987597, "c1i6jh4"},
    {-1, "-1", "1606k7ib"},
    {-13, "-d", "1606k7hm"},
    {-12345, "-107h", "1606j7ai"},
    // {-2147483648, "-ebelf96", "ebelf966"}
  };
};

template <>
struct TestBaseData<32>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "7"},
    {8, "8"},
    {14, "e"},
    {16, "g"},
    {23, "n"},
    {32, "10"},
    {36, "14"},
    {127, "3v"},
    {8158, "7uu"},
    {17875098, "h1g4q"},
    {1787987597, "1l951kd"},
    {-1, "-1", "3vvvvvv"},
    {-13, "-d", "3vvvvvj"},
    {-12345, "-c1p", "3vvvju7"},
    // {-2147483648, "-2000000", "20000000"}
  };
};

template <>
struct TestBaseData<36>
{
  static constexpr TestData data[]{
    {0, "0"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {7, "7"},
    {8, "8"},
    {14, "e"},
    {16, "g"},
    {23, "n"},
    {32, "w"},
    {36, "10"},
    {127, "3j"},
    {8158, "6am"},
    {17875098, "an4ii"},
    {1787987597, "tkis25"},
    {-1, "-1", "1z141z3"},
    {-13, "-d", "1z141yr"},
    {-12345, "-9ix", "1z13sg7"},
    // {-2147483648, "-zik0zk", "zik0zkk"}
  };
};

template <int Base>
__host__ __device__ constexpr void test_base()
{
  using TestData = TestBaseData<Base>;

  constexpr cuda::std::size_t buff_size = 150;

  for (const auto& data : TestData::data)
  {
    char buff[buff_size]{};

    // Test signed
    {
      const auto result = cuda::std::to_chars(buff, buff + buff_size, data.val, Base);
      assert(result.ec == cuda::std::errc{});
      assert(result.ptr == buff + str_len(data.str_signed));
      assert(str_equal(buff, data.str_signed, buff_size));
    }

    // Test unsigned
    {
      const auto result = cuda::std::to_chars(buff, buff + buff_size, static_cast<unsigned>(data.val), Base);
      assert(result.ec == cuda::std::errc{});
      assert(result.ptr == buff + str_len(data.str_unsigned));
      assert(str_equal(buff, data.str_unsigned, buff_size));
    }
  }
}

__host__ __device__ constexpr bool test()
{
  // basic tests
  // test_basics<char>{}();

  //   test_basics<signed char>{}();
  //   test_basics<signed short>{}();
  //   test_basics<signed int>{}();
  //   test_basics<signed long>{}();
  //   test_basics<signed long long>{}();
  // #if _CCCL_HAS_INT128()
  //   test_basics<__int128_t>{}();
  // #endif // _CCCL_HAS_INT128()

  //   test_basics<unsigned char>{}();
  //   test_basics<unsigned short>{}();
  //   test_basics<unsigned int>{}();
  //   test_basics<unsigned long>{}();
  //   test_basics<unsigned long long>{}();
  // #if _CCCL_HAS_INT128()
  //   test_basics<__uint128_t>{}();
  // #endif // _CCCL_HAS_INT128()

  //   // signed tests
  //   if constexpr (cuda::std::numeric_limits<char>::is_signed)
  //   {
  //     test_signed<char>{}();
  //   }

  //   test_signed<signed char>{}();
  //   test_signed<signed short>{}();
  //   test_signed<signed int>{}();
  //   test_signed<signed long>{}();
  //   test_signed<signed long long>{}();
  // #if _CCCL_HAS_INT128()
  //   test_signed<__int128_t>{}();
  // #endif // _CCCL_HAS_INT128()

  test_base<2>();
  test_base<3>();
  test_base<4>();
  test_base<7>();
  test_base<8>();
  test_base<14>();
  test_base<16>();
  test_base<23>();
  test_base<32>();
  test_base<36>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
