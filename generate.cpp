#include <charconv>
#include <cstring>
#include <iostream>
#include <limits>

constexpr long long values[]{
  0,
  1,
  2,
  3,
  4,
  7,
  8,
  14,
  16,
  23,
  32,
  36,
  std::numeric_limits<signed char>::max(),
  200,
  std::numeric_limits<unsigned char>::max(),
  8158,
  50243,
  std::numeric_limits<signed short>::max(),
  87231,
  std::numeric_limits<unsigned short>::max(),
  17875098,
  1787987597,
  std::numeric_limits<int>::max(),
  std::numeric_limits<unsigned int>::max(),
  657687465411ll,
  89098098408713,
  4987654351689798,
  std::numeric_limits<long long>::max(),
  -1,
  -13,
  std::numeric_limits<signed char>::min(),
  -879,
  -12345,
  std::numeric_limits<signed short>::min(),
  -5165781,
  -97897347,
  std::numeric_limits<int>::min(),
  -165789751156,
  -8798798743521135,
  std::numeric_limits<long long>::min()};

void print_base(int base)
{
  constexpr int size = 150;

  std::cout
    << "template <>\nconstexpr cuda::std::array<TestItem, " << std::size(values) << "> test_items<" << base << ">{{\n";

  for (const auto value : values)
  {
    std::cout << "  TestItem{";
    if (value == std::numeric_limits<long long>::min())
    {
      std::cout << "-9223372036854775807 - 1";
    }
    else
    {
      std::cout << value;
    }
    std::cout << ", \"";

    char buff[size]{};
    std::to_chars(buff, buff + size, value, base);
    std::cout << buff;

    if (value < 0)
    {
      std::cout << "\", \"";
      std::memset(buff, 0, size);
      std::to_chars(buff, buff + size, (unsigned long long) (value), base);
      std::cout << buff;
    }
    std::cout << "\"},\n";
  }
  std::cout << "}};";
}

int main()
{
  for (std::size_t b = 2; b <= 36; ++b)
  {
    print_base(b);
    std::cout << "\n\n";
  }
}
