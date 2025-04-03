#include <cuda/std/__charconv_>
#include <cuda/std/limits>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdio>
#include <iostream>

constexpr auto val = unsigned(cuda::std::numeric_limits<int>::min());
constexpr int base = 4;

__global__ void kernel(ptrdiff_t diff)
{
  constexpr size_t size = 100;
  char str[size]{};

  const auto result = cuda::std::to_chars(str, str + size, val, base);
  printf("%s\n", str);

  if (result.ptr - str != diff)
  {
    printf("invalid diff %llu, should be %llu\n", result.ptr - str, diff);
  }
  if (result.ec != cuda::std::errc{})
  {
    printf("invalid ec\n");
  }
}

int main()
{
  constexpr size_t size = 100;
  char str[size]{};

  const auto result = cuda::std::to_chars(str, str + size, val, base);
  std::cout << str << std::endl;
  std::fill_n(str, size, 0);
  const auto ref = std::to_chars(str, str + size, val, base);
  std::cout << str << std::endl;

  assert(result.ptr == ref.ptr);
  assert(int(result.ec) == int(ref.ec));

  kernel<<<1, 1>>>(result.ptr - str);
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    std::cerr << "Kernel launch failed!" << std::endl;
    return 1;
  }
}
