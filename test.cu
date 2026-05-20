struct T
{
  operator int()
  {
    return 0;
  }
};

namespace test
{
template <bool _Dummy = false>
__device__ void atomicAdd(...)
{
  static_assert(_Dummy);
}
using ::atomicAdd;
// using ::omg;
} // namespace test

__global__ void kernel(int v)
{
  test::atomicAdd(nullptr, T{});

  if constexpr (false)
  {
    test::atomicAdd();
  }
}
