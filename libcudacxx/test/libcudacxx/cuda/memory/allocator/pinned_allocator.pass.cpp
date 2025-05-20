//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class T>
// struct cuda::pinned_allocator

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <stdexcept>

template <class T>
void test_type()
{
  using A = cuda::pinned_allocator<T>;

  // 1. test default constructors & assignment operators
  {
    static_assert(cuda::std::is_trivially_default_constructible_v<A>);
    static_assert(cuda::std::is_trivially_copy_constructible_v<A>);
    static_assert(cuda::std::is_trivially_copy_assignable_v<A>);
  }

  // 2. test construction from cuda::pinned_allocator instantiated with a different type
  {
    struct UniqueType
    {};
    static_assert(cuda::std::is_nothrow_constructible_v<A, cuda::pinned_allocator<UniqueType>>);
    static_assert(cuda::std::is_nothrow_assignable_v<A, cuda::pinned_allocator<UniqueType>>);
  }

  // 3. test value_type
  {
    static_assert(cuda::std::is_same_v<typename A::value_type, T>);
  }

  // 4. test allocate & deallocate signatures
  {
    static_assert(cuda::std::is_same_v<T*, decltype(A{}.allocate(cuda::std::size_t{}))>);
    static_assert(!noexcept(A{}.allocate(cuda::std::size_t{})));
    static_assert(cuda::std::is_same_v<void, decltype(A{}.deallocate(cuda::std::declval<T*>(), cuda::std::size_t{}))>);
    static_assert(noexcept(A{}.deallocate(cuda::std::declval<T*>(), cuda::std::size_t{})));
  }

  // 5. test allocate with size 0
  {
    const cuda::std::size_t size = 0;

    A a;
    T* p = a.allocate(size);
    assert(p == nullptr);
    a.deallocate(p, size);
  }

  // 6. test allocate with size > 0
  {
    const cuda::std::size_t size = 16;

    A a;
    T* p = a.allocate(size);
    assert(p != nullptr);
    a.deallocate(p, size);
  }

#if _CCCL_HAS_EXCEPTIONS()
  // 7. test allocate throws on bad_alloc
  {
    const cuda::std::size_t size = cuda::std::numeric_limits<cuda::std::size_t>::max() / sizeof(T);
    A a;
    try
    {
      T* p = a.allocate(size);
      assert(p == nullptr);
      a.deallocate(p, size);
    }
    catch (const std::bad_alloc&)
    {
      // expected
    }
    catch (...)
    {
      assert(false);
    }
  }

  // 8. test allocate throws on bad_array_new_length
  if constexpr (sizeof(T) > 1)
  {
    const cuda::std::size_t size = cuda::std::numeric_limits<cuda::std::size_t>::max();
    A a;
    try
    {
      T* p = a.allocate(size);
      assert(p == nullptr);
      a.deallocate(p, size);
    }
    catch (const std::bad_array_new_length&)
    {
      // expected
    }
    catch (...)
    {
      assert(false);
    }
  }
#endif // _CCCL_HAS_EXCEPTIONS()

  // 9. test equality operators
  {
    A a1;
    A a2;
    assert(a1 == a2);
    assert(!(a1 != a2));
  }

  // todo: tests on alignment?
}

struct S
{
  double x;
  double y;
};

void test()
{
  test_type<signed char>();
  test_type<int>();
  test_type<float>();
  test_type<S>();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))

  return 0;
}
