//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// logical_or

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::logical_or<int> F;
    const F f = F();
#if TEST_STD_VER <= 17
    static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((cuda::std::is_same<bool, F::result_type>::value), "" );
#endif
    assert(f(36, 36));
    assert(f(36, 0));
    assert(f(0, 36));
    assert(!f(0, 0));
#if TEST_STD_VER > 11
    typedef cuda::std::logical_or<> F2;
    const F2 f2 = F2();
    assert( f2(36, 36));
    assert( f2(36, 36L));
    assert( f2(36L, 36));
    assert( f2(36, 0));
    assert( f2(0, 36));
    assert( f2(36, 0L));
    assert( f2(0, 36L));
    assert(!f2(0, 0));
    assert(!f2(0, 0L));
    assert(!f2(0L, 0));

    constexpr bool foo = cuda::std::logical_or<int> () (36, 36);
    static_assert ( foo, "" );

    constexpr bool bar = cuda::std::logical_or<> () (36.0, 36);
    static_assert ( bar, "" );
#endif

  return 0;
}
