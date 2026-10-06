//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error:calling a __host__ __device__ function in tile is not allowed
// error: function-to-pointer decay is unsupported in tile code
// error: taking address of a function is unsupported in tile code

// <cuda/std/utility>

// _CCCL_TYPEID(<type>)

#define _CCCL_USE_TYPEID_FALLBACK

#include <cuda/std/__utility/typeid.h>
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_macros.h"

_CCCL_DIAG_SUPPRESS_GCC("-Wtautological-compare")

// Explicit instantiation makes GCC before 12 recognize that these objects are
// defined when comparing their addresses in constant expressions.
#if !defined(__CUDA_ARCH__) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG)
template cuda::std::__type_info const cuda::std::__typeid_v<int>;
template cuda::std::__type_info const cuda::std::__typeid_v<float>;
#endif

struct a_dummy_class_type
{};

TEST_FUNC constexpr bool test()
{
  static_assert(cuda::std::is_same_v<decltype((_CCCL_TYPEID(int))), cuda::std::__type_info_ref>);
  static_assert(noexcept(_CCCL_TYPEID(int)));
  static_assert(!cuda::std::is_default_constructible_v<cuda::std::type_info>);
  static_assert(!cuda::std::is_copy_constructible_v<cuda::std::type_info>);

  assert(_CCCL_TYPEID(int) == _CCCL_TYPEID(int));
  assert(!(_CCCL_TYPEID(int) != _CCCL_TYPEID(int)));
  assert(_CCCL_TYPEID(int) != _CCCL_TYPEID(float));
  assert(!(_CCCL_TYPEID(int) == _CCCL_TYPEID(float)));
  assert(_CCCL_TYPEID(const int) == _CCCL_TYPEID(int));
  assert(_CCCL_TYPEID(int&) != _CCCL_TYPEID(int));
  assert(_CCCL_TYPEID(int).before(_CCCL_TYPEID(float)) || _CCCL_TYPEID(float).before(_CCCL_TYPEID(int)));
  assert(_CCCL_TYPEID(int) == _CCCL_TYPEID(int));
  assert(!(_CCCL_TYPEID(int) != _CCCL_TYPEID(int)));
  assert(_CCCL_TYPEID(int) != _CCCL_TYPEID(float));
  assert(!(_CCCL_TYPEID(int) == _CCCL_TYPEID(float)));
  assert(_CCCL_TYPEID(const int) == _CCCL_TYPEID(int));
  assert(_CCCL_TYPEID(int&) != _CCCL_TYPEID(int));

  assert(&_CCCL_TYPEID(int) == &_CCCL_TYPEID(int));
  assert(&_CCCL_TYPEID(int) != &_CCCL_TYPEID(float));

  assert(_CCCL_TYPEID(float).before(_CCCL_TYPEID(int)));
  assert(!_CCCL_TYPEID(int).before(_CCCL_TYPEID(int)));
  assert(!_CCCL_TYPEID(int).before(_CCCL_TYPEID(float)));

  assert(_CCCL_TYPEID(int).__name_view() == "int");
  assert(_CCCL_TYPEID(float).__name_view() == "float");
  assert(_CCCL_TYPEID(a_dummy_class_type).__name_view().find("a_dummy_class_type") != -1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
