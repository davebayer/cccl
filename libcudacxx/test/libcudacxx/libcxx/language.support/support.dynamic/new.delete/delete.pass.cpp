//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__new_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

int main(int, char**)
{
  // 1. delete(void*)
  {
    void* ptr = cuda::std::__cccl_operator_new(10);
    assert(ptr);
    cuda::std::__cccl_operator_delete(ptr);
  }

  // 2. delete(void*, align_val_t)
  {
    constexpr cuda::std::align_val_t alignment{32};
    void* ptr = cuda::std::__cccl_operator_new(10, alignment);
    assert(ptr);
    assert(reinterpret_cast<cuda::std::uintptr_t>(ptr) % static_cast<cuda::std::size_t>(alignment) == 0);
    cuda::std::__cccl_operator_delete(ptr, alignment);
  }

  // 3. delete(void*, size_t)
  {
    constexpr cuda::std::size_t size{128};
    void* ptr = cuda::std::__cccl_operator_new(size);
    assert(ptr);
    cuda::std::__cccl_operator_delete(ptr, size);
  }

  // 4. delete(void*, size_t, align_val_t)
  {
    constexpr cuda::std::size_t size{128};
    constexpr cuda::std::align_val_t alignment{32};
    void* ptr = cuda::std::__cccl_operator_new(size, alignment);
    assert(ptr);
    cuda::std::__cccl_operator_delete(ptr, size, alignment);
  }

  // 5. delete(void*, const nothrow_t&)
  {
    void* ptr = cuda::std::__cccl_operator_new(10);
    assert(ptr);
    cuda::std::__cccl_operator_delete(ptr, cuda::std::nothrow);
  }

  // 6. delete(void*, align_val_t, const nothrow_t&)
  {
    constexpr cuda::std::align_val_t alignment{32};
    void* ptr = cuda::std::__cccl_operator_new(10, alignment);
    assert(ptr);
    assert(reinterpret_cast<cuda::std::uintptr_t>(ptr) % static_cast<cuda::std::size_t>(alignment) == 0);
    cuda::std::__cccl_operator_delete(ptr, alignment, cuda::std::nothrow);
  }

  return 0;
}
