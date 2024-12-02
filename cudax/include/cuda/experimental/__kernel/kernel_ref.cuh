//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FUNCTION_KERNEL_REF
#define _CUDAX__FUNCTION_KERNEL_REF

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#if CUDA_VERSION >= 12010

namespace cuda::experimental
{
template <class>
class kernel_ref;

template <class... _Args>
class kernel_ref<void(_Args...)>
{
public:
  using value_type = ::cudaKernel_t;

  kernel_ref(_CUDA_VSTD::nullptr_t) = delete; // Delete construction from nullptr

  constexpr kernel_ref(value_type __kernel) noexcept
      : __kernel_(__kernel)
  {}

  kernel_ref(void (*__entry_func_address)(_Args...))
  {
    _CCCL_TRY_CUDA_API(
      ::cudaGetKernel, "Failed to get kernel from entry function address", &__kernel_, __entry_func_address);
  }

  _CCCL_NODISCARD ::std::string_view get_name() const
  {
    return detail::driver::kernelGetName(__kernel_);
  }

  _CCCL_NODISCARD constexpr value_type get() const noexcept
  {
    return __kernel_;
  }

  _CCCL_NODISCARD CUlibrary get_library() const
  {
    return detail::driver::kernelGetLibrary(__kernel_);
  }

  template <class _Attr>
  _CCCL_NODISCARD auto get_attr(const _Attr& __attr, device_ref __dev) const
  {
    // Get __attr value for __dev via cuKernelGetAttribute
  }

  template <class _Attr, class _Value>
  _CCCL_NODISCARD void set_attr(const _Attr& __attr, _Value&& __value, device_ref __dev) const
  {
    // Check __value type
    // Set __attr __value for __dev via cuKernelSetAttribute
  }

  template <class _CacheConfig>
  _CCCL_NODISCARD void set_cache_config(_CacheConfig __cacheConfig, device_ref __dev) const
  {
    // Set __cacheConfig for __dev via cuKernelSetCacheConfig
  }

  _CCCL_NODISCARD_FRIEND constexpr bool operator==(kernel_ref __lhs, kernel_ref __rhs) noexcept
  {
    return __lhs.__kernel_ == __rhs.__kernel_;
  }

private:
  value_type __kernel_{};
};
} // namespace cuda::experimental

#endif // CUDA_VERSION >= 12010

#endif // _CUDAX__FUNCTION_KERNEL_REF
