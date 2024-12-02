//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FUNCTION_FUNCTION_REF
#define _CUDAX__FUNCTION_FUNCTION_REF

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__function/attributes.cuh>
#include <cuda/experimental/__kernel/kernel_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

namespace cuda::experimental
{
template <class = void>
class function
{
public:
  using attrs = __func_attrs;

  template <::CUfunction_attribute _Attr>
  using attr_result_t = delctype(_CUDA_VSTD::declval<detail::__func_attr<_Attr>>.get());
};

template <class>
class function_ref;

template <class _Ret, class... _Args>
class function_ref<_Ret(_Args...)>
{
public:
  using value_type = ::cudaFunction_t;

  function_ref(_CUDA_VSTD::nullptr_t) = delete; // Delete construction from nullptr

  constexpr function_ref(value_type __function) noexcept
      : __function_(__function)
  {}

#if CUDA_VERSION >= 12010
  function_ref(kernel_ref<void(_Args...)> __kernel)
  {
    static_assert(_CUDA_VSTD::is_void_v<_Ret>, "The return type of the kernel must be void");

    __function_ = detail::driver::kernelGetFunction(__kernel.get());
  }

  function_ref(kernel_ref<void(_Args...)> __kernel, CUcontext __context)
  {
    static_assert(_CUDA_VSTD::is_void_v<_Ret>, "The return type of the kernel must be void");

    __ensure_current_device __current_dev{__context};

    __function_ = detail::driver::kernelGetFunction(__kernel.get());
  }
#endif // CUDA_VERSION >= 12010

  _CCCL_NODISCARD ::std::string_view get_name() const
  {
    return detail::driver::funcGetName(__function_);
  }

  _CCCL_NODISCARD constexpr value_type get() const noexcept
  {
    return __function_;
  }

  _CCCL_NODISCARD CUmodule get_module() const
  {
    return detail::driver::funcGetModule(__function_);
  }

  template <class _Attr>
  _CCCL_NODISCARD function<>::attr_result_t<_Attr> get_attr(const _Attr& __attr) const
  {
    return __attr.get(__function_);
  }

  template <class _Attr>
  void set_attr(const _Attr& __attr, function<>::attr_result_t<_Attr> __value) const
  {
    __attr.set(__function_, __value);
  }

  template <class _CacheConfig>
  _CCCL_NODISCARD void set_cache_config(_CacheConfig __cacheConfig) const
  {
    // Set __cacheConfig via cuFunctionSetCacheConfig
  }

  _CCCL_NODISCARD bool is_loaded() const
  {
    return detail::driver::funcIsLoaded(__function_) == CU_FUNCTION_LOADING_STATE_LOADED;
  }

  void load() const
  {
    detail::driver::funcLoad(__function_);
  }

  _CCCL_NODISCARD_FRIEND constexpr bool operator==(function_ref __lhs, function_ref __rhs) noexcept
  {
    return __lhs.__function_ == __rhs.__function_;
  }

private:
  value_type __function_{};
};
} // namespace cuda::experimental

#endif // _CUDAX__FUNCTION_FUNCTION_REF
