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
#include <cuda/experimental/__function/attributes.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#include <string_view>

#if CUDA_VERSION >= 12000

namespace cuda::experimental
{

//! @brief A non-owning representation of a CUDA kernel
//!
//! @tparam _Signature The signature of the kernel
//!
//! @note The return type of the kernel must be `void`
template <class _Signature>
class kernel_ref;

template <class... _Args>
class kernel_ref<void(_Args...)>
{
public:
  using value_type = ::CUkernel;

  kernel_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Constructs a `kernel_ref` from a kernel object
  //!
  //! @param __kernel The kernel object
  constexpr kernel_ref(value_type __kernel) noexcept
      : __kernel_(__kernel)
  {}

#  if CUDA_VERSION >= 12010
  //! @brief Constructs a `kernel_ref` from an entry function address
  //!
  //! @param __entry_func_address The entry function address
  //!
  //! @throws cuda_error if the kernel cannot be obtained from the entry function address
  kernel_ref(void (*__entry_func_address)(_Args...))
  {
    _CCCL_TRY_CUDA_API(
      ::cudaGetKernel, "Failed to get kernel from entry function address", &__kernel_, __entry_func_address);
  }
#  endif // CUDA_VERSION >= 12010

  //! @brief Get the name of the kernel
  //!
  //! @return The name of the kernel
  //!
  //! @throws cuda_error if the kernel name cannot be obtained
  _CCCL_NODISCARD ::std::string_view get_name() const
  {
    return detail::driver::kernelGetName(__kernel_);
  }

  //! Retrieve the native kernel handle
  //!
  //! @return The native kernel handle
  _CCCL_NODISCARD constexpr value_type get() const noexcept
  {
    return __kernel_;
  }

  //! @brief Get the library that the kernel belongs to
  //!
  //! @return The library that the kernel belongs to
  //!
  //! @throws cuda_error if the library cannot be obtained
  _CCCL_NODISCARD CUlibrary get_library() const
  {
    return detail::driver::kernelGetLibrary(__kernel_);
  }

  //! @brief Get the attributes of the kernel
  //!
  //! @return The attributes of the kernel
  //!
  //! @throws cuda_error if the kernel attributes cannot be obtained
  template <class _Attr>
  _CCCL_NODISCARD function_attr_result_t<_Attr> get_attr(const _Attr& __attr, device_ref __dev) const
  {
    return __attr.get(__kernel_, driver::deviceGet(__dev.get()));
  }

  //! @overload
  template <cudaFuncAttribute _Attr>
  _CCCL_NODISCARD function_attr_result_t<_Attr> get_attr(device_ref __dev) const
  {
    return get_attr(detail::__func_attr<_Attr>{}, __dev);
  }

  //! @brief Set the attributes of the kernel
  //!
  //! @param __attr The attribute to set
  //! @param __value The value to set the attribute to
  //!
  //! @throws cuda_error if the kernel attributes cannot be set
  template <class _Attr>
  void set_attr(const _Attr& __attr, function_attr_result_t<_Attr> __value, device_ref __dev) const
  {
    __attr.set(__kernel_, __value, driver::deviceGet(__dev.get()));
  }

  //! @overload
  template <cudaFuncAttribute _Attr>
  void set_attr(function_attr_result_t<_Attr> __value, device_ref __dev) const
  {
    set_attr(detail::__func_attr<_Attr>{}, __value, __dev);
  }

  //! @brief Set the cache configuration of the kernel
  //!
  //! @param __cacheConfig The cache configuration to set
  //!
  //! @throws cuda_error if the cache configuration cannot be set
  void set_cache_config(cudaFuncCache __cache_config, device_ref __dev) const
  {
    detail::driver::kernelSetCacheConfig(
      __kernel_, static_cast<CUfunc_cache>(__cache_config), detail::driver::deviceGet(__dev.get()));
  }

  //! @brief Compares two `kernel_ref` for equality
  //!
  //! @param __lhs The first `kernel_ref` to compare
  //! @param __rhs The second `kernel_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same function
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(kernel_ref __lhs, kernel_ref __rhs) noexcept
  {
    return __lhs.__kernel_ == __rhs.__kernel_;
  }

private:
  value_type __kernel_{};
};

namespace detail
{

template <::CUfunction_attribute _Attr, class _Type>
template <class... _Args>
_CCCL_NODISCARD auto
__func_attr_impl<_Attr, _Type>::operator()(kernel_ref<void(_Args...)> __kernel, device_ref __dev) const -> type
{
  return get(__kernel.get(), driver::deviceGet(__dev.get()));
}

} // namespace detail

} // namespace cuda::experimental

#endif // CUDA_VERSION >= 12000

#endif // _CUDAX__FUNCTION_KERNEL_REF
