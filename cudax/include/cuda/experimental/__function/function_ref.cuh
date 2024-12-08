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

#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__function/attributes.cuh>
#include <cuda/experimental/__kernel/kernel_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#include <string_view>

namespace cuda::experimental
{

//! @brief A non-owning representation of a CUDA function
//!
//! @tparam _Signature The signature of the function
//!
//! @note The return type of the function must be `void`
template <class _Signature>
class function_ref;

template <class... _Args>
class function_ref<void(_Args...)>
{
public:
  using value_type = ::CUfunction;

  function_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Construct a `function_ref` object from a native function handle
  //!
  //! @param __function The native function handle
  constexpr function_ref(value_type __function) noexcept
      : __function_(__function)
  {}

  //! @brief Construct a `function_ref` object from an entry function address for the current context
  //!
  //! @param __entry_func_address The entry function address
  //!
  //! @throws cuda_error if the function creation fails
  function_ref(void (*__entry_func_address)(_Args...))
      : function_ref{__entry_func_address, detail::driver::ctxGetCurrent()}
  {}

  //! @brief Construct a `function_ref` object from an entry function address for a specified context
  //!
  //! @param __entry_func_address The entry function address
  //! @param __context The context in which to create the function
  //!
  //! @throws cuda_error if the function creation fails
  function_ref(void (*__entry_func_address)(_Args...), CUcontext __context)
  {
    _CCCL_ASSERT(__entry_func_address != nullptr,
                 "cuda::experimental::function_ref invalid entry function address passed");
    _CCCL_ASSERT(__context != nullptr, "cuda::experimental::function_ref invalid context passed");

    [[maybe_unused]] __ensure_current_device __current_dev{__context};

    void* __symbol_ptr;

    _CCCL_TRY_CUDA_API(::cudaGetSymbolAddress,
                       "Failed to get symbol address from entry function address",
                       &__symbol_ptr,
                       (const void*) __entry_func_address);

    _CCCL_TRY_CUDA_API(::cudaGetFuncBySymbol, "Failed to get kernel from symbol address", &__function_, __symbol_ptr);
  }

#if CUDA_VERSION >= 12000
  //! @brief Construct a `function_ref` object from a kernel reference for the current context
  //!
  //! @param __kernel The kernel reference
  //!
  //! @throws cuda_error if the function creation fails
  function_ref(kernel_ref<void(_Args...)> __kernel)
      : function_ref{__kernel, detail::driver::ctxGetCurrent()}
  {}

  //! @brief Construct a `function_ref` object from a kernel reference for a specified context
  //!
  //! @param __kernel The kernel reference
  //! @param __context The context in which to create the function
  //!
  //! @throws cuda_error if the function creation fails
  function_ref(kernel_ref<void(_Args...)> __kernel, CUcontext __context)
  {
    _CCCL_ASSERT(__kernel.get() != nullptr, "cuda::experimental::function_ref invalid kernel passed");
    _CCCL_ASSERT(__context != nullptr, "cuda::experimental::function_ref invalid context passed");

    [[maybe_unused]] __ensure_current_device __current_dev{__context};

    __function_ = detail::driver::kernelGetFunction(__kernel.get());
  }
#endif // CUDA_VERSION >= 12000

  //! @brief Get the name of the function
  //!
  //! @return The name of the function
  //!
  //! @throws cuda_error if the function name retrieval fails
  _CCCL_NODISCARD ::std::string_view get_name() const
  {
    return detail::driver::funcGetName(__function_);
  }

  //! @brief Retrieve the native function handle
  //!
  //! @return The native function handle
  _CCCL_NODISCARD constexpr value_type get() const noexcept
  {
    return __function_;
  }

  //! @brief Retrieve the module containing the function
  //!
  //! @return The module containing the function
  //!
  //! @throws cuda_error if the module retrieval fails
  _CCCL_NODISCARD CUmodule get_module() const
  {
    return detail::driver::funcGetModule(__function_);
  }

  //! @brief Get an attribute of the function
  //!
  //! @param __attr The attribute to query
  //!
  //! @return The value of the attribute
  //!
  //! @throws cuda_error if the attribute query fails
  template <class _Attr>
  _CCCL_NODISCARD typename _Attr::type get_attr(const _Attr& __attr) const
  {
    return __attr.get(__function_);
  }

  //! @overload
  template <cudaFuncAttribute _Attr>
  _CCCL_NODISCARD auto get_attr() const
  {
    return get_attr(detail::__func_attr<static_cast<CUfunction_attribute>(_Attr)>{});
  }

  //! @brief Set an attribute of the function
  //!
  //! @param __attr The attribute to set
  //! @param __value The value to set the attribute to
  //!
  //! @throws cuda_error if the attribute set fails
  template <class _Attr>
  void set_attr(const _Attr& __attr, typename _Attr::type __value) const
  {
    __attr.set(__function_, __value);
  }

  //! @overload
  template <cudaFuncAttribute _Attr, class _Value>
  void set_attr(_Value&& __value) const
  {
    set_attr(detail::__func_attr<static_cast<CUfunction_attribute>(_Attr)>{}, _CUDA_VSTD::forward<_Value>(__value));
  }

  //! @brief Set the cache configuration of the function
  //!
  //! @param __cache_config The cache configuration to set
  //!
  //! @throws cuda_error if the cache configuration set fails
  void set_cache_config(cudaFuncCache __cache_config) const
  {
    detail::driver::funcSetCacheConfig(__function_, static_cast<CUfunc_cache>(__cache_config));
  }

  //! @brief Check if the function is loaded
  //!
  //! @return true if the function is loaded, false otherwise
  //!
  //! @throws cuda_error if the function state query fails
  _CCCL_NODISCARD bool is_loaded() const
  {
    return detail::driver::funcIsLoaded(__function_) == CU_FUNCTION_LOADING_STATE_LOADED;
  }

  //! @brief Load the function
  //!
  //! @throws cuda_error if the function load fails
  void load() const
  {
    detail::driver::funcLoad(__function_);
  }

  //! @brief Compares two `function_ref`s for equality
  //!
  //! @param __lhs The first `function_ref` to compare
  //! @param __rhs The second `function_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same function
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(function_ref __lhs, function_ref __rhs) noexcept
  {
    return __lhs.__function_ == __rhs.__function_;
  }

private:
  value_type __function_{};
};

namespace detail
{

template <::CUfunction_attribute _Attr, class _Type, bool _ReadOnly>
template <class... _Args>
_CCCL_NODISCARD auto
__func_attr_impl<_Attr, _Type, _ReadOnly>::operator()(function_ref<void(_Args...)> __function) const -> type
{
  return get(__function.get());
}

} // namespace detail

} // namespace cuda::experimental

#endif // _CUDAX__FUNCTION_FUNCTION_REF
