//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FUNCTION_ATTRIBUTES
#define _CUDAX__FUNCTION_ATTRIBUTES

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

namespace cuda::experimental
{

#if CUDA_VERSION >= 12000
template <class>
class kernel_ref;
#endif // CUDA_VERSION >= 12000

template <class>
class function_ref;

namespace detail
{

template <::CUfunction_attribute _Attr, class _Type, bool _ReadOnly>
class __func_attr_impl
{
#if CUDA_VERSION >= 12000
  template <class>
  friend class kernel_ref;
#endif // CUDA_VERSION >= 12000

  template <class>
  friend class function_ref;

public:
  using type = _Type;

  static constexpr bool read_only = _ReadOnly;

  _CCCL_NODISCARD constexpr operator ::CUfunction_attribute() const noexcept
  {
    return _Attr;
  }

#if CUDA_VERSION >= 12000
  // Implemented in __kernel/kernel_ref.cuh
  template <class... _Args>
  _CCCL_NODISCARD type operator()(kernel_ref<void(_Args...)> __kernel, device_ref __dev) const;
#endif // CUDA_VERSION >= 12000

  // Implemented in __function/function_ref.cuh
  template <class... _Args>
  _CCCL_NODISCARD type operator()(function_ref<void(_Args...)> __kernel) const;

private:
#if CUDA_VERSION >= 12000
  _CCCL_NODISCARD type get(CUkernel __kernel, CUdevice __dev) const
  {
    return static_cast<type>(driver::kernelGetAttribute(_Attr, __kernel, __dev));
  }

  _CCCL_TEMPLATE(bool _RO = read_only)
  _CCCL_REQUIRES(!_RO)
  void set(CUkernel __kernel, type __value, CUdevice __dev) const
  {
    driver::kernelSetAttribute(__kernel, _Attr, static_cast<int>(__value), __dev);
  }
#endif // CUDA_VERSION >= 12000

  _CCCL_NODISCARD type get(CUfunction __func) const
  {
    return static_cast<type>(driver::funcGetAttribute(_Attr, __func));
  }

  _CCCL_TEMPLATE(bool _RO = read_only)
  _CCCL_REQUIRES(!_RO)
  void set(CUfunction __func, type __value) const
  {
    driver::funcSetAttribute(__func, _Attr, static_cast<int>(__value));
  }
};

template <::CUfunction_attribute _Attr>
struct __func_attr;

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, int, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, size_t, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, size_t, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, size_t, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_NUM_REGS> : __func_attr_impl<CU_FUNC_ATTRIBUTE_NUM_REGS, int, true>
{};

// todo: change value_type to strong type (PTX version)
template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_PTX_VERSION> : __func_attr_impl<CU_FUNC_ATTRIBUTE_PTX_VERSION, int, true>
{};

// todo: change value_type to strong type (arch version)
template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_BINARY_VERSION> : __func_attr_impl<CU_FUNC_ATTRIBUTE_BINARY_VERSION, int, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_CACHE_MODE_CA> : __func_attr_impl<CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, bool, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size_t, false>
{};

// todo: change value_type to strong type (percentage)
template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, int, false>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, bool, true>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, int, false>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, int, false>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, int, false>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, bool, false>
{};

template <>
struct __func_attr<CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE>
    : __func_attr_impl<CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, cudaClusterSchedulingPolicy, false>
{
  static constexpr type default_value  = cudaClusterSchedulingPolicyDefault;
  static constexpr type spread         = cudaClusterSchedulingPolicySpread;
  static constexpr type load_balancing = cudaClusterSchedulingPolicyLoadBalancing;
};

} // namespace detail

//! @brief Function attributes for CUDA functions and kernels
struct function_attrs
{
  // Maximum number of threads per block
  using max_threads_per_block_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK>;
  static constexpr max_threads_per_block_t max_threads_per_block{};

  // Shared memory size in bytes
  using shared_size_bytes_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES>;
  static constexpr shared_size_bytes_t shared_size_bytes{};

  // Constant memory size in bytes
  using const_size_bytes_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES>;
  static constexpr const_size_bytes_t const_size_bytes{};

  // Local memory size in bytes
  using local_size_bytes_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES>;
  static constexpr local_size_bytes_t local_size_bytes{};

  // Number of registers used
  using num_regs_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_NUM_REGS>;
  static constexpr num_regs_t num_regs{};

  // PTX version
  using ptx_version_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_PTX_VERSION>;
  static constexpr ptx_version_t ptx_version{};

  // Binary version
  using binary_version_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_BINARY_VERSION>;
  static constexpr binary_version_t binary_version{};

  // Cache mode
  using cache_mode_ca_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_CACHE_MODE_CA>;
  static constexpr cache_mode_ca_t cache_mode_ca{};

  // Maximum dynamic shared memory size
  using max_dynamic_shared_size_bytes_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES>;
  static constexpr max_dynamic_shared_size_bytes_t max_dynamic_shared_size_bytes{};

  // Preferred shared memory carveout
  using preferred_shared_memory_carveout_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT>;
  static constexpr preferred_shared_memory_carveout_t preferred_shared_memory_carveout{};

  // Cluster size must be set
  using cluster_size_must_be_set_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET>;
  static constexpr cluster_size_must_be_set_t cluster_size_must_be_set{};

  // Requested cluster width
  using requested_cluster_width_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH>;
  static constexpr requested_cluster_width_t requested_cluster_width{};

  // Requested cluster height
  using requested_cluster_height_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT>;
  static constexpr requested_cluster_height_t requested_cluster_height{};

  // Requested cluster depth
  using requested_cluster_dim_depth_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH>;
  static constexpr requested_cluster_dim_depth_t requested_cluster_dim_depth{};

  // Non-portable cluster size allowed
  using non_portable_cluster_size_allowed_t = detail::__func_attr<CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED>;
  static constexpr non_portable_cluster_size_allowed_t non_portable_cluster_size_allowed{};

  // Cluster scheduling policy preference
  using cluster_scheduling_policy_preference_t =
    detail::__func_attr<CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE>;
  static constexpr cluster_scheduling_policy_preference_t cluster_scheduling_policy_preference{};
};

} // namespace cuda::experimental

#endif // _CUDAX__FUNCTION_ATTRIBUTES
