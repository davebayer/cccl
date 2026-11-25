//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Concrete implementations of data_place_interface
 *
 * This file contains implementations for standard data place types:
 * host, managed, device, invalid, affine, and device_auto.
 */

#pragma once

#include <cuda/__cccl_config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__memory_pool/memory_pool_base.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__host_stdlib/stdexcept>

#include <cuda/experimental/__places/data_place_interface.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/exception_policy.cuh>

#include <string>

#include <cuda.h>

namespace cuda::experimental::places
{
using ::cuda::experimental::stf::cuda_try;

/**
 * @brief Implementation for the invalid data place
 */
class data_place_invalid final : public data_place_interface
{
public:
  _CCCL_HOST_API bool is_resolved() const override
  {
    return false;
  }

  _CCCL_HOST_API int get_device_ordinal() const override
  {
    return data_place_interface::invalid;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "invalid";
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::invalid);
  }

  _CCCL_HOST_API int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  _CCCL_HOST_API void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::logic_error, "Cannot allocate from invalid data_place");
  }

  _CCCL_HOST_API void deallocate(void*, size_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::logic_error, "Cannot deallocate from invalid data_place");
  }

  _CCCL_HOST_API bool allocation_is_stream_ordered() const override
  {
    return false;
  }
};

/**
 * @brief Implementation for the host (pinned memory) data place
 */
class data_place_host final : public data_place_interface
{
public:
  _CCCL_HOST_API bool is_resolved() const override
  {
    return true;
  }

  _CCCL_HOST_API int get_device_ordinal() const override
  {
    return data_place_interface::host;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "host";
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::host);
  }

  _CCCL_HOST_API int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  _CCCL_HOST_API void* allocate(::std::ptrdiff_t size, cudaStream_t) const override
  {
    void* result = nullptr;
    cuda_try(cudaMallocHost(&result, static_cast<size_t>(size)));
    return result;
  }

  _CCCL_HOST_API void deallocate(void* ptr, size_t, cudaStream_t) const override
  {
    cuda_try(cudaFreeHost(ptr));
  }

  _CCCL_HOST_API bool allocation_is_stream_ordered() const override
  {
    return false;
  }

  _CCCL_HOST_API CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const override
  {
#if _CCCL_CTK_AT_LEAST(12, 2)
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_HOST;
    prop.location.id         = 0;
    return cuMemCreate(handle, size, &prop, 0);
#else
    (void) handle;
    (void) size;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
  }
};

/**
 * @brief Implementation for managed memory data place
 */
class data_place_managed final : public data_place_interface
{
public:
  _CCCL_HOST_API bool is_resolved() const override
  {
    return true;
  }

  _CCCL_HOST_API int get_device_ordinal() const override
  {
    return data_place_interface::managed;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "managed";
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::managed);
  }

  _CCCL_HOST_API int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  _CCCL_HOST_API void* allocate(::std::ptrdiff_t size, cudaStream_t) const override
  {
    void* result = nullptr;
    cuda_try(cudaMallocManaged(&result, static_cast<size_t>(size)));
    return result;
  }

  _CCCL_HOST_API void deallocate(void* ptr, size_t, cudaStream_t) const override
  {
    cuda_try(cudaFree(ptr));
  }

  _CCCL_HOST_API bool allocation_is_stream_ordered() const override
  {
    return false;
  }
};

/**
 * @brief Implementation for a specific CUDA device data place
 */
class data_place_device final : public data_place_interface
{
public:
  _CCCL_HOST_API bool is_resolved() const override
  {
    return true;
  }

  _CCCL_HOST_API explicit data_place_device(int device_id)
      : device_id_(device_id)
  {
    _CCCL_ASSERT(device_id >= 0, "Device ID must be non-negative");
  }

  _CCCL_HOST_API int get_device_ordinal() const override
  {
    return device_id_;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "dev" + ::std::to_string(device_id_);
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<int>()(device_id_);
  }

  _CCCL_HOST_API int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    const int other_id = other.get_device_ordinal();
    return (device_id_ > other_id) - (device_id_ < other_id);
  }

  _CCCL_HOST_API void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const override
  {
    // The device's primary context must be current for a driver allocation on
    // the null stream (cudaMallocAsync created it on demand; the driver call
    // does not): libcu++'s scope pushes it, initializing it on first use.
    const ::cuda::__ensure_current_context __ctx{::cuda::device_ref{device_id_}};
    // The device's default pool, resolved through libcu++'s policy site (which
    // also sets the release threshold): places do not manage pool properties.
    // Filled member by member: the layout of CUmemLocation differs between
    // toolkits (its id became a member of an anonymous union).
    CUmemLocation location{};
    location.type           = CU_MEM_LOCATION_TYPE_DEVICE;
    location.id             = device_id_;
    const CUmemoryPool pool = ::cuda::__get_default_memory_pool(location, ::CU_MEM_ALLOCATION_TYPE_PINNED);
    CUdeviceptr ptr         = 0;
    cuda_try(cuMemAllocFromPoolAsync(&ptr, static_cast<size_t>(size), pool, reinterpret_cast<CUstream>(stream)));
    return reinterpret_cast<void*>(ptr); // NOLINT(performance-no-int-to-ptr)
  }

  _CCCL_HOST_API void deallocate(void* ptr, size_t, cudaStream_t stream) const override
  {
    const ::cuda::__ensure_current_context __ctx{::cuda::device_ref{device_id_}};
    cuda_try(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(ptr), reinterpret_cast<CUstream>(stream)));
  }

  _CCCL_HOST_API bool allocation_is_stream_ordered() const override
  {
    return true;
  }

  _CCCL_HOST_API CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const override
  {
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = device_id_;
    return cuMemCreate(handle, size, &prop, 0);
  }

private:
  int device_id_;
};

/**
 * @brief Implementation for the affine data place (uses exec_place's affine data place)
 */
class data_place_affine final : public data_place_interface
{
public:
  _CCCL_HOST_API bool is_resolved() const override
  {
    return false;
  }

  _CCCL_HOST_API int get_device_ordinal() const override
  {
    return data_place_interface::affine;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "affine";
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::affine);
  }

  _CCCL_HOST_API int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  _CCCL_HOST_API void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::logic_error, "Cannot allocate from affine data_place directly");
  }

  _CCCL_HOST_API void deallocate(void*, size_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::logic_error, "Cannot deallocate from affine data_place directly");
  }

  _CCCL_HOST_API bool allocation_is_stream_ordered() const override
  {
    return false;
  }
};

/**
 * @brief Implementation for device_auto data place (auto-select device)
 */
class data_place_device_auto final : public data_place_interface
{
public:
  _CCCL_HOST_API bool is_resolved() const override
  {
    return false;
  }

  _CCCL_HOST_API int get_device_ordinal() const override
  {
    return data_place_interface::device_auto;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "auto";
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::device_auto);
  }

  _CCCL_HOST_API int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  _CCCL_HOST_API void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::logic_error, "Cannot allocate from device_auto data_place directly");
  }

  _CCCL_HOST_API void deallocate(void*, size_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::logic_error, "Cannot deallocate from device_auto data_place directly");
  }

  _CCCL_HOST_API bool allocation_is_stream_ordered() const override
  {
    return true;
  }
};
} // end namespace cuda::experimental::places
