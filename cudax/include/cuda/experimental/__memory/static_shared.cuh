//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MEMORY_STATIC_SHARED_H
#define _CUDA_EXPERIMENTAL___MEMORY_STATIC_SHARED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/no_init.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_fully_bounded_array.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_destructible.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/rank.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__memory/static_shared_storage.cuh>

namespace cuda::experimental
{
template <class _Tp, ::cuda::std::size_t _Align>
class __static_shared_base : protected static_shared_storage<sizeof(_Tp), _Align>
{
  static_assert(::cuda::is_power_of_two(_Align), "_Align must be power of two");
  static_assert(_Align >= alignof(_Tp), "_Align must be at least alignof(_Tp)");

  using __base_type = static_shared_storage<sizeof(_Tp), _Align>;

protected:
  using value_type   = _Tp; //!< The type of the object stored in the static shared memory.
  using element_type = ::cuda::std::remove_all_extents_t<_Tp>; //!< The element type of the object in the static shared
                                                               //!< memory.

  enum class __state_type
  {
    __uninitialized, //!< The object is created but in uninitialized state.
    __constructed, //!< The object is created and initialized.
    __destroyed, //!< The object is destroyed.
  };

  __state_type __state_{__state_type::__uninitialized}; //!< The state of the object.

  _CCCL_DEVICE_API _CCCL_FORCEINLINE __static_shared_base() noexcept {}

  __static_shared_base(const __static_shared_base&) = delete;

  __static_shared_base(__static_shared_base&&) = delete;

  __static_shared_base& operator=(const __static_shared_base&) = delete;

  __static_shared_base& operator=(__static_shared_base&&) = delete;

  //! @brief Gets the pointer to the object stored in the static shared memory.
  //!
  //! @return The pointer to the object stored in the static shared memory.
  [[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<element_type> __ptr() const noexcept
  {
    return shared_memory_ptr<element_type>{__base_type::get().__get_smem_addr()};
  }

  //! @brief Gets a reference to the object stored in the static shared memory.
  //!
  //! @returns A reference to the object stored in the static shared memory.
  [[nodiscard]] _CCCL_DEVICE_API _Tp& get() const noexcept
  {
    if constexpr (!::cuda::std::is_trivially_default_constructible_v<element_type>)
    {
      _CCCL_ASSERT(__state_ != __state_type::__uninitialized, "accessing uninitialized static shared memory object");
    }
    _CCCL_ASSERT(__state_ != __state_type::__destroyed, "accessing destroyed static shared memory object");
    return *shared_memory_ptr<_Tp>{__base_type::get().__get_smem_addr()};
  }

  //! @brief Gets a pointer to the object stored in the static shared memory.
  //!
  //! @returns A pointer to the object stored in the static shared memory.
  [[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<element_type> operator&() const noexcept
  {
    return __ptr();
  }
};

template <class _Tp, ::cuda::std::size_t _Align = alignof(_Tp)>
class [[nodiscard]] static_shared;

//! @brief A RAII wrapper for an object living in static shared memory.
//!
//! @tparam _Tp The type of the object.
//! @tparam _Align The alignment of the object.
template <class _Tp, ::cuda::std::size_t _Align>
class [[nodiscard]] static_shared : __static_shared_base<_Tp, _Align>
{
  static_assert(!::cuda::std::is_void_v<_Tp>, "_Tp must not be void");

  using __base_type = __static_shared_base<_Tp, _Align>;
  using __base_type::__ptr;
  using __base_type::__state_;
  using typename __base_type::__state_type;

  //! @brief Implements the destruction of the object. No assertions are performed.
  //!
  //! @param __chosen_thread The thread that will perform the destruction.
  _CCCL_DEVICE_API void __destroy_by_impl(::uint3 __chosen_thread) noexcept(::cuda::std::is_nothrow_destructible_v<_Tp>)
  {
    if (__state_ == __state_type::__constructed)
    {
      if (__chosen_thread.x == threadIdx.x && __chosen_thread.y == threadIdx.y && __chosen_thread.z == threadIdx.z)
      {
        ::cuda::std::__destroy_at(__ptr().get());
      }
      __state_ = __state_type::__destroyed;
    }
  }

public:
  //! @brief The default 3D index of the thread used to construct/destroy the object.
  static constexpr ::uint3 default_thread_index{0, 0, 0};

  using __base_type::alignment;
  using __base_type::size;
  using typename __base_type::element_type;
  using typename __base_type::value_type;

  //! @brief Allocates the static shared memory without constructing the object. The object is expected to be
  //!        constructed later by calling construct(...)/construct_by(...) methods.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static_shared(cuda::no_init_t) noexcept {}

  //! @brief Allocates the static shared memory and constructs the object.
  //!
  //! @param __args The arguments to forward to the constructor of the object.
  _CCCL_TEMPLATE(class _Tp2 = _Tp, class... _Args)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Tp2, _Args...>)
  _CCCL_DEVICE_API _CCCL_FORCEINLINE
  static_shared(_Args&&... __args) noexcept(::cuda::std::is_nothrow_constructible_v<_Tp, _Args...>)
  {
    construct(::cuda::std::forward<_Args>(__args)...);
  }

  static_shared(const static_shared&) = delete;

  static_shared(static_shared&&) = delete;

  //! @brief Destroys the stored object.
  _CCCL_DEVICE_API ~static_shared() noexcept(::cuda::std::is_nothrow_destructible_v<_Tp>)
  {
    __destroy_by_impl(default_thread_index);
  }

  static_shared& operator=(const static_shared&) = delete;

  static_shared& operator=(static_shared&&) = delete;

  //! @brief Constructs the stored object in-place by calling its constructor with the given arguments by the
  //!        \c default_thread_index thread.
  //!
  //! @param __args The arguments to forward to the constructor of the stored object.
  _CCCL_TEMPLATE(class _Tp2 = _Tp, class... _Args)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Tp2, _Args...>)
  _CCCL_DEVICE_API void construct(_Args&&... __args) noexcept(::cuda::std::is_nothrow_constructible_v<_Tp, _Args...>)
  {
    construct_by(default_thread_index, ::cuda::std::forward<_Args>(__args)...);
  }

  //! @brief Constructs the stored object in-place by calling its constructor with the given arguments by the thread at
  //!        the given index.
  //!
  //! @param __chosen_thread The thread index of the thread that will construct the object.
  //! @param __args The arguments to forward to the constructor of the stored object.
  _CCCL_TEMPLATE(class _Tp2 = _Tp, class... _Args)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Tp2, _Args...>)
  _CCCL_DEVICE_API void construct_by(::uint3 __chosen_thread,
                                     _Args&&... __args) noexcept(::cuda::std::is_nothrow_constructible_v<_Tp, _Args...>)
  {
    _CCCL_ASSERT(__state_ != __state_type::__constructed, "static shared memory object is already constructed");
    _CCCL_ASSERT(__state_ != __state_type::__destroyed, "static shared memory object cannot be reconstructed");
    if (__chosen_thread.x == threadIdx.x && __chosen_thread.y == threadIdx.y && __chosen_thread.z == threadIdx.z)
    {
      ::cuda::std::__construct_at(__ptr().get(), ::cuda::std::forward<_Args>(__args)...);
    }
    __state_ = __state_type::__constructed;
  }

  //! @brief Destroys the object stored in the static shared memory by the \c default_thread_index thread and
  //!        invalidates this instance.
  _CCCL_DEVICE_API void destroy() noexcept(::cuda::std::is_nothrow_destructible_v<_Tp>)
  {
    destroy_by(default_thread_index);
  }

  //! @brief Destroys the object stored in the static shared memory by the \c __chosen_thread thread and
  //!        invalidates this instance.
  //!
  //! @param __chosen_thread The thread that destroys the object.
  _CCCL_DEVICE_API void destroy_by(::uint3 __chosen_thread) noexcept(::cuda::std::is_nothrow_destructible_v<_Tp>)
  {
    _CCCL_ASSERT(__state_ != __state_type::__destroyed, "destroying already destroyed static shared memory object");
    __destroy_by_impl(__chosen_thread);
  }

  using __base_type::get;
  using __base_type::operator&;

  //! @brief Casts the static shared memory object to a reference to the object stored in the static shared memory.
  _CCCL_DEVICE_API operator _Tp&() const noexcept
  {
    return get();
  }
};

template <class _Tp, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_CONSTEVAL ::cuda::std::size_t __cccl_mdarray_nelem_impl(::cuda::std::index_sequence<_Is...>) noexcept
{
  return (::cuda::std::size_t{0} + ... + ::cuda::std::extent_v<_Tp, _Is>);
}

template <class _Tp>
[[nodiscard]] _CCCL_CONSTEVAL ::cuda::std::size_t __cccl_mdarray_nelem() noexcept
{
  return __cccl_mdarray_nelem_impl<_Tp>(::cuda::std::make_index_sequence<::cuda::std::rank_v<_Tp>>{});
}

template <class _Tp, ::cuda::std::size_t _Size, ::cuda::std::size_t _Align>
class static_shared<_Tp[_Size], _Align> : __static_shared_base<_Tp[_Size], _Align>
{
  static_assert(!::cuda::std::is_array_v<_Tp> || ::cuda::std::__is_fully_bounded_array_v<_Tp>,
                "_Tp must be a fully-bounded array");
  static_assert(::cuda::std::is_default_constructible_v<_Tp>, "_Tp must be default constructible");

  static constexpr auto __nelems = static_cast<unsigned>(__cccl_mdarray_nelem<_Tp[_Size]>());

  using __full_type = _Tp[_Size];
  using __base_type = __static_shared_base<_Tp[_Size], _Align>;
  using typename __base_type::__state_type;

  using __base_type::__ptr;
  using __base_type::__state_;

  //! @brief Implements the destruction of the objects. No assertions are performed.
  _CCCL_DEVICE_API void __destroy_impl() noexcept(::cuda::std::is_nothrow_destructible_v<element_type>)
  {
    if (__state_ == __state_type::__constructed)
    {
      auto __smem_addr = static_cast<unsigned>(operator&().__get_smem_addr());
      unsigned __count;
      for (unsigned __i = ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x; __i < __nelems;
           __i += blockDim.z * blockDim.y * blockDim.x)
      {
        void* __ptr = ::__cvta_shared_to_generic(__smem_addr + __i * sizeof(element_type));
        ::cuda::std::__destroy_at(static_cast<element_type*>(__ptr));
      }
      __state_ = __state_type::__destroyed;
    }
  }

public:
  using __base_type::alignment;
  using __base_type::size;
  using typename __base_type::element_type;
  using typename __base_type::value_type;
  using index_type = unsigned;

  //! @brief Allocates the static shared memory and default constructs the objects.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE
  static_shared() noexcept(::cuda::std::is_nothrow_default_constructible_v<element_type>)
  {
    construct();
  }

  //! @brief Allocates the static shared memory without constructing the objects. The objects are expected to be
  //!        constructed later by calling construct(...) methods.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static_shared(cuda::no_init_t) noexcept {}

  static_shared(const static_shared&) = delete;

  static_shared(static_shared&&) = delete;

  //! @brief Destroys the stored objects.
  _CCCL_DEVICE_API ~static_shared() noexcept(::cuda::std::is_nothrow_destructible_v<element_type>)
  {
    __destroy_impl();
  }

  static_shared& operator=(const static_shared&) = delete;

  static_shared& operator=(static_shared&&) = delete;

  //! @brief Constructs the stored objects in-place by calling the default constructor.
  _CCCL_DEVICE_API void construct() noexcept(::cuda::std::is_nothrow_default_constructible_v<element_type>)
  {
    _CCCL_ASSERT(__state_ != __state_type::__constructed, "static shared memory objects are already constructed");
    _CCCL_ASSERT(__state_ != __state_type::__destroyed, "static shared memory objects cannot be reconstructed");

    auto __smem_addr = static_cast<unsigned>(operator&().__get_smem_addr());
    for (unsigned __i = ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x; __i < __nelems;
         __i += blockDim.z * blockDim.y * blockDim.x)
    {
      void* __ptr = ::__cvta_shared_to_generic(__smem_addr + __i * sizeof(element_type));
      ::cuda::std::__construct_at(static_cast<element_type*>(__ptr));
    }
    __state_ = __state_type::__constructed;
  }

  //! @brief Destroys the objects stored in the static shared memory.
  _CCCL_DEVICE_API void destroy() noexcept(::cuda::std::is_nothrow_destructible_v<element_type>)
  {
    _CCCL_ASSERT(__state_ != __state_type::__destroyed, "destroying already destroyed static shared memory objects");
    __destroy_impl();
  }

  using __base_type::get;
  using __base_type::operator&;

  //! @brief Casts the static shared memory object to a reference to the objects stored in the static shared memory.
  _CCCL_DEVICE_API operator __full_type&() const noexcept
  {
    return get();
  }
};

template <class _Tp, ::cuda::std::size_t _Align>
class static_shared<_Tp[], _Align>
{
  static_assert(::cuda::std::__always_false_v<_Tp>, "static_shared<T[]> is not allowed");
};
} // namespace cuda::experimental

#endif // _CUDA_EXPERIMENTAL___MEMORY_STATIC_SHARED_H
