//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/array>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/tags.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
template <class _Group>
_CCCL_DEVICE_API group_view<typename _Group::unit_type, _Group, typename _Group::__mapping_result_type>
__as_group_view(const _Group& __group)
{
  return group_view<typename _Group::unit_type, _Group, typename _Group::__mapping_result_type>{__group};
}

template <class _Unit, class _Group, class _MappingResult>
_CCCL_DEVICE_API group_view<_Unit, _Group, _MappingResult>
__as_group_view(const group_view<_Unit, _Group, _MappingResult>& __group)
{
  return __group;
}

// Callback result types are deduced to preserve references and void.
template <::cuda::std::size_t _First, ::cuda::std::size_t _Last, class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch_by_group_rank_range(const _Group& __group, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  using _Unit          = typename _Group::unit_type;

  const auto& __mapping_result = __group.__mapping_result();

  if constexpr (_Last - _First > 1)
  {
    // Rank thresholds keep large collective callbacks in a branch tree instead
    // of encouraging jump-table dispatch from a chain of equality comparisons.
    constexpr auto __mid = _First + (_Last - _First) / 2;
    return (__mapping_result.group_rank() < __mid)
           ? ::cuda::experimental::coop::__dispatch_by_group_rank_range<_First, __mid>(
               __group, ::cuda::std::forward<_Fn>(__fn))
           : ::cuda::experimental::coop::__dispatch_by_group_rank_range<__mid, _Last>(
               __group, ::cuda::std::forward<_Fn>(__fn));
  }
  else
  {
    using _PerGroupMappingResult = ::cuda::experimental::coop::__mapping_result<
      __per_group_count<_MappingResult::static_group_count(), _First, _MappingResult::static_unit_count(_First)>,
      _MappingResult::is_always_exhaustive(),
      _MappingResult::is_always_contiguous()>;

    return ::cuda::std::forward<_Fn>(__fn)(group_view<_Unit, __viewed_group_t<_Group>, _PerGroupMappingResult>{
      __group,
      _PerGroupMappingResult{
        __mapping_result.unit_count(), __mapping_result.unit_rank(), __mapping_result.lane_mask()}});
  }
}

template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch(const _Group& __group, by_group_rank_t, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  if constexpr (_MappingResult::static_group_rank() != ::cuda::std::dynamic_extent
                || _MappingResult::static_group_count() == 0
                || _MappingResult::static_group_count() == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::forward<_Fn>(__fn)(::cuda::experimental::coop::__as_group_view(__group));
  }
  else
  {
    return ::cuda::experimental::coop::__dispatch_by_group_rank_range<0, _MappingResult::static_group_count()>(
      __group, ::cuda::std::forward<_Fn>(__fn));
  }
}

template <class _MappingResult>
struct __distinct_unit_counts
{
  ::cuda::std::array<::cuda::std::size_t, _MappingResult::static_group_count()> __values_{};
  ::cuda::std::size_t __size_ = 0;

  _CCCL_DEVICE_API constexpr __distinct_unit_counts() noexcept
  {
    for (::cuda::std::size_t __i = 0; __i < _MappingResult::static_group_count(); ++__i)
    {
      const auto __count        = _MappingResult::static_unit_count(__i);
      ::cuda::std::size_t __pos = 0;
      while (__pos < __size_ && __values_[__pos] < __count)
      {
        ++__pos;
      }
      if (__pos == __size_ || __values_[__pos] != __count)
      {
        for (auto __j = __size_; __j > __pos; --__j)
        {
          __values_[__j] = __values_[__j - 1];
        }
        __values_[__pos] = __count;
        ++__size_;
      }
    }
  }
};

template <::cuda::std::size_t _First, ::cuda::std::size_t _Last, class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch_by_unit_count_range(const _Group& __group, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  using _Unit          = typename _Group::unit_type;
  constexpr __distinct_unit_counts<_MappingResult> __counts{};
  const auto __mapping_result = __group.__mapping_result();

  if constexpr (_Last - _First > 1)
  {
    constexpr auto __mid = _First + (_Last - _First) / 2;
    return (__mapping_result.unit_count() < __counts.__values_[__mid])
           ? ::cuda::experimental::coop::__dispatch_by_unit_count_range<_First, __mid>(
               __group, ::cuda::std::forward<_Fn>(__fn))
           : ::cuda::experimental::coop::__dispatch_by_unit_count_range<__mid, _Last>(
               __group, ::cuda::std::forward<_Fn>(__fn));
  }
  else
  {
    using _CountMappingResult = ::cuda::experimental::coop::__mapping_result<
      __uniform_static_unit_count<_MappingResult::static_group_count(), __counts.__values_[_First]>,
      _MappingResult::is_always_exhaustive(),
      _MappingResult::is_always_contiguous()>;

    return ::cuda::std::forward<_Fn>(__fn)(group_view<_Unit, __viewed_group_t<_Group>, _CountMappingResult>{
      __group,
      _CountMappingResult{__mapping_result.group_count(),
                          __mapping_result.group_rank(),
                          __mapping_result.unit_count(),
                          __mapping_result.unit_rank(),
                          __mapping_result.lane_mask()}});
  }
}

template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch(const _Group& __group, by_unit_count_t, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  if constexpr (_MappingResult::has_uniform_static_unit_count())
  {
    return ::cuda::std::forward<_Fn>(__fn)(::cuda::experimental::coop::__as_group_view(__group));
  }
  else
  {
    constexpr __distinct_unit_counts<_MappingResult> __counts{};
    return ::cuda::experimental::coop::__dispatch_by_unit_count_range<0, __counts.__size_>(
      __group, ::cuda::std::forward<_Fn>(__fn));
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH
