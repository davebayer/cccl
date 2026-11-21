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

#include <cuda/std/__bit/integral.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/tags.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
// The view passed to the callable when dispatching _Group, which is a group or a view of one.
template <class _Group>
using __dispatch_view_t =
  group_view<typename _Group::unit_type, __viewed_group_t<_Group>, typename _Group::__mapping_result_type>;

// A dispatch calls the callable with one of several differently specialized views, its leaves. The leaf is selected
// by comparing a key, which is computed once from the group rank, against a threshold per leaf. The leaves are
// described by a class with the following static members:
//
//   __size               Number of leaves.
//   __key(group_rank)    Key of the group with the given rank.
//   __threshold(leaf)    Smallest key that selects the given leaf or a later one.
//   __view<leaf>(group)  View passed to the callable by the given leaf.

// Dispatching by group rank has one leaf per group, selected by the group rank itself.
template <class _MappingResult>
struct __dispatch_by_group_rank
{
  static constexpr ::cuda::std::size_t __size = _MappingResult::static_group_count();

  template <::cuda::std::size_t _Leaf>
  using __leaf_mapping_result_t = ::cuda::experimental::coop::__mapping_result<
    __per_group_count<__size, _Leaf, _MappingResult::static_unit_count(_Leaf)>,
    _MappingResult::is_always_exhaustive(),
    _MappingResult::is_always_contiguous()>;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::uint32_t
  __key(::cuda::std::uint32_t __group_rank) noexcept
  {
    return __group_rank;
  }

  [[nodiscard]] _CCCL_DEVICE_API static _CCCL_CONSTEVAL ::cuda::std::uint32_t
  __threshold(::cuda::std::size_t __leaf) noexcept
  {
    return static_cast<::cuda::std::uint32_t>(__leaf);
  }

  template <::cuda::std::size_t _Leaf, class _Group>
  [[nodiscard]] _CCCL_DEVICE_API static __dispatch_view_t<_Group, __leaf_mapping_result_t<_Leaf>>
  __view(const _Group& __group)
  {
    const auto& __mapping_result = __group.__mapping_result();
    return __dispatch_view_t<_Group, __leaf_mapping_result_t<_Leaf>>{
      __group,
      __leaf_mapping_result_t<_Leaf>{
        __mapping_result.unit_count(), __mapping_result.unit_rank(), __mapping_result.lane_mask()}};
  }
};

// The leaves of a dispatch by unit count for a mapping with _GroupCount groups: one per distinct static unit count.
template <::cuda::std::size_t _GroupCount>
struct __unit_count_leaves
{
  // Number of leaves.
  ::cuda::std::size_t __size_ = 0;
  // Unit count of each leaf. The leaves are ordered by the rank of their first group.
  ::cuda::std::size_t __unit_counts_[_GroupCount]{};
  // Rank of the first group of each leaf.
  ::cuda::std::uint32_t __first_group_ranks_[_GroupCount]{};
  // Leaf of each group.
  ::cuda::std::uint32_t __group_leaves_[_GroupCount]{};
  // Whether the groups of every leaf have consecutive ranks.
  bool __is_contiguous_ = true;
};

template <class _MappingResult, ::cuda::std::size_t... _GroupRanks>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_CONSTEVAL __unit_count_leaves<sizeof...(_GroupRanks)>
  __make_unit_count_leaves(::cuda::std::index_sequence<_GroupRanks...>) noexcept
{
  constexpr ::cuda::std::size_t __unit_counts[]{_MappingResult::static_unit_count(_GroupRanks)...};

  __unit_count_leaves<sizeof...(_GroupRanks)> __leaves{};
  for (::cuda::std::uint32_t __group_rank = 0; __group_rank < sizeof...(_GroupRanks); ++__group_rank)
  {
    ::cuda::std::uint32_t __leaf = 0;
    while (__leaf < __leaves.__size_ && __leaves.__unit_counts_[__leaf] != __unit_counts[__group_rank])
    {
      ++__leaf;
    }

    if (__leaf == __leaves.__size_)
    {
      __leaves.__unit_counts_[__leaf]       = __unit_counts[__group_rank];
      __leaves.__first_group_ranks_[__leaf] = __group_rank;
      ++__leaves.__size_;
    }
    else if (__leaf + 1 != __leaves.__size_)
    {
      // The unit count of an earlier leaf reappears after groups of another leaf.
      __leaves.__is_contiguous_ = false;
    }
    __leaves.__group_leaves_[__group_rank] = __leaf;
  }
  return __leaves;
}

// Dispatching by unit count has one leaf per distinct static unit count. The leaf is selected from the group rank
// rather than from the unit count, since the unit count of a group with per-group static unit counts is only known at
// runtime through a lookup that grows with the number of groups.
template <class _MappingResult>
struct __dispatch_by_unit_count
{
  static constexpr ::cuda::std::size_t __group_count = _MappingResult::static_group_count();

  static constexpr __unit_count_leaves<__group_count> __leaves =
    ::cuda::experimental::coop::__make_unit_count_leaves<_MappingResult>(
      ::cuda::std::make_index_sequence<__group_count>{});

  static constexpr ::cuda::std::size_t __size = __leaves.__size_;

  template <::cuda::std::size_t _Leaf>
  using __leaf_mapping_result_t = ::cuda::experimental::coop::__mapping_result<
    __uniform_static_unit_count<__group_count, __leaves.__unit_counts_[_Leaf]>,
    _MappingResult::is_always_exhaustive(),
    _MappingResult::is_always_contiguous()>;

  // If the groups of a leaf don't have consecutive ranks, the leaf of a group is looked up in a table which packs the
  // leaves of all groups into 32-bit words. Using a power of two number of bits per leaf lets shifts and masks locate
  // the leaf of a group.
  static constexpr ::cuda::std::uint32_t __leaf_bits =
    ::cuda::std::bit_ceil(static_cast<::cuda::std::uint32_t>(::cuda::std::bit_width(__size - 1)));
  static constexpr ::cuda::std::uint32_t __leaves_per_word = 32 / __leaf_bits;
  static constexpr ::cuda::std::size_t __word_count = (__group_count + __leaves_per_word - 1) / __leaves_per_word;

  [[nodiscard]] _CCCL_DEVICE_API static _CCCL_CONSTEVAL ::cuda::std::uint32_t
  __packed_leaves(::cuda::std::size_t __word) noexcept
  {
    ::cuda::std::uint32_t __result = 0;
    for (::cuda::std::uint32_t __i = 0; __i < __leaves_per_word; ++__i)
    {
      const auto __group_rank = __word * __leaves_per_word + __i;
      if (__group_rank < __group_count)
      {
        __result |= __leaves.__group_leaves_[__group_rank] << (__i * __leaf_bits);
      }
    }
    return __result;
  }

  template <::cuda::std::size_t... _Words>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::uint32_t
  __lookup_leaf(::cuda::std::uint32_t __group_rank, ::cuda::std::index_sequence<_Words...>) noexcept
  {
    constexpr ::cuda::std::uint32_t __words[]{__packed_leaves(_Words)...};

    // Select the word with conditional moves rather than by indexing the array to keep the lookup in registers.
    auto __word = __words[0];
    ((__word = (__group_rank / __leaves_per_word == _Words) ? __words[_Words] : __word), ...);
    return (__word >> (__group_rank % __leaves_per_word * __leaf_bits))
         & (~::cuda::std::uint32_t{0} >> (32 - __leaf_bits));
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::uint32_t
  __key(::cuda::std::uint32_t __group_rank) noexcept
  {
    if constexpr (__leaves.__is_contiguous_)
    {
      // Every leaf covers one range of group ranks, which the thresholds separate without a lookup.
      return __group_rank;
    }
    else
    {
      return __lookup_leaf(__group_rank, ::cuda::std::make_index_sequence<__word_count>{});
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static _CCCL_CONSTEVAL ::cuda::std::uint32_t
  __threshold(::cuda::std::size_t __leaf) noexcept
  {
    return __leaves.__is_contiguous_
           ? __leaves.__first_group_ranks_[__leaf]
           : static_cast<::cuda::std::uint32_t>(__leaf);
  }

  template <::cuda::std::size_t _Leaf, class _Group>
  [[nodiscard]] _CCCL_DEVICE_API static __dispatch_view_t<_Group, __leaf_mapping_result_t<_Leaf>>
  __view(const _Group& __group)
  {
    const auto& __mapping_result = __group.__mapping_result();
    return __dispatch_view_t<_Group, __leaf_mapping_result_t<_Leaf>>{
      __group,
      __leaf_mapping_result_t<_Leaf>{
        __mapping_result.group_count(),
        __mapping_result.group_rank(),
        __mapping_result.unit_count(),
        __mapping_result.unit_rank(),
        __mapping_result.lane_mask()}};
  }
};

// Returns the key unchanged, but hides its value from the compiler. Mappings like group_as find the group of a unit
// with a search that has one exit per group, so the group rank is a known constant on each of those exits. If the
// compiler sees that, it resolves the comparisons of the dispatch on every exit of the search rather than once after
// it, which is up to an order of magnitude slower for many groups.
[[nodiscard]] _CCCL_CONST _CCCL_DEVICE_API inline ::cuda::std::uint32_t
__dispatch_hide_key(::cuda::std::uint32_t __key) noexcept
{
  ::cuda::std::uint32_t __hidden_key;
  asm("mov.u32 %0, %1;" : "=r"(__hidden_key) : "r"(__key));
  return __hidden_key;
}

// Calls the callable with the view of the leaf in [_First, _Last) that the key selects. The leaf is found with a
// balanced tree of comparisons, which keeps large callables in a branch tree rather than a jump table. The return type
// is deduced to preserve references and void, and to convert the results of the leaves to a common type the way the
// conditional operator does.
template <::cuda::std::size_t _First, ::cuda::std::size_t _Last, class _Leaves, class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch_tree(const _Group& __group, ::cuda::std::uint32_t __key, _Fn&& __fn)
{
  if constexpr (_Last - _First == 1)
  {
    return ::cuda::std::forward<_Fn>(__fn)(_Leaves::template __view<_First>(__group));
  }
  else
  {
    constexpr auto __mid       = _First + (_Last - _First) / 2;
    constexpr auto __threshold = _Leaves::__threshold(__mid);
    if constexpr (_Last - _First == 2)
    {
      // Calling both leaves here rather than through this function halves the number of its instantiations.
      return (__key < __threshold)
             ? ::cuda::std::forward<_Fn>(__fn)(_Leaves::template __view<_First>(__group))
             : ::cuda::std::forward<_Fn>(__fn)(_Leaves::template __view<__mid>(__group));
    }
    else
    {
      return (__key < __threshold)
             ? ::cuda::experimental::coop::__dispatch_tree<_First, __mid, _Leaves>(
                 __group, __key, ::cuda::std::forward<_Fn>(__fn))
             : ::cuda::experimental::coop::__dispatch_tree<__mid, _Last, _Leaves>(
                 __group, __key, ::cuda::std::forward<_Fn>(__fn));
    }
  }
}

// The return types of the dispatch functions are deduced to return exactly what the callable returns.
template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch(const _Group& __group, by_group_rank_t, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  if constexpr (_MappingResult::static_group_rank() != ::cuda::std::dynamic_extent
                || _MappingResult::static_group_count() == 0
                || _MappingResult::static_group_count() == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::forward<_Fn>(__fn)(__dispatch_view_t<_Group>{__group});
  }
  else
  {
    using _Leaves = __dispatch_by_group_rank<_MappingResult>;
    const auto __key =
      ::cuda::experimental::coop::__dispatch_hide_key(_Leaves::__key(__group.__mapping_result().group_rank()));
    return ::cuda::experimental::coop::__dispatch_tree<0, _Leaves::__size, _Leaves>(
      __group, __key, ::cuda::std::forward<_Fn>(__fn));
  }
}

template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch(const _Group& __group, by_unit_count_t, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  if constexpr (_MappingResult::has_uniform_static_unit_count())
  {
    return ::cuda::std::forward<_Fn>(__fn)(__dispatch_view_t<_Group>{__group});
  }
  else
  {
    using _Leaves = __dispatch_by_unit_count<_MappingResult>;
    const auto __key =
      ::cuda::experimental::coop::__dispatch_hide_key(_Leaves::__key(__group.__mapping_result().group_rank()));
    return ::cuda::experimental::coop::__dispatch_tree<0, _Leaves::__size, _Leaves>(
      __group, __key, ::cuda::std::forward<_Fn>(__fn));
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH
