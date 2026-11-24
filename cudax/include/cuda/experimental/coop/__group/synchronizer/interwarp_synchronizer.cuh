//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/static_index_sequence.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
using __interwarp_barrier_id _CCCL_NODEBUG = ::cuda::std::uint32_t;

[[nodiscard]] _CCCL_API constexpr ::cuda::std::size_t __interwarp_thread_count(::cuda::std::size_t __warp_count) noexcept
{
  return __warp_count == ::cuda::std::dynamic_extent ? ::cuda::std::dynamic_extent : __warp_count * 32;
}

// Store only operands that the mapping and barrier ID sequence leave dynamic.
template <::cuda::std::size_t _ThreadCount, ::cuda::std::size_t _BarrierId>
using __interwarp_operands _CCCL_NODEBUG = ::cuda::std::__mdspan_detail::
  __maybe_static_array<::cuda::std::uint32_t, ::cuda::std::size_t, ::cuda::std::dynamic_extent, _BarrierId, _ThreadCount>;

template <::cuda::std::size_t _ThreadCount,
          ::cuda::std::size_t _BarrierId = ::cuda::std::dynamic_extent,
          class _ArrivalToken            = void>
class __interwarp_synchronizer_instance : private __interwarp_operands<_ThreadCount, _BarrierId>
{
  using _Operands _CCCL_NODEBUG = __interwarp_operands<_ThreadCount, _BarrierId>;

protected:
  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t __thread_count() const noexcept
  {
    return _Operands::__value(1);
  }

public:
  using arrival_token _CCCL_NODEBUG =
    ::cuda::std::conditional_t<::cuda::std::is_void_v<_ArrivalToken>,
                               __dummy_arrival_token<__interwarp_synchronizer_instance>,
                               _ArrivalToken>;

  _CCCL_HIDE_FROM_ABI __interwarp_synchronizer_instance() = default;

  _CCCL_DEVICE_API constexpr __interwarp_synchronizer_instance(
    __interwarp_barrier_id __barrier_id, ::cuda::std::uint32_t __thread_count) noexcept
      : _Operands{__barrier_id, __thread_count}
  {
    _CCCL_ASSERT(__barrier_id < 16u, "Invalid native warp barrier id");
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy&) const
  {
    ::__barrier_sync_count(_Operands::__value(0), __thread_count());
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy& __hier) const
  {
    // The .aligned version of the barrier instruction requires whole CTA to execute the same instruction, which is not
    // usable with groups. That means that we need to always use the unaligned version.
    do_sync(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive(const _Hierarchy&) const
  {
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive_aligned(const _Hierarchy&) const
  {
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait(const _Hierarchy& __hier, [[maybe_unused]] arrival_token&& __tok) const
  {
    // Construct a temporary object, so we set __tok to moved-from state.
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__is_valid(),
                 "invalid arrival token passed to do_wait function");
    do_sync(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy& __hier, [[maybe_unused]] arrival_token&& __tok) const
  {
    // Construct a temporary object, so we set __tok to moved-from state.
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__is_valid(),
                 "invalid arrival token passed to do_wait_aligned function");
    do_sync_aligned(__hier);
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}
};

template <class _MappingResult>
using __interwarp_group_instance _CCCL_NODEBUG =
  __interwarp_synchronizer_instance<::cuda::experimental::coop::__interwarp_thread_count(
    _MappingResult::static_unit_count())>;

template <class _MappingResult, class _BarrierIds>
class __interwarp_ranked_instance : public __interwarp_group_instance<_MappingResult>
{
  static_assert(_MappingResult::static_group_count() == ::cuda::std::dynamic_extent
                  || _MappingResult::static_group_count() <= _BarrierIds::size(),
                "insufficient amount of warp barrier ids were passed");
  using _Base _CCCL_NODEBUG = __interwarp_group_instance<_MappingResult>;

  template <::cuda::std::size_t _GroupRank>
  using __rank_instance _CCCL_NODEBUG = __interwarp_synchronizer_instance<
    ::cuda::experimental::coop::__interwarp_thread_count(_MappingResult::static_unit_count(_GroupRank)),
    ::cuda::std::get<_GroupRank>(_BarrierIds{}),
    typename _Base::arrival_token>;

public:
  using _Base::_Base;

  template <::cuda::std::size_t _GroupRank>
  [[nodiscard]] _CCCL_DEVICE_API constexpr __rank_instance<_GroupRank> specialize() const noexcept
  {
    if constexpr (_MappingResult::static_unit_count(_GroupRank) == ::cuda::std::dynamic_extent)
    {
      return {::cuda::std::get<_GroupRank>(_BarrierIds{}), this->__thread_count()};
    }
    else
    {
      return {};
    }
  }
};

template <class _Instance, class _Range, class _Unit, class _ParentGroup, class _MappingResult>
[[nodiscard]] _CCCL_DEVICE_API _Instance __make_interwarp_instance(
  _Range& __barrier_ids, const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result)
{
  using _Bound = typename _ParentGroup::boundary_level;

  constexpr auto __static_group_count      = _MappingResult::static_group_count();
  constexpr auto __static_barrier_id_count = ::cuda::std::ranges::__static_size_v<_Range>;
  if constexpr (__static_group_count != ::cuda::std::dynamic_extent
                && __static_barrier_id_count != ::cuda::std::dynamic_extent)
  {
    static_assert(__static_group_count <= __static_barrier_id_count,
                  "insufficient amount of warp barrier ids were passed");
  }

  static_assert(::cuda::std::is_same_v<_Unit, warp_level> && ::cuda::std::is_same_v<_Bound, block_level>,
                "interwarp_synchronizer can be only used to group warps within a single block");

  // Default construct the synchronizer instance for units that don't have a valid mapping result.
  if (!__mapping_result.is_valid())
  {
    return _Instance{};
  }

  if constexpr (::cuda::std::ranges::sized_range<_Range>)
  {
    _CCCL_ASSERT(__mapping_result.group_count() <= ::cuda::std::ranges::size(__barrier_ids),
                 "insufficient amount of warp barrier ids were passed");

    auto __it = ::cuda::std::ranges::begin(__barrier_ids);
    ::cuda::std::ranges::advance(__it, __mapping_result.group_rank());
    return _Instance{static_cast<__interwarp_barrier_id>(*__it), __mapping_result.unit_count() * 32};
  }
  else
  {
    ::cuda::std::uint32_t __i = 0;
    auto __it                 = ::cuda::std::ranges::begin(__barrier_ids);
    do
    {
      if (__i > 0)
      {
        ++__it;
      }
      _CCCL_ASSERT(__it != ::cuda::std::ranges::end(__barrier_ids),
                   "insufficient amount of warp barrier ids were passed");
    } while (__i++ < __mapping_result.group_rank());
    return _Instance{static_cast<__interwarp_barrier_id>(*__it), __mapping_result.unit_count() * 32};
  }
}

template <class _RangeView>
class interwarp_synchronizer
{
  static_assert(::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_RangeView>, __interwarp_barrier_id>,
                "_RangeView's value_type must be convertible to the interwarp barrier id");

  _RangeView __barrier_ids_;

public:
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_RangeView, ::cuda::std::ranges::views::all_t<_Range>>)
  _CCCL_DEVICE_API interwarp_synchronizer(_Range&& __barrier_ids) // NOLINT(bugprone-forwarding-reference-overload)
      : __barrier_ids_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__barrier_ids))}
  {}

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _MappingResult)
  _CCCL_REQUIRES((!::cuda::std::ranges::range<const _RangeView>) )
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result)
  {
    return ::cuda::experimental::coop::__make_interwarp_instance<__interwarp_group_instance<_MappingResult>>(
      __barrier_ids_, __unit, __parent, __mapping_result);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _MappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result) const
  {
    return ::cuda::experimental::coop::__make_interwarp_instance<__interwarp_group_instance<_MappingResult>>(
      __barrier_ids_, __unit, __parent, __mapping_result);
  }
};

template <::cuda::std::size_t... _Ids>
class interwarp_synchronizer<::cuda::std::index_sequence<_Ids...>>
{
  using _BarrierIds _CCCL_NODEBUG = ::cuda::std::index_sequence<_Ids...>;

  [[nodiscard]] _CCCL_API static constexpr bool __valid_ids() noexcept
  {
    bool __used[16]{};
    for (const auto __id : ::cuda::std::array<::cuda::std::size_t, sizeof...(_Ids)>{_Ids...})
    {
      if (__id >= 16 || __used[__id])
      {
        return false;
      }
      __used[__id] = true;
    }
    return true;
  }

  static_assert(__valid_ids(), "interwarp barrier IDs must be distinct and in the range [0, 16)");

public:
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<__static_index_sequence_t<_Range>, _BarrierIds>)
  _CCCL_DEVICE_API explicit constexpr interwarp_synchronizer(const _Range&) noexcept {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result) const
  {
    constexpr ::cuda::std::array<__interwarp_barrier_id, sizeof...(_Ids)> __ids{_Ids...};
    using _Instance = __interwarp_ranked_instance<_MappingResult, _BarrierIds>;
    const auto __instance =
      ::cuda::experimental::coop::__make_interwarp_instance<_Instance>(__ids, __unit, __parent, __mapping_result);
    if constexpr (_MappingResult::static_group_rank() != ::cuda::std::dynamic_extent)
    {
      return __instance.template specialize<_MappingResult::static_group_rank()>();
    }
    else
    {
      return __instance;
    }
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, __interwarp_barrier_id>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(_Range&&)
  -> interwarp_synchronizer<::cuda::std::ranges::views::all_t<_Range>>;

template <class _Range>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(const _Range&)
  -> interwarp_synchronizer<__static_index_sequence_t<_Range>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
