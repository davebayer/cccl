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

#include <cuda/__numeric/narrow.h>
#include <cuda/hierarchy>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/constant_wrapper.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/static_index_sequence.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
using __interwarp_barrier_id _CCCL_NODEBUG = ::cuda::std::uint32_t;

template <::cuda::std::size_t... _Ids>
[[nodiscard]] _CCCL_DEVICE_API constexpr __interwarp_barrier_id
__interwarp_barrier_id_at(::cuda::std::index_sequence<_Ids...>, ::cuda::std::size_t __rank) noexcept
{
  // All 16 four-bit barrier IDs fit in one word, even when the rank is only known at runtime.
  ::cuda::std::uint64_t __packed    = 0;
  [[maybe_unused]] unsigned __shift = 0;
  ((__packed |= ::cuda::std::uint64_t{_Ids} << __shift, __shift += 4u), ...);
  return static_cast<__interwarp_barrier_id>((__packed >> (4u * __rank)) & 0xfu);
}

template <class _MappingResult, class _BarrierIds>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::size_t __interwarp_static_barrier_id() noexcept
{
  if constexpr (::cuda::std::is_void_v<_BarrierIds>
                || _MappingResult::static_group_rank() == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::dynamic_extent;
  }
  else
  {
    return ::cuda::experimental::coop::__interwarp_barrier_id_at(_BarrierIds{}, _MappingResult::static_group_rank());
  }
}

template <class _MappingResult>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::size_t
__interwarp_thread_count(unsigned __threads_per_unit) noexcept
{
  constexpr auto __count = _MappingResult::static_unit_count();
  return __count == ::cuda::std::dynamic_extent ? ::cuda::std::dynamic_extent : __count * __threads_per_unit;
}

template <class _MappingResult>
_CCCL_CONCEPT __interwarp_thread_mapping = _CCCL_REQUIRES_EXPR((_MappingResult))(
  requires(::cuda::std::is_same_v<typename _MappingResult::unit_level, thread_level>));

template <::cuda::std::size_t _BarrierId, ::cuda::std::size_t _ThreadCount>
struct __interwarp_barrier_operands
{
  static constexpr __interwarp_barrier_id __barrier_id_{_BarrierId};
  static constexpr ::cuda::std::uint32_t __thread_count_{_ThreadCount};
};

template <::cuda::std::size_t _ThreadCount>
struct __interwarp_barrier_operands<::cuda::std::dynamic_extent, _ThreadCount>
{
  __interwarp_barrier_id __barrier_id_{};
  static constexpr ::cuda::std::uint32_t __thread_count_{_ThreadCount};
};

template <::cuda::std::size_t _BarrierId>
struct __interwarp_barrier_operands<_BarrierId, ::cuda::std::dynamic_extent>
{
  static constexpr __interwarp_barrier_id __barrier_id_{_BarrierId};
  ::cuda::std::uint32_t __thread_count_{};
};

template <>
struct __interwarp_barrier_operands<::cuda::std::dynamic_extent, ::cuda::std::dynamic_extent>
{
  __interwarp_barrier_id __barrier_id_{};
  ::cuda::std::uint32_t __thread_count_{};
};

template <::cuda::std::size_t _ThreadCount,
          class _BarrierIds              = void,
          ::cuda::std::size_t _BarrierId = ::cuda::std::dynamic_extent>
class __interwarp_synchronizer_instance : private __interwarp_barrier_operands<_BarrierId, _ThreadCount>
{
public:
  // Specializing a view keeps the same barrier and its outstanding arrival tokens.
  using arrival_token _CCCL_NODEBUG =
    __dummy_arrival_token<__interwarp_synchronizer_instance<::cuda::std::dynamic_extent>>;

  // An invalid mapping has no barrier to initialize.
  _CCCL_HIDE_FROM_ABI __interwarp_synchronizer_instance() = default;

  _CCCL_DEVICE_API
  __interwarp_synchronizer_instance(__interwarp_barrier_id __barrier_id, ::cuda::std::uint32_t __thread_count)
  {
    if constexpr (_BarrierId == ::cuda::std::dynamic_extent)
    {
      this->__barrier_id_ = __barrier_id;
    }
    if constexpr (_ThreadCount == ::cuda::std::dynamic_extent)
    {
      this->__thread_count_ = __thread_count;
    }
    _CCCL_ASSERT(__barrier_id < 16u && __barrier_id == this->__barrier_id_, "Invalid native warp barrier id");
    _CCCL_ASSERT(__thread_count == this->__thread_count_, "Invalid interwarp barrier thread count");
  }

  template <class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto specialize(const _MappingResult&) const
  {
    using ::cuda::std::dynamic_extent;
    // A thread view already reports its count in threads. Native interwarp mappings count warps.
    constexpr auto __count =
      _ThreadCount != dynamic_extent
        ? _ThreadCount
        : ::cuda::experimental::coop::__interwarp_thread_count<_MappingResult>(
            __interwarp_thread_mapping<_MappingResult> ? 1 : 32);
    constexpr auto __id =
      _BarrierId != dynamic_extent
        ? _BarrierId
        : ::cuda::experimental::coop::__interwarp_static_barrier_id<_MappingResult, _BarrierIds>();
    return __interwarp_synchronizer_instance<__count, _BarrierIds, __id>{this->__barrier_id_, this->__thread_count_};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy&) const
  {
    ::__barrier_sync_count(this->__barrier_id_, this->__thread_count_);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy& __hier) const
  {
    // The .aligned instruction requires the whole CTA to execute the same barrier.
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

template <class _MappingResult, class _BarrierIds = void>
using __mapped_interwarp_synchronizer_instance _CCCL_NODEBUG = __interwarp_synchronizer_instance<
  ::cuda::experimental::coop::__interwarp_thread_count<_MappingResult>(32),
  _BarrierIds,
  ::cuda::experimental::coop::__interwarp_static_barrier_id<_MappingResult, _BarrierIds>()>;

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
[[nodiscard]] _CCCL_DEVICE_API __interwarp_barrier_id
__interwarp_barrier_id_at(_Range& __barrier_ids, ::cuda::std::uint32_t __rank)
{
  auto __it = ::cuda::std::ranges::begin(__barrier_ids);
  if constexpr (::cuda::std::ranges::sized_range<_Range>)
  {
    ::cuda::std::ranges::advance(__it, __rank);
  }
  else
  {
    for (; __rank > 0; --__rank, ++__it)
    {
      _CCCL_ASSERT(__it != ::cuda::std::ranges::end(__barrier_ids),
                   "insufficient amount of warp barrier ids were passed");
    }
    _CCCL_ASSERT(__it != ::cuda::std::ranges::end(__barrier_ids),
                 "insufficient amount of warp barrier ids were passed");
  }
  return static_cast<__interwarp_barrier_id>(*__it);
}

template <class _Ids = void, class _Range, class _Unit, class _ParentGroup, class _MappingResult>
[[nodiscard]] _CCCL_DEVICE_API auto __make_interwarp_instance(
  _Range&& __barrier_ids, const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result)
{
  using _Bound = typename _ParentGroup::boundary_level;
  static_assert(::cuda::std::is_same_v<_Unit, warp_level> && ::cuda::std::is_same_v<_Bound, block_level>,
                "interwarp_synchronizer can be only used to group warps within a single block");

  constexpr auto __static_group_count      = _MappingResult::static_group_count();
  constexpr auto __static_barrier_id_count = ::cuda::std::ranges::__static_size_v<_Range>;
  static_assert(
    __static_group_count == ::cuda::std::dynamic_extent || __static_group_count <= __static_barrier_id_count,
    "insufficient amount of warp barrier ids were passed");

  using _Instance = __mapped_interwarp_synchronizer_instance<_MappingResult, _Ids>;
  if (!__mapping_result.is_valid())
  {
    return _Instance{};
  }

  if constexpr (!::cuda::std::ranges::range<_Range> || ::cuda::std::ranges::sized_range<_Range>)
  {
    _CCCL_ASSERT(__mapping_result.group_count() <= ::cuda::std::ranges::size(__barrier_ids),
                 "insufficient amount of warp barrier ids were passed");
  }
  const auto __id = ::cuda::experimental::coop::__interwarp_barrier_id_at(__barrier_ids, __mapping_result.group_rank());
  return _Instance{__id, __mapping_result.unit_count() * 32};
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
    return ::cuda::experimental::coop::__make_interwarp_instance(__barrier_ids_, __unit, __parent, __mapping_result);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _MappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result) const
  {
    return ::cuda::experimental::coop::__make_interwarp_instance(__barrier_ids_, __unit, __parent, __mapping_result);
  }
};

template <::cuda::std::size_t... _BarrierIds>
class interwarp_synchronizer<::cuda::std::index_sequence<_BarrierIds...>>
{
  using _Ids _CCCL_NODEBUG = ::cuda::std::index_sequence<_BarrierIds...>;
  static_assert(((_BarrierIds < 16) && ...)
                  && ::cuda::std::popcount((0u | ... | (1u << _BarrierIds))) == sizeof...(_BarrierIds),
                "native warp barrier ids must be distinct and less than 16");

public:
  _CCCL_TEMPLATE(class _Int, _Int... _Values)
  _CCCL_REQUIRES(
    ::cuda::std::is_same_v<_Ids, ::cuda::std::index_sequence<::cuda::narrow<::cuda::std::size_t>(_Values)...>>)
  _CCCL_DEVICE_API constexpr interwarp_synchronizer(const ::cuda::std::integer_sequence<_Int, _Values...>&) {}

  _CCCL_TEMPLATE(auto _Range, class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Ids, __static_index_sequence_t<::cuda::std::constant_wrapper<_Range, _Tp>>>)
  _CCCL_DEVICE_API constexpr interwarp_synchronizer(const ::cuda::std::constant_wrapper<_Range, _Tp>&) {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result) const
  {
    return ::cuda::experimental::coop::__make_interwarp_instance<_Ids>(_Ids{}, __unit, __parent, __mapping_result);
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, __interwarp_barrier_id>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(_Range&&)
  -> interwarp_synchronizer<::cuda::std::ranges::views::all_t<_Range>>;

template <class _Int, _Int... _BarrierIds>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(const ::cuda::std::integer_sequence<_Int, _BarrierIds...>&)
  -> interwarp_synchronizer<::cuda::std::index_sequence<::cuda::narrow<::cuda::std::size_t>(_BarrierIds)...>>;

_CCCL_TEMPLATE(auto _Range, class _Tp)
_CCCL_REQUIRES(__static_index_range<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(const ::cuda::std::constant_wrapper<_Range, _Tp>&)
  -> interwarp_synchronizer<__static_index_sequence_t<::cuda::std::constant_wrapper<_Range, _Tp>>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
