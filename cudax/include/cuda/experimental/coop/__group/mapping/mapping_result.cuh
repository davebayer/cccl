//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_MAPPING_RESULT_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_MAPPING_RESULT_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/lane_mask.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental::coop
{
template <::cuda::std::size_t _StaticGroupCount, ::cuda::std::size_t _StaticUnitCount>
struct __uniform_static_unit_count;

template <::cuda::std::size_t... _StaticUnitCounts>
struct __static_unit_counts;

template <class _CountsTag, bool _IsAlwaysExhaustive, bool _IsAlwaysContiguous>
struct __mapping_result;

template <::cuda::std::size_t _StaticGroupCount,
          ::cuda::std::size_t _StaticUnitCount,
          bool _IsAlwaysExhaustive,
          bool _IsAlwaysContiguous>
struct __mapping_result<__uniform_static_unit_count<_StaticGroupCount, _StaticUnitCount>,
                        _IsAlwaysExhaustive,
                        _IsAlwaysContiguous>
{
  ::cuda::std::uint32_t __group_count_;
  ::cuda::std::uint32_t __group_rank_;
  ::cuda::std::uint32_t __unit_count_;
  ::cuda::std::uint32_t __unit_rank_;
  ::cuda::device::lane_mask __lane_mask_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result __invalid()
  {
    return {__invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            ::cuda::device::lane_mask::none()};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count()
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group count of thread that is not part of the group is UB");
    }

    if constexpr (_StaticGroupCount != ::cuda::std::dynamic_extent)
    {
      return static_cast<::cuda::std::uint32_t>(_StaticGroupCount);
    }
    else
    {
      return __group_count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_rank()
  {
    return _StaticGroupCount == 1 ? 0 : ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
    }

    if constexpr (static_group_rank() != ::cuda::std::dynamic_extent)
    {
      return static_cast<::cuda::std::uint32_t>(static_group_rank());
    }
    else
    {
      return __group_rank_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool has_uniform_static_unit_count()
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count()
  {
    return _StaticUnitCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    if constexpr (_StaticGroupCount != ::cuda::std::dynamic_extent)
    {
      _CCCL_ASSERT(__i < _StaticGroupCount, "__i out of range");
    }
    return _StaticUnitCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit count of thread that is not part of the group is UB");
    }

    if constexpr (_StaticUnitCount != ::cuda::std::dynamic_extent)
    {
      return static_cast<::cuda::std::uint32_t>(_StaticUnitCount);
    }
    else
    {
      return __unit_count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit rank of thread that is not part of the group is UB");
    }

    if constexpr (_StaticUnitCount == 1)
    {
      return 0;
    }
    else
    {
      return __unit_rank_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting lane mask of thread that is not part of the group is UB");
    }
    return __lane_mask_;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const
  {
    if constexpr (_IsAlwaysExhaustive)
    {
      return true;
    }
    else
    {
      return __unit_rank_ != __invalid_count_or_rank;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive()
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _IsAlwaysContiguous;
  }
};

template <::cuda::std::size_t... _StaticUnitCounts, bool _IsAlwaysExhaustive, bool _IsAlwaysContiguous>
struct __mapping_result<__static_unit_counts<_StaticUnitCounts...>, _IsAlwaysExhaustive, _IsAlwaysContiguous>
{
  static constexpr ::cuda::std::size_t _StaticGroupCount = sizeof...(_StaticUnitCounts);

  static_assert(_StaticGroupCount > 0, "per-group unit counts require at least one group");
  static_assert(((_StaticUnitCounts != ::cuda::std::dynamic_extent) && ...),
                "per-group unit counts must be statically known");

  ::cuda::std::uint32_t __group_rank_;
  ::cuda::std::uint32_t __unit_rank_;
  ::cuda::device::lane_mask __lane_mask_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result __invalid()
  {
    return {__invalid_count_or_rank, __invalid_count_or_rank, ::cuda::device::lane_mask::none()};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count()
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group count of thread that is not part of the group is UB");
    }
    return static_cast<::cuda::std::uint32_t>(_StaticGroupCount);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_rank()
  {
    return _StaticGroupCount == 1 ? 0 : ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
    }
    if constexpr (static_group_rank() != ::cuda::std::dynamic_extent)
    {
      return static_cast<::cuda::std::uint32_t>(static_group_rank());
    }
    else
    {
      return __group_rank_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool has_uniform_static_unit_count()
  {
    return ((static_unit_count(0) == _StaticUnitCounts) && ...);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count()
  {
    if constexpr (has_uniform_static_unit_count())
    {
      return static_unit_count(0);
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    _CCCL_ASSERT(__i < _StaticGroupCount, "__i out of range");
    constexpr ::cuda::std::size_t __counts[]{_StaticUnitCounts...};
    return __counts[__i];
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit count of thread that is not part of the group is UB");
    }
    return static_cast<::cuda::std::uint32_t>(static_unit_count(__group_rank_));
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit rank of thread that is not part of the group is UB");
    }
    return __unit_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting lane mask of thread that is not part of the group is UB");
    }
    return __lane_mask_;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const
  {
    if constexpr (_IsAlwaysExhaustive)
    {
      return true;
    }
    else
    {
      return __unit_rank_ != __invalid_count_or_rank;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive()
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _IsAlwaysContiguous;
  }
};

// Return the low-bit prefix ending at the set bit with the given rank. Keeping
// the prefix as a mask avoids converting it to a bit position and back.
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t
__prefix_mask_for_rank(::cuda::std::uint32_t __value, ::cuda::std::uint32_t __rank) noexcept
{
  ::cuda::std::uint32_t __mask = ~0u;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (unsigned __step = 16; __step != 0; __step /= 2)
  {
    const auto __candidate = __mask >> __step;
    if (static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__value & __candidate)) > __rank)
    {
      __mask = __candidate;
    }
  }
  return __mask;
}

template <bool _IsAlwaysContiguous>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask __make_lane_mask_for_n(
  ::cuda::device::lane_mask __prev_lane_mask, ::cuda::std::uint32_t __n, ::cuda::std::uint32_t __rank)
{
  if constexpr (_IsAlwaysContiguous)
  {
    const auto __lane   = ::cuda::ptx::get_sreg_laneid();
    const auto __before = ::cuda::std::min(__rank, __lane);
    const auto __after  = ::cuda::std::min(__n - __rank - 1, 31u - __lane);
    return __prev_lane_mask & (::cuda::device::lane_mask::all() << (__lane - __before))
         & (::cuda::device::lane_mask::all() >> (31u - __lane - __after));
  }
  else
  {
    if (_CCCL_IS_CONSTANT_P(__n) && __n == 1)
    {
      return ::cuda::device::lane_mask::this_lane();
    }

    const auto __mask = __prev_lane_mask.value();
    const auto __nless =
      static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__mask & ::cuda::ptx::get_sreg_lanemask_lt()));
    const auto __ngreater  = static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__mask)) - __nless - 1;
    const auto __remaining = __n - __rank;
    if (__nless <= __rank && __ngreater < __remaining)
    {
      return __prev_lane_mask;
    }

    const auto __first_rank = __nless - ::cuda::std::min(__nless, __rank);
    const auto __last_rank  = __nless + ::cuda::std::min(__ngreater, __remaining - 1);
    const auto __first_mask = ::cuda::experimental::coop::__prefix_mask_for_rank(__mask, __first_rank);
    const auto __last_mask  = ::cuda::experimental::coop::__prefix_mask_for_rank(__mask, __last_rank);
    return ::cuda::device::lane_mask{__mask & ~(__first_mask >> 1) & __last_mask};
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_MAPPING_RESULT_CUH
