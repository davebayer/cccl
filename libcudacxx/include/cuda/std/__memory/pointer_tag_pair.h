//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_POINTER_TAG_PAIR_H
#define _CUDA_STD___MEMORY_POINTER_TAG_PAIR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Up, class _Ptr, unsigned _BitsRequested, size_t _Alignment = alignof(_Up)>
concept __tagging_compatible_pointee = true;
//       = convertible_to<_Up*, _Ptr> && pointer_bits_available(_Alignment) >= _BitsRequested
//         && (is_void_v<remove_pointer_t<_Ptr>>
//           || is_scalar_v<remove_pointer_t<_Ptr>>
//           || is_union_v<remove_pointer_t<_Ptr>>
//           || is_pointer_interconvertible_base_of_v<remove_pointer_t<_Ptr>, _Up>);

//   template <class TagT> constexpr unsigned tag-bit-width(TagT value) noexcept; // exposition only

template <class _Tp, class = void>
inline constexpr bool __ptrtag_is_unsigned_v = is_unsigned_v<_Tp>;
template <class _Tp>
inline constexpr bool __ptrtag_is_unsigned_v<_Tp, enable_if_t<is_enum_v<_Tp>>> = is_unsigned_v<underlying_type_t<_Tp>>;

template <class _Ptr,
          unsigned _BitsRequested = 0, // bits-available<remove_pointer_t<_Ptr>>,
          class _Tag              = unsigned>
class pointer_tag_pair
{
  static_assert(is_same_v<remove_cvref_t<_Ptr>, _Ptr>, "_Ptr must be cvref-unqualified");
  static_assert(is_same_v<remove_cvref_t<_Tag>, _Tag>, "_Tag must be cvref-unqualified");
  static_assert(is_pointer_v<_Ptr> && !is_function_v<remove_pointer_t<_Ptr>>, "_Ptr mustn't be a function pointer");
  static_assert(__ptrtag_is_unsigned_v<_Tag>);
  static_assert(sizeof(TagT) <= sizeof(void*));
  static_assert(_BitsRequested <= max_pointer_bits_available);

public:
  using pointer_type        = _Ptr;
  using element_type        = remove_pointer_t<_Ptr>;
  using tagged_pointer_type = see below;
  using tag_type            = _Tag;

  static constexpr unsigned bits_requested = _BitsRequested;

  // Constructors and assignment
  _CCCL_API constexpr pointer_tag_pair() noexcept {}

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__tagging_compatible_pointee<_Tp, pointer_type, bits_requested>)
  _CCCL_API constexpr pointer_tag_pair(_Tp* __ptr, tag_type __tag) {}

  _CCCL_API constexpr pointer_tag_pair(nullptr_t __ptr, tag_type __tag) {}

  // Special construction helpers
  _CCCL_TEMPLATE(size_t PromisedAlignment, class _Tp)
  _CCCL_REQUIRES(__tagging_compatible_pointee<_Tp, pointer_type, bits_requested, PromisedAlignment>)
  [[nodiscard]] _CCCL_API static constexpr pointer_tag_pair from_overaligned(_Tp* __ptr, tag_type __tag) {}

  [[nodiscard]] _CCCL_API static pointer_tag_pair from_tagged(tagged_pointer_type __ptr) noexcept {}

  // Accessors
  [[nodiscard]] _CCCL_API tagged_pointer_type tagged_pointer() const noexcept {}

  [[nodiscard]] _CCCL_API constexpr pointer_type pointer() const noexcept {}

  [[nodiscard]] _CCCL_API constexpr tag_type tag() const noexcept {}

  // Swap
  [[nodiscard]] _CCCL_API constexpr void swap(pointer_tag_pair& __other) noexcept {}

  // Comparisons
  friend constexpr see
    - below operator<=>(pointer_tag_pair lhs, pointer_tag_pair rhs) noexcept
      requires three_way_comparable<tag_type>;
  friend constexpr bool operator==(pointer_tag_pair, pointer_tag_pair) noexcept
    requires equality_comparable<tag_type>;
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES pointer_tag_pair(_Tp*) -> pointer_tag_pair<_Tp*>;

// template <class _Tp, class _Tag>
// _CCCL_DEDUCTION_GUIDE_ATTRIBUTES pointer_tag_pair(_Tp*, _Tag tag) -> pointer_tag_pair<_Tp*, bits-available<_Tp>,
// _Tag>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_POINTER_TAG_PAIR_H
