//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FORMAT_FORMAT_ARG_STORE_H
#define _LIBCUDACXX___FORMAT_FORMAT_ARG_STORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/concepts.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_bounded_array.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/cstdint>
#include <cuda/std/string_view>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __format
{

template <class _Arr, class _Elem>
inline constexpr bool __is_bounded_array_of = false;

template <class _Elem, size_t _Len>
inline constexpr bool __is_bounded_array_of<_Elem[_Len], _Elem> = true;

template <class _Context, class _Tp>
[[nodiscard]] _CCCL_CONSTEVAL __arg_t __determine_arg_t() noexcept
{
  if constexpr (is_same_v<_Tp, bool>)
  {
    return __arg_t::__boolean;
  }
  else if constexpr (is_same_v<_Tp, typename _Context::char_type>)
  {
    return __arg_t::__char_type;
  }
#if _CCCL_HAS_WCHAR_T()
  else if constexpr (is_same_v<_Tp, char> && is_same_v<typename _Context::char_type, wchar_t>)
  {
    return __arg_t::__char_type;
  }
#endif // _CCCL_HAS_WCHAR_T()
  else if constexpr (__cccl_is_integer_v<_Tp>)
  {
    if constexpr (sizeof(_Tp) <= sizeof(int))
    {
      return (is_signed_v<_Tp>) ? __arg_t::__int : __arg_t::__unsigned;
    }
    else if constexpr (sizeof(_Tp) <= sizeof(long long))
    {
      return (is_signed_v<_Tp>) ? __arg_t::__long_long : __arg_t::__unsigned_long_long;
    }
#if _CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(__int128_t))
    {
      return (is_signed_v<_Tp>) ? __arg_t::__i128 : __arg_t::__u128;
    }
#endif // _CCCL_HAS_INT128()
    else
    {
      static_assert(__always_false_v<_Tp>, "an unsupported integer type was used");
    }
  }
  else if (is_same_v<_Tp, float>)
  {
    return __arg_t::__float;
  }
  else if (is_same_v<_Tp, double>)
  {
    return __arg_t::__double;
  }
  else if (is_same_v<_Tp, long double>)
  {
    return __arg_t::__long_double;
  }
  else if constexpr (is_same_v<_Tp, typename _Context::char_type*>
                     || is_same_v<_Tp, const typename _Context::char_type*>)
  {
    return __arg_t::__const_char_type_ptr;
  }
  else if constexpr (is_bounded_array_v<_Tp> && is_same_v<remove_extent_t<_Tp>, typename _Context::char_type>)
  {
    return __arg_t::__string_view;
  }
  else if constexpr (is_same_v<_Tp, void*> || is_same_v<_Tp, const void*> || is_same_v<_Tp, nullptr_t>)
  {
    return __arg_t::__ptr;
  }
  // todo: string & string_view
  else if constexpr (__formattable_with<_Tp, _Context>)
  {
    return __arg_t::__handle;
  }
  else
  {
    static_assert(__always_false_v<_Tp>, "the supplied type is not formattable");
    _CCCL_UNREACHABLE();
  }
}

// // String view
// template <class _Context, class _Tp>
//   requires(same_as<typename _Context::char_type, typename _Tp::value_type> &&
//            same_as<_Tp, basic_string_view<typename _Tp::value_type, typename _Tp::traits_type>>)
// _CCCL_CONSTEVAL __arg_t __determine_arg_t() {
//   return __arg_t::__string_view;
// }

// // String
// template <class _Context, class _Tp>
//   requires(
//       same_as<typename _Context::char_type, typename _Tp::value_type> &&
//       same_as<_Tp, basic_string<typename _Tp::value_type, typename _Tp::traits_type, typename _Tp::allocator_type>>)
// _CCCL_CONSTEVAL __arg_t __determine_arg_t() {
//   return __arg_t::__string_view;
// }

// Pseudo constructor for basic_format_arg
//
// Modeled after template<class T> explicit basic_format_arg(T& v) noexcept;
// [format.arg]/4-6
template <class _Context, class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI basic_format_arg<_Context> __create_format_arg(_Tp& __value) noexcept
{
  using _Dp               = remove_const_t<_Tp>;
  constexpr __arg_t __arg = __format::__determine_arg_t<_Context, _Dp>();
  static_assert(__formattable_with<_Tp, _Context>);

  using __context_char_type = _Context::char_type;
  // Not all types can be used to directly initialize the
  // __basic_format_arg_value.  First handle all types needing adjustment, the
  // final else requires no adjustment.
  if constexpr (__arg == __arg_t::__char_type)

#if _LIBCPP_HAS_WIDE_CHARACTERS
    if constexpr (same_as<__context_char_type, wchar_t> && same_as<_Dp, char>)
    {
      return basic_format_arg<_Context>{__arg, static_cast<wchar_t>(static_cast<unsigned char>(__value))};
    }
    else
#endif
      return basic_format_arg<_Context>{__arg, __value};
  else if constexpr (__arg == __arg_t::__int)
  {
    return basic_format_arg<_Context>{__arg, static_cast<int>(__value)};
  }
  else if constexpr (__arg == __arg_t::__long_long)
  {
    return basic_format_arg<_Context>{__arg, static_cast<long long>(__value)};
  }
  else if constexpr (__arg == __arg_t::__unsigned)
  {
    return basic_format_arg<_Context>{__arg, static_cast<unsigned>(__value)};
  }
  else if constexpr (__arg == __arg_t::__unsigned_long_long)
  {
    return basic_format_arg<_Context>{__arg, static_cast<unsigned long long>(__value)};
  }
  else if constexpr (__arg == __arg_t::__string_view)
  {
    // Using std::size on a character array will add the NUL-terminator to the size.
    if constexpr (__is_bounded_array_of<_Dp, __context_char_type>)
    {
      const __context_char_type* const __pbegin = std::begin(__value);
      const __context_char_type* const __pzero =
        char_traits<__context_char_type>::find(__pbegin, extent_v<_Dp>, __context_char_type{});
      _LIBCPP_ASSERT_VALID_INPUT_RANGE(__pzero != nullptr, "formatting a non-null-terminated array");
      return basic_format_arg<_Context>{
        __arg, basic_string_view<__context_char_type>{__pbegin, static_cast<size_t>(__pzero - __pbegin)}};
    }
    else
    {
      // When the _Traits or _Allocator are different an implicit conversion will fail.
      return basic_format_arg<_Context>{__arg, basic_string_view<__context_char_type>{__value.data(), __value.size()}};
    }
  }
  else if constexpr (__arg == __arg_t::__ptr)
  {
    return basic_format_arg<_Context>{__arg, static_cast<const void*>(__value)};
  }
  else if constexpr (__arg == __arg_t::__handle)
  {
    return basic_format_arg<_Context>{__arg, typename __basic_format_arg_value<_Context>::__handle{__value}};
  }
  else
  {
    return basic_format_arg<_Context>{__arg, __value};
  }
}

template <class _Context, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI void
__create_packed_storage(uint64_t& __types, __basic_format_arg_value<_Context>* __values, _Args&... __args) noexcept
{
  int __shift = 0;
  (
    [&] {
      basic_format_arg<_Context> __arg = __format::__create_format_arg<_Context>(__args);
      if (__shift != 0)
      {
        __types |= static_cast<uint64_t>(__arg.__type_) << __shift;
      }
      else
      {
        // Assigns the initial value.
        __types = static_cast<uint64_t>(__arg.__type_);
      }
      __shift += __packed_arg_t_bits;
      *__values++ = __arg.__value_;
    }(),
    ...);
}

template <class _Context, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI void __store_basic_format_arg(basic_format_arg<_Context>* __data, _Args&... __args) noexcept
{
  (
    [&] {
      *__data++ = __format::__create_format_arg<_Context>(__args);
    }(),
    ...);
}

template <class _Context, size_t _Np>
struct __packed_format_arg_store
{
  __basic_format_arg_value<_Context> __values_[_Np];
  uint64_t __types_ = 0;
};

template <class _Context>
struct __packed_format_arg_store<_Context, 0>
{
  uint64_t __types_ = 0;
};

template <class _Context, size_t _Np>
struct __unpacked_format_arg_store
{
  basic_format_arg<_Context> __args_[_Np];
};

} // namespace __format

template <class _Context, class... _Args>
struct __format_arg_store
{
  _LIBCPP_HIDE_FROM_ABI __format_arg_store(_Args&... __args) noexcept
  {
    if constexpr (sizeof...(_Args) != 0)
    {
      if constexpr (__format::__use_packed_format_arg_store(sizeof...(_Args)))
      {
        __format::__create_packed_storage(__storage.__types_, __storage.__values_, __args...);
      }
      else
      {
        __format::__store_basic_format_arg<_Context>(__storage.__args_, __args...);
      }
    }
  }

  using _Storage _LIBCPP_NODEBUG =
    conditional_t<__format::__use_packed_format_arg_store(sizeof...(_Args)),
                  __format::__packed_format_arg_store<_Context, sizeof...(_Args)>,
                  __format::__unpacked_format_arg_store<_Context, sizeof...(_Args)>>;

  _Storage __storage;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FORMAT_FORMAT_ARG_STORE_H
