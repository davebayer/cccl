//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_OUTPUT_UTILS_H
#define _CUDA_STD__FORMAT_OUTPUT_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__algorithm/transform.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/buffer.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr char __fmt_hex_to_upper(char __c) noexcept
{
  switch (__c)
  {
    case 'a':
      return 'A';
    case 'b':
      return 'B';
    case 'c':
      return 'C';
    case 'd':
      return 'D';
    case 'e':
      return 'E';
    case 'f':
      return 'F';
    default:
      return __c;
  }
}

struct __fmt_padding_size_result
{
  size_t __before_;
  size_t __after_;
};

// nvcc warns about missing return statement when compiling with msvc host compiler, adding more unreachables doesn't
// help, so let's just suppress the warning
#if _CCCL_COMPILER(MSVC)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(940) // missing return statement at end of non-void function
#endif // _CCCL_COMPILER(MSVC)

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __fmt_padding_size_result
__fmt_padding_size(size_t __size, size_t __width, __fmt_spec_alignment __align)
{
  _CCCL_ASSERT(__width > __size, "don't call this function when no padding is required");
  _CCCL_ASSERT(__align != __fmt_spec_alignment::__zero_padding, "the caller should have handled the zero-padding");

  const size_t __fill = __width - __size;
  switch (__align)
  {
    case __fmt_spec_alignment::__left:
      return {0, __fill};
    case __fmt_spec_alignment::__center: {
      // The extra padding is divided per [format.string.std]/3
      // __before = floor(__fill, 2);
      // __after = ceil(__fill, 2);
      const size_t __before = __fill / 2;
      const size_t __after  = __fill - __before;
      return {__before, __after};
    }
    case __fmt_spec_alignment::__default:
    case __fmt_spec_alignment::__right:
      return {__fill, 0};
    case __fmt_spec_alignment::__zero_padding:
    default:
      _CCCL_UNREACHABLE();
  }
}

#if _CCCL_COMPILER(MSVC)
_CCCL_END_NV_DIAG_SUPPRESS()
#endif // _CCCL_COMPILER(MSVC)

//! Copy wrapper.
//!
//! This uses a "mass output function" of __fmt_output_buffer when possible.
template <class _CharT, class _OutCharT = _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_copy(basic_string_view<_CharT> __str, _OutIt __out_it)
{
  // todo: handle __fmt_retarget_buffer once implemented
  if constexpr (same_as<decltype(__out_it), __back_insert_iterator<__fmt_output_buffer<_OutCharT>>>)
  {
    __out_it.__get_container()->__copy(__str);
    return __out_it;
  }
  else
  {
    return ::cuda::std::copy(__str.begin(), __str.end(), ::cuda::std::move(__out_it));
  }
}

template <class _It, class _CharT = iter_value_t<_It>, class _OutCharT = _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_copy(_It __first, _It __last, _OutIt __out_it)
{
  return ::cuda::std::__fmt_copy(basic_string_view{__first, __last}, ::cuda::std::move(__out_it));
}

template <class _It, class _CharT = iter_value_t<_It>, class _OutCharT = _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_copy(_It __first, size_t __n, _OutIt __out_it)
{
  return ::cuda::std::__fmt_copy(basic_string_view{::cuda::std::to_address(__first), __n}, ::cuda::std::move(__out_it));
}

//! Transform wrapper.
//!
//! This uses a "mass output function" of __fmt_output_buffer when possible.
template <class _It, class _CharT = iter_value_t<_It>, class _OutCharT = _CharT, class _OutIt, class _UnaryOp>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_transform(_It __first, _It __last, _OutIt __out_it, _UnaryOp __operation)
{
  // todo: handle __fmt_retarget_buffer once implemented
  if constexpr (same_as<decltype(__out_it), __back_insert_iterator<__fmt_output_buffer<_OutCharT>>>)
  {
    __out_it.__get_container()->__transform(__first, __last, ::cuda::std::move(__operation));
    return __out_it;
  }
  else
  {
    return ::cuda::std::transform(__first, __last, ::cuda::std::move(__out_it), __operation);
  }
}

//! Fill wrapper.
//!
//! This uses a "mass output function" of __fmt_output_buffer when possible.
template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_fill(_OutIt __out_it, size_t __n, _CharT __value)
{
  // todo: handle __fmt_retarget_buffer once implemented
  if constexpr (same_as<decltype(__out_it), __back_insert_iterator<__fmt_output_buffer<_CharT>>>)
  {
    __out_it.__get_container()->__fill(__n, __value);
    return __out_it;
  }
  else
  {
    return ::cuda::std::fill_n(::cuda::std::move(__out_it), __n, __value);
  }
}

template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_fill(_OutIt __out_it, size_t __n, __fmt_spec_code_point<_CharT> __value)
{
  return ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __n, __value.__data[0]);
}

//! Writes the input to the output with the required padding.
//!
//! Since the output column width is specified the function can be used for
//! ASCII and Unicode output.
//!
//! @pre \a __size <= \a __width. Using this function when this pre-condition
//!      doesn't hold incurs an unwanted overhead.
//!
//! @param __str       The string to write.
//! @param __out_it    The output iterator to write to.
//! @param __specs     The parsed formatting specifications.
//! @param __size      The (estimated) output column width. When the elements
//!                    to be written are ASCII the following condition holds
//!                    \a __size == \a __last - \a __first.
//!
//! @returns           An iterator pointing beyond the last element written.
//!
//! @note The type of the elements in range [\a __first, \a __last) can differ
//! from the type of \a __specs. Integer output uses \c std::to_chars for its
//! conversion, which means the [\a __first, \a __last) always contains elements
//! of the type \c char.
template <class _CharT, class _ParserCharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_write(basic_string_view<_CharT> __str, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs, ptrdiff_t __size)
{
  if (__size >= static_cast<ptrdiff_t>(__specs.__width_))
  {
    return ::cuda::std::__fmt_copy(__str, ::cuda::std::move(__out_it));
  }

  const auto __padding =
    ::cuda::std::__fmt_padding_size(__size, __specs.__width_, __fmt_spec_alignment{__specs.__std_.__alignment_});
  __out_it = ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__before_, __specs.__fill_);
  __out_it = ::cuda::std::__fmt_copy(__str, ::cuda::std::move(__out_it));
  return ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__after_, __specs.__fill_);
}

template <class _It, class _ParserCharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_write(_It __first, _It __last, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs, ptrdiff_t __size)
{
  _CCCL_ASSERT(__first <= __last, "Not a valid range");
  return ::cuda::std::__fmt_write(basic_string_view{__first, __last}, ::cuda::std::move(__out_it), __specs, __size);
}

// Calls the function above where \a __size = \a __last - \a __first.
template <class _It, class _ParserCharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_write(_It __first, _It __last, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs)
{
  _CCCL_ASSERT(__first <= __last, "Not a valid range");
  return ::cuda::std::__fmt_write(__first, __last, ::cuda::std::move(__out_it), __specs, __last - __first);
}

template <class _It, class _CharT = iter_value_t<_It>, class _ParserCharT, class _OutIt, class _UnaryOp>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_write_transformed(
  _It __first, _It __last, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs, _UnaryOp __op)
{
  _CCCL_ASSERT(__first <= __last, "Not a valid range");

  ptrdiff_t __size = __last - __first;
  if (__size >= __specs.__width_)
  {
    return ::cuda::std::__fmt_transform(__first, __last, ::cuda::std::move(__out_it), __op);
  }
  const auto __padding =
    ::cuda::std::__fmt_padding_size(__size, __specs.__width_, __fmt_spec_alignment{__specs.__alignment_});
  __out_it = ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__before_, __specs.__fill_);
  __out_it = ::cuda::std::__fmt_transform(__first, __last, ::cuda::std::move(__out_it), __op);
  return ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__after_, __specs.__fill_);
}

//! Writes a string using format's width estimation algorithm.
//!
//! @pre !__specs.__has_precision()
//!
//! @note When \c _LIBCPP_HAS_UNICODE is false the function assumes the input is ASCII.
template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_write_string_no_precision(basic_string_view<_CharT> __str, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  _CCCL_ASSERT(!__specs.__has_precision(), "use __write_string");

  // No padding -> copy the string
  if (!__specs.__has_width())
  {
    return ::cuda::std::__fmt_copy(__str, ::cuda::std::move(__out_it));
  }

  // Note when the estimated width is larger than size there's no padding. So
  // there's no reason to get the real size when the estimate is larger than or
  // equal to the minimum field width.
  size_t __size =
    ::cuda::std::__fmt_estimate_column_width(__str, __specs.__width_, __fmt_column_width_rounding::__up).__width_;
  return ::cuda::std::__fmt_write(__str, ::cuda::std::move(__out_it), __specs, __size);
}

template <class _CharT>
[[nodiscard]] _CCCL_HOST_DEVICE_API int __fmt_truncate(basic_string_view<_CharT>& __str, int __precision)
{
  const auto __result =
    ::cuda::std::__fmt_estimate_column_width(__str, __precision, __fmt_column_width_rounding::__down);
  __str = basic_string_view<_CharT>{__str.begin(), __result.__last_};
  return static_cast<int>(__result.__width_);
}

//! Writes a string using format's width estimation algorithm.
template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_write_string(basic_string_view<_CharT> __str, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  if (!__specs.__has_precision())
  {
    return ::cuda::std::__fmt_write_string_no_precision(__str, ::cuda::std::move(__out_it), __specs);
  }
  int __size = ::cuda::std::__fmt_truncate(__str, __specs.__precision_);
  return ::cuda::std::__fmt_write(__str.begin(), __str.end(), ::cuda::std::move(__out_it), __specs, __size);
}

enum class __fmt_nul_terminator
{
};

template <class _CharT>
[[nodiscard]] _CCCL_API bool operator==(const _CharT* __cstr, __fmt_nul_terminator)
{
  return *__cstr == _CharT('\0');
}

template <class _CharT>
_CCCL_API void __fmt_write_escaped_code_unit(basic_string<_CharT>& __str, char32_t __value, const _CharT* __prefix)
{
  back_insert_iterator __out_it{__str};
  std::ranges::copy(__prefix, __fmt_nul_terminator{}, __out_it);

  char __buffer[8];
  to_chars_result __r = std::to_chars(std::begin(__buffer), std::end(__buffer), __value, 16);
  _LIBCPP_ASSERT_INTERNAL(__r.ec == errc(0), "Internal buffer too small");
  std::ranges::copy(std::begin(__buffer), __r.ptr, __out_it);

  __str += _CharT('}');
}

// [format.string.escaped]/2.2.1.2
// ...
// then the sequence \u{hex-digit-sequence} is appended to E, where
// hex-digit-sequence is the shortest hexadecimal representation of C using
// lower-case hexadecimal digits.
template <class _CharT>
_LIBCPP_HIDE_FROM_ABI void __write_well_formed_escaped_code_unit(basic_string<_CharT>& __str, char32_t __value)
{
  __formatter::__write_escaped_code_unit(__str, __value, _LIBCPP_STATICALLY_WIDEN(_CharT, "\\u{"));
}

// [format.string.escaped]/2.2.3
// Otherwise (X is a sequence of ill-formed code units), each code unit U is
// appended to E in order as the sequence \x{hex-digit-sequence}, where
// hex-digit-sequence is the shortest hexadecimal representation of U using
// lower-case hexadecimal digits.
template <class _CharT>
_LIBCPP_HIDE_FROM_ABI void __write_escape_ill_formed_code_unit(basic_string<_CharT>& __str, char32_t __value)
{
  __formatter::__write_escaped_code_unit(__str, __value, _LIBCPP_STATICALLY_WIDEN(_CharT, "\\x{"));
}

template <class _CharT>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool
__fmt_is_escaped_sequence_written(basic_string<_CharT>& __str, bool __last_escaped, char32_t __value)
{
#if !_LIBCPP_HAS_UNICODE
  // For ASCII assume everything above 127 is printable.
  if (__value > 127)
  {
    return false;
  }
#endif

  // [format.string.escaped]/2.2.1.2.1
  //   CE is UTF-8, UTF-16, or UTF-32 and C corresponds to a Unicode scalar
  //   value whose Unicode property General_Category has a value in the groups
  //   Separator (Z) or Other (C), as described by UAX #44 of the Unicode Standard,
  if (!__escaped_output_table::__needs_escape(__value))
  {
    // [format.string.escaped]/2.2.1.2.2
    //   CE is UTF-8, UTF-16, or UTF-32 and C corresponds to a Unicode scalar
    //   value with the Unicode property Grapheme_Extend=Yes as described by UAX
    //   #44 of the Unicode Standard and C is not immediately preceded in S by a
    //   character P appended to E without translation to an escape sequence,
    if (!__last_escaped
        || __extended_grapheme_custer_property_boundary::__get_property(__value)
             != __extended_grapheme_custer_property_boundary::__property::__Extend)
    {
      return false;
    }
  }

  __formatter::__write_well_formed_escaped_code_unit(__str, __value);
  return true;
}

template <class _CharT>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr char32_t __to_char32(_CharT __value)
{
  return static_cast<make_unsigned_t<_CharT>>(__value);
}

enum class __escape_quotation_mark
{
  __apostrophe,
  __double_quote
};

// [format.string.escaped]/2
template <class _CharT>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool __is_escaped_sequence_written(
  basic_string<_CharT>& __str, char32_t __value, bool __last_escaped, __escape_quotation_mark __mark)
{
  // 2.2.1.1 - Mapped character in [tab:format.escape.sequences]
  switch (__value)
  {
    case _CharT('\t'):
      __str += _LIBCPP_STATICALLY_WIDEN(_CharT, "\\t");
      return true;
    case _CharT('\n'):
      __str += _LIBCPP_STATICALLY_WIDEN(_CharT, "\\n");
      return true;
    case _CharT('\r'):
      __str += _LIBCPP_STATICALLY_WIDEN(_CharT, "\\r");
      return true;
    case _CharT('\''):
      if (__mark == __escape_quotation_mark::__apostrophe)
      {
        __str += _LIBCPP_STATICALLY_WIDEN(_CharT, R"(\')");
      }
      else
      {
        __str += __value;
      }
      return true;
    case _CharT('"'):
      if (__mark == __escape_quotation_mark::__double_quote)
      {
        __str += _LIBCPP_STATICALLY_WIDEN(_CharT, R"(\")");
      }
      else
      {
        __str += __value;
      }
      return true;
    case _CharT('\\'):
      __str += _LIBCPP_STATICALLY_WIDEN(_CharT, R"(\\)");
      return true;

    // 2.2.1.2 - Space
    case _CharT(' '):
      __str += __value;
      return true;
  }

  // 2.2.2
  //   Otherwise, if X is a shift sequence, the effect on E and further
  //   decoding of S is unspecified.
  // For now shift sequences are ignored and treated as Unicode. Other parts
  // of the format library do the same. It's unknown how ostream treats them.
  // TODO FMT determine what to do with shift sequences.

  // 2.2.1.2.1 and 2.2.1.2.2 - Escape
  return __formatter::__is_escaped_sequence_written(__str, __last_escaped, __formatter::__to_char32(__value));
}

// Helper struct for the result of a consume operation.
//
// The status value for a correct code point is 0. This allows a valid value to
// be used without masking.
// When the decoding fails it know the number of code units affected. For the
// current use-cases that value is not needed, therefore it is not stored.
// The escape routine needs the number of code units for both a valid and
// invalid character and keeps track of it itself. Doing it in this result
// unconditionally would give some overhead when the value is unneeded.
struct __fmt_consume_result
{
  // When __status == __ok it contains the decoded code point.
  // Else it contains the replacement character U+FFFD
  char32_t __code_point : 31;

  enum : char32_t
  {
    // Consumed a well-formed code point.
    __ok = 0,
    // Encountered invalid UTF-8
    __error = 1
  } __status : 1 {__ok};
};
static_assert(sizeof(__fmt_consume_result) == sizeof(char32_t));

// For ASCII every character is a "code point".
// This makes it easier to write code agnostic of the _LIBCPP_HAS_UNICODE define.
template <class _CharT>
class __fmt_unicode_code_point_view
{
  using _Iterator _CCCL_NODEBUG_ALIAS = typename basic_string_view<_CharT>::const_iterator;

  _Iterator __first_;
  _Iterator __last_;

public:
  _CCCL_API constexpr explicit __fmt_unicode_code_point_view(_Iterator __first, _Iterator __last)
      : __first_(__first)
      , __last_(__last)
  {}

  _CCCL_API constexpr bool __at_end() const noexcept
  {
    return __first_ == __last_;
  }

  _CCCL_API constexpr _Iterator __position() const noexcept
  {
    return __first_;
  }

  [[nodiscard]] _CCCL_API constexpr __consume_result __consume() noexcept
  {
    _CCCL_ASSERT(__first_ != __last_, "can't move beyond the end of input");
    return {static_cast<char32_t>(*__first_++)};
  }
};

template <class _CharT>
_CCCL_API void
__fmt_escape(basic_string<_CharT>& __str, basic_string_view<_CharT> __values, __escape_quotation_mark __mark)
{
  __fmt_unicode_code_point_view<_CharT> __view{__values.begin(), __values.end()};

  // When the first code unit has the property Grapheme_Extend=Yes it needs to
  // be escaped. This happens when the previous code unit was also escaped.
  bool __escape = true;
  while (!__view.__at_end())
  {
    auto __first  = __view.__position();
    auto __result = __view.__consume();
    if (__result.__status == __fmt_consume_result::__ok)
    {
      __escape = __formatter::__is_escaped_sequence_written(__str, __result.__code_point, __escape, __mark);
      if (!__escape)
      {
        // 2.2.1.3 - Add the character
        ranges::copy(__first, __view.__position(), std::back_insert_iterator(__str));
      }
    }
    else
    {
      // 2.2.3 sequence of ill-formed code units
      ranges::for_each(__first, __view.__position(), [&](_CharT __value) {
        __formatter::__write_escape_ill_formed_code_unit(__str, __formatter::__to_char32(__value));
      });
    }
  }
}

template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt __format_escaped_char(_CharT __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  _CharT __buffer[10]; // todo:
  __str += _CharT{'\''};
  ::cuda::std::__fmt_escape(__str, basic_string_view{&__value, 1}, __escape_quotation_mark::__apostrophe);
  __str += _CharT{'\''};
  return ::cuda::std::__fmt_write(
    __str.data(), __str.data() + __str.size(), ::cuda::std::move(__out_it), __specs, __str.size());
}

template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt
__fmt_format_escaped_string(basic_string_view<_CharT> __values, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  __fmt_allocating_buffer<_CharT> __buffer;
  __buffer.push_back(_CharT{'"'});
  ::cuda::std::__fmt_escape(__back_insert_iterator{__buffer}, __values, __fmt_escape_quotation_mark::__double_quote);
  __buffer.push_back(_CharT{'"'});
  return ::cuda::std::__fmt_write_string(__buffer.__view(), ::cuda::std::move(__out_it), __specs);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_OUTPUT_UTILS_H
