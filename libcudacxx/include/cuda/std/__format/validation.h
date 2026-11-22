//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_VALIDATION_H
#define _CUDA_STD___FORMAT_VALIDATION_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_arg_store.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief Dummy format_context only providing the parts used during compile-time validation of the basic_format_string.
template <class _CharT>
class __fmt_validation_format_context
{
public:
  using __parse_function
    _CCCL_NODEBUG = void (*)(basic_format_parse_context<_CharT>&, const __fmt_validation_format_context&);

private:
  const __fmt_arg_t* __args_;
  const __parse_function* __parsers_;
  size_t __size_;

public:
  using char_type = _CharT;

  //! @brief Dummy iterator. During the compile-time validation nothing needs to be written. Therefore all operations of
  //!        this iterator are a NOP.
  struct iterator
  {
    _CCCL_HOST_DEVICE_API constexpr iterator& operator=(_CharT)
    {
      return *this;
    }
    _CCCL_HOST_DEVICE_API constexpr iterator& operator*()
    {
      return *this;
    }
    _CCCL_HOST_DEVICE_API constexpr iterator operator++(int)
    {
      return *this;
    }
  };

  _CCCL_HOST_DEVICE_API constexpr explicit __fmt_validation_format_context(
    const __fmt_arg_t* __args, const __parse_function* __parsers, size_t __size)
      : __args_{__args}
      , __parsers_{__parsers}
      , __size_{__size}
  {}

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __fmt_arg_t arg(size_t __id) const
  {
    if (__id >= __size_)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    return __args_[__id];
  }

  _CCCL_HOST_DEVICE_API constexpr void
  __validate_arg(size_t __id, basic_format_parse_context<_CharT>& __parse_ctx, bool __parse) const
  {
    const __fmt_arg_t __type = arg(__id);
    if (__type == __fmt_arg_t::__none)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    if (__parse || __type == __fmt_arg_t::__handle)
    {
      __parsers_[__id](__parse_ctx, *this);
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator out()
  {
    return {};
  }
  _CCCL_HOST_DEVICE_API constexpr void advance_to(iterator) {}
};

// [format.string.std]/8
// If { arg-idopt } is used in a width or precision, the value of the
// corresponding formatting argument is used in its place. If the
// corresponding formatting argument is not of standard signed or unsigned
// integer type, or its value is negative for precision or non-positive for
// width, an exception of type format_error is thrown.
//
// _HasPrecision does the formatter have a precision?
template <class _CharT, class _Tp, bool _HasPrecision = false>
_CCCL_HOST_DEVICE_API constexpr void __fmt_validate_format_arg(
  basic_format_parse_context<_CharT>& __parse_ctx, const __fmt_validation_format_context<_CharT>& __ctx)
{
  // LWG3720 originally allowed "signed or unsigned integer types", however
  // the final version explicitly changed it to "*standard* signed or unsigned
  // integer types". It's trivial to use 128-bit integrals in libc++'s
  // implementation, but other implementations may not implement it.
  // (Using a width or precision, that does not fit in 64-bits, sounds very
  // unlikely in real world code.)

  formatter<_Tp, _CharT> __formatter;
  __parse_ctx.advance_to(__formatter.parse(__parse_ctx));
  if (__formatter.__parser_.__width_as_arg_)
  {
    switch (__ctx.arg(__formatter.__parser_.__width_))
    {
      case __fmt_arg_t::__int:
      case __fmt_arg_t::__long_long:
      case __fmt_arg_t::__unsigned:
      case __fmt_arg_t::__unsigned_long_long:
        break;
      default:
        ::cuda::std::__throw_format_error("Replacement argument isn't a standard signed or unsigned integer type");
    }
  }

  if constexpr (_HasPrecision)
  {
    if (__formatter.__parser_.__precision_as_arg_)
    {
      switch (__ctx.arg(__formatter.__parser_.__precision_))
      {
        case __fmt_arg_t::__int:
        case __fmt_arg_t::__long_long:
        case __fmt_arg_t::__unsigned:
        case __fmt_arg_t::__unsigned_long_long:
          break;
        default:
          ::cuda::std::__throw_format_error("Replacement argument isn't a standard signed or unsigned integer type");
      }
    }
  }
}

template <class _CharT, class _Tp>
_CCCL_HOST_DEVICE_API constexpr void
__fmt_validate_handle(basic_format_parse_context<_CharT>& __parse_ctx, const __fmt_validation_format_context<_CharT>&)
{
  formatter<_Tp, _CharT> __formatter;
  __parse_ctx.advance_to(__formatter.parse(__parse_ctx));
}

// Select the parser when the argument types are known. Several stored types use the same format specification.
template <class _Context, class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr typename _Context::__parse_function
__fmt_select_validation_parser() noexcept
{
  using _CharT          = typename _Context::char_type;
  constexpr auto __type = ::cuda::std::__fmt_determine_arg_t<_Context, _Tp>();
  if constexpr (__type == __fmt_arg_t::__none)
  {
    return nullptr;
  }
  else if constexpr (__type == __fmt_arg_t::__handle)
  {
    return ::cuda::std::__fmt_validate_handle<_CharT, _Tp>;
  }
  else if constexpr (__type == __fmt_arg_t::__boolean)
  {
    return ::cuda::std::__fmt_validate_format_arg<_CharT, bool>;
  }
  else if constexpr (__type == __fmt_arg_t::__char_type)
  {
    return ::cuda::std::__fmt_validate_format_arg<_CharT, _CharT>;
  }
  else if constexpr (__type == __fmt_arg_t::__ptr)
  {
    return ::cuda::std::__fmt_validate_format_arg<_CharT, const void*>;
  }
  else if constexpr (__type == __fmt_arg_t::__const_char_type_ptr || __type == __fmt_arg_t::__string_view)
  {
    return ::cuda::std::__fmt_validate_format_arg<_CharT, basic_string_view<_CharT>, true>;
  }
  else if constexpr (__type == __fmt_arg_t::__float || __type == __fmt_arg_t::__double
                     || __type == __fmt_arg_t::__long_double)
  {
    return ::cuda::std::__fmt_validate_format_arg<_CharT, float, true>;
  }
  else
  {
    static_assert(__type == __fmt_arg_t::__int || __type == __fmt_arg_t::__unsigned
                  || __type == __fmt_arg_t::__long_long || __type == __fmt_arg_t::__unsigned_long_long);
    return ::cuda::std::__fmt_validate_format_arg<_CharT, int>;
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_VALIDATION_H
