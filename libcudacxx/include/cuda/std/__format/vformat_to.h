//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_VFORMAT_TO_H
#define _CUDA_STD___FORMAT_VFORMAT_TO_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/buffer.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_arg_store.h>
#include <cuda/std/__format/format_args.h>
#include <cuda/std/__format/format_context.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__format/parse_arg_id.h>
#include <cuda/std/__format/validation.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/monostate.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool __fmt_is_default_field(basic_string_view<_CharT> __fmt)
{
  return __fmt.size() == 2 && __fmt[0] == _CharT{'{'} && __fmt[1] == _CharT{'}'};
}

// Share the code for explicit format specifications across typed and erased device calls.
template <class _Tp, class _ParseCtx, class _FormatCtx>
_CCCL_HOST_DEVICE_API _CCCL_NOINLINE_DEVICE void
__fmt_format_with_spec(_Tp __arg, _ParseCtx& __parse_ctx, _FormatCtx& __format_ctx)
{
  formatter<_Tp, typename _FormatCtx::char_type> __formatter;
  __parse_ctx.advance_to(__formatter.parse(__parse_ctx));
  __format_ctx.advance_to(__formatter.format(__arg, __format_ctx));
}

template <class _ParseCtx, class _FormatCtx>
struct __fmt_replacement_field_visitor
{
  _ParseCtx& __parse_ctx_;
  _FormatCtx& __format_ctx_;
  bool __parse_;

  template <class _Tp>
  _CCCL_HOST_DEVICE_API constexpr void operator()([[maybe_unused]] _Tp __arg) const
  {
    if constexpr (is_same_v<_Tp, monostate>)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    else if constexpr (is_same_v<_Tp, typename basic_format_arg<_FormatCtx>::handle>)
    {
      __arg.format(__parse_ctx_, __format_ctx_);
    }
    else if (__parse_)
    {
      ::cuda::std::__fmt_format_with_spec(__arg, __parse_ctx_, __format_ctx_);
    }
    else
    {
      const formatter<_Tp, typename _FormatCtx::char_type> __formatter{};
      __format_ctx_.advance_to(__formatter.format(__arg, __format_ctx_));
    }
  }
};

template <class _KnownType, class _It, class _ParseCtx, class _Ctx>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _It
__fmt_handle_replacement_field(_It __begin, _It __end, _ParseCtx& __parse_ctx, _Ctx& __ctx)
{
  using _CharT = iter_value_t<_It>;

  const auto __r = ::cuda::std::__fmt_parse_arg_id(__begin, __end, __parse_ctx);
  if (__r.__last == __end)
  {
    ::cuda::std::__throw_format_error("The argument index should end with a ':' or a '}'");
  }

  const bool __parse = (*__r.__last == _CharT{':'});
  switch (*__r.__last)
  {
    case _CharT{':'}:
      // The arg-id has a format-specifier, advance the input to the format-spec.
      __parse_ctx.advance_to(__r.__last + 1);
      break;
    case _CharT{'}'}:
      // The arg-id has no format-specifier.
      __parse_ctx.advance_to(__r.__last);
      break;
    default:
      ::cuda::std::__throw_format_error("The argument index should end with a ':' or a '}'");
  }

  if constexpr (is_same_v<_Ctx, __fmt_validation_format_context<_CharT>>)
  {
    __ctx.__validate_arg(__r.__value, __parse_ctx, __parse);
  }
  else
  {
    auto __arg = __ctx.arg(__r.__value);
    if constexpr (!is_void_v<_KnownType>)
    {
      if (!__arg)
      {
        ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
      }
      // All valid indices in this store select the same union member.
      __arg.__type_ = _KnownType::value;
    }
    ::cuda::std::visit_format_arg(__fmt_replacement_field_visitor<_ParseCtx, _Ctx>{__parse_ctx, __ctx, __parse}, __arg);
  }

  __begin = __parse_ctx.begin();
  if (__begin == __end || *__begin != _CharT{'}'})
  {
    ::cuda::std::__throw_format_error("The replacement field misses a terminating '}'");
  }
  return ++__begin;
}

// NVHPC warns about using __attribute__((noinline)) on an inline function.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(inline_gnu_noinline_conflict)

// We mark this function as noinline during device compilation because ptxas takes a lot of time and resources to inline
// and optimize the formatting function. We expect the function to be mostly used for debugging anyway.
template <class _KnownType = void, class _ParseCtx, class _Ctx>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_NOINLINE_DEVICE constexpr typename _Ctx::iterator
__fmt_vformat_to(_ParseCtx&& __parse_ctx, _Ctx&& __ctx)
{
  using _CharT = typename _ParseCtx::char_type;
  static_assert(is_same_v<typename _Ctx::char_type, _CharT>);

  auto __begin                     = __parse_ctx.begin();
  auto __end                       = __parse_ctx.end();
  typename _Ctx::iterator __out_it = __ctx.out();
  while (__begin != __end)
  {
    switch (*__begin)
    {
      case _CharT{'{'}:
        ++__begin;
        if (__begin == __end)
        {
          ::cuda::std::__throw_format_error("The format string terminates at a '{'");
        }

        if (*__begin != _CharT{'{'})
        {
          __ctx.advance_to(::cuda::std::move(__out_it));
          __begin  = ::cuda::std::__fmt_handle_replacement_field<_KnownType>(__begin, __end, __parse_ctx, __ctx);
          __out_it = __ctx.out();

          // The output is written and __begin points to the next character. So
          // start the next iteration.
          continue;
        }
        // The string is an escape character.
        break;

      case _CharT{'}'}:
        ++__begin;
        if (__begin == __end || *__begin != _CharT{'}'})
        {
          ::cuda::std::__throw_format_error("The format string contains an invalid escape sequence");
        }
        break;
    }

    // Copy the character to the output verbatim.
    *__out_it++ = *__begin++;
  }
  return __out_it;
}

_CCCL_DIAG_POP

template <class _OutIt, class _CharT, class _Context>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_write_format_args(_OutIt __out_it, basic_string_view<_CharT> __fmt, basic_format_args<_Context> __args)
{
  return ::cuda::std::__fmt_vformat_to(basic_format_parse_context{__fmt, __args.__size()},
                                       ::cuda::std::__fmt_make_format_context(::cuda::std::move(__out_it), __args));
}

template <class _OutIt, class _CharT, class _Context, class... _Args>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_write_format_args(
  _OutIt __out_it, basic_string_view<_CharT> __fmt, const __format_arg_store<_Context, _Args...>& __store)
{
  using _ParseCtx = basic_format_parse_context<_CharT>;
  _ParseCtx __parse_ctx{__fmt, sizeof...(_Args)};
  auto __ctx = ::cuda::std::__fmt_make_format_context(::cuda::std::move(__out_it), basic_format_args{__store});
  constexpr __fmt_arg_t __types[] = {
    ::cuda::std::__fmt_determine_arg_t<_Context, remove_const_t<_Args>>()..., __fmt_arg_t::__none};

  if constexpr (__types[0] != __fmt_arg_t::__none && __types[0] != __fmt_arg_t::__handle)
  {
    if (::cuda::std::__fmt_is_default_field(__fmt))
    {
      ::cuda::std::visit_format_arg(__fmt_replacement_field_visitor<_ParseCtx, _Context>{__parse_ctx, __ctx, false},
                                    basic_format_arg<_Context>{__types[0], __ctx.arg(0).__value_});
      return __ctx.out();
    }
  }

  // A homogeneous pack needs only one alternative of visit_format_arg. Heterogeneous packs share the erased scanner,
  // keeping the number of device implementations bounded as new combinations of argument types appear.
  constexpr bool __homogeneous =
    ((::cuda::std::__fmt_determine_arg_t<_Context, remove_const_t<_Args>>() == __types[0]) && ...);
  using _KnownType = conditional_t<__homogeneous, integral_constant<__fmt_arg_t, __types[0]>, void>;
  return ::cuda::std::__fmt_vformat_to<_KnownType>(::cuda::std::move(__parse_ctx), ::cuda::std::move(__ctx));
}

template <class _OutIt, class _CharT, class _FmtArgs>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__vformat_to_impl(_OutIt __out_it, basic_string_view<_CharT> __fmt, const _FmtArgs& __args)
{
  if constexpr (is_same_v<_OutIt, __back_insert_iterator<__fmt_output_buffer<_CharT>>>)
  {
    return ::cuda::std::__fmt_write_format_args(::cuda::std::move(__out_it), __fmt, __args);
  }
  else
  {
    __fmt_buffer_select_t<_OutIt, _CharT> __buffer{::cuda::std::move(__out_it)};
    (void) ::cuda::std::__fmt_write_format_args(__buffer.__make_output_iterator(), __fmt, __args);
    return ::cuda::std::move(__buffer).__out_it();
  }
}

_CCCL_TEMPLATE(class _OutIt)
_CCCL_REQUIRES(output_iterator<_OutIt, const char&>)
/*discard*/ _CCCL_HOST_DEVICE_API _OutIt vformat_to(_OutIt __out_it, string_view __fmt, format_args __args)
{
  return ::cuda::std::__vformat_to_impl(::cuda::std::move(__out_it), __fmt, __args);
}

#if _CCCL_HAS_WCHAR_T()
_CCCL_TEMPLATE(class _OutIt)
_CCCL_REQUIRES(output_iterator<_OutIt, const wchar_t&>)
/*discard*/ _CCCL_HOST_DEVICE_API _OutIt vformat_to(_OutIt __out_it, wstring_view __fmt, wformat_args __args)
{
  return ::cuda::std::__vformat_to_impl(::cuda::std::move(__out_it), __fmt, __args);
}
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_VFORMAT_TO_H
