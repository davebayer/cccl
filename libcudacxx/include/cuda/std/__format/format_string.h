//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_STRING_H
#define _CUDA_STD___FORMAT_FORMAT_STRING_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__format/dynamic_format.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_arg_store.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__format/validation.h>
#include <cuda/std/__format/vformat_to.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_string
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(convertible_to<const _Tp&, basic_string_view<_CharT>>)
  _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL basic_format_string(const _Tp& __str)
      : __str_{__str}
  {
    if constexpr (sizeof...(_Args) > 0)
    {
      constexpr __fmt_arg_t __types[] = {::cuda::std::__fmt_determine_arg_t<_FmtContext, remove_cvref_t<_Args>>()...};
      // Built-in formatters accept an empty spec. A custom formatter must still get to parse it.
      if constexpr (__types[0] != __fmt_arg_t::__none && __types[0] != __fmt_arg_t::__handle)
      {
        if (::cuda::std::__fmt_is_default_field(__str_))
        {
          return;
        }
      }

      constexpr typename _FmtContext::__parse_function __parsers[] = {
        ::cuda::std::__fmt_select_validation_parser<_FmtContext, remove_cvref_t<_Args>>()...};

      (void) ::cuda::std::__fmt_vformat_to(basic_format_parse_context<_CharT>{__str_, sizeof...(_Args)},
                                           _FmtContext{__types, __parsers, sizeof...(_Args)});
    }
    else
    {
      (void) ::cuda::std::__fmt_vformat_to(
        basic_format_parse_context<_CharT>{__str_, sizeof...(_Args)}, _FmtContext{nullptr, nullptr, sizeof...(_Args)});
    }
  }

  _CCCL_HOST_DEVICE_API constexpr basic_format_string(__dynamic_format_string<_CharT> __s) noexcept
      : __str_{__s.__get()}
  {}

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_string_view<_CharT> get() const noexcept
  {
    return __str_;
  }

private:
  basic_string_view<_CharT> __str_;

  using _FmtContext _CCCL_NODEBUG = __fmt_validation_format_context<_CharT>;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_STRING_H
