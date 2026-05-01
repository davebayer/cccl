//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H
#define _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2020

#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__type_traits/is_constructible.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/integer_sequence.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct __cw_fixed_value
{
  using __type _CCCL_NODEBUG_ALIAS = _Tp;
  consteval __cw_fixed_value(__type __v) noexcept
      : __data(__v)
  {}
  _Tp __data;
};

template <class _Tp, size_t _Extent>
struct __cw_fixed_value<_Tp[_Extent]>
{
  using __type _CCCL_NODEBUG_ALIAS = _Tp[_Extent];
  _Tp __data[_Extent];

  consteval __cw_fixed_value(_Tp (&__arr)[_Extent]) noexcept
      : __cw_fixed_value(__arr, make_index_sequence<_Extent>{})
  {}

private:
  template <size_t... _Idxs>
  consteval __cw_fixed_value(_Tp (&__arr)[_Extent], index_sequence<_Idxs...>) noexcept
      : __data{__arr[_Idxs]...}
  {}
};

template <class _Tp, size_t _Extent>
_CCCL_HOST_DEVICE __cw_fixed_value(_Tp (&)[_Extent]) -> __cw_fixed_value<_Tp[_Extent]>;

template <__cw_fixed_value _Xp,
#  if _CCCL_COMPILER(GCC)
          // gcc bug:  https://gcc.gnu.org/PR117392
          class = typename decltype(__cw_fixed_value(_Xp))::__type
#  else
          class = typename decltype(_Xp)::__type
#  endif
          >
struct constant_wrapper;

template <class _Tp>
concept __constexpr_param = requires { typename constant_wrapper<_Tp::value>; };

template <__cw_fixed_value _Xp>
constexpr auto cw = constant_wrapper<_Xp>{};

struct __cw_operators
{
  // unary operators
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator+(_Tp) noexcept -> constant_wrapper<(+_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator-(_Tp) noexcept -> constant_wrapper<(-_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator~(_Tp) noexcept -> constant_wrapper<(~_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator!(_Tp) noexcept -> constant_wrapper<(!_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator&(_Tp) noexcept -> constant_wrapper<(&_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator*(_Tp) noexcept -> constant_wrapper<(*_Tp::value)>
  {
    return {};
  }

  // binary operators
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator+(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value + _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator-(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value - _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator*(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value * _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator/(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value / _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator%(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value % _Rp::value)>
  {
    return {};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator<<(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value << _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator>>(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value >> _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator&(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value & _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator|(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value | _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator^(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value ^ _Rp::value)>
  {
    return {};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires(!is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>)
  [[nodiscard]] _CCCL_API friend constexpr auto operator&&(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value && _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires(!is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>)
  [[nodiscard]] _CCCL_API friend constexpr auto operator||(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value || _Rp::value)>
  {
    return {};
  }

  // comparisons
  // template <__constexpr_param _Lp, __constexpr_param _Rp>
  // _CCCL_API friend constexpr auto operator<=>(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value <=> _Rp::value)>
  // {
  //   return {};
  // }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _CCCL_API friend constexpr auto operator<(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value < _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _CCCL_API friend constexpr auto operator<=(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value <= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _CCCL_API friend constexpr auto operator==(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value == _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _CCCL_API friend constexpr auto operator!=(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value != _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _CCCL_API friend constexpr auto operator>(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value > _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _CCCL_API friend constexpr auto operator>=(_Lp, _Rp) noexcept -> constant_wrapper<(_Lp::value >= _Rp::value)>
  {
    return {};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  friend auto operator,(_Lp, _Rp) = delete;

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API friend constexpr auto operator->*(_Lp, _Rp) noexcept
    -> constant_wrapper<(_Lp::value->*_Rp::value)>
  {
    return {};
  }

  // pseudo-mutators
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API constexpr auto operator++() const noexcept -> constant_wrapper<(++_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API constexpr auto operator++(int) const noexcept -> constant_wrapper<(_Tp::value++)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API constexpr auto operator--() const noexcept -> constant_wrapper<(--_Tp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _CCCL_API constexpr auto operator--(int) const noexcept -> constant_wrapper<(_Tp::value--)>
  {
    return {};
  }

  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator+=(_Rp) const noexcept -> constant_wrapper<(_Tp::value += _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator-=(_Rp) const noexcept -> constant_wrapper<(_Tp::value -= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator*=(_Rp) const noexcept -> constant_wrapper<(_Tp::value *= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator/=(_Rp) const noexcept -> constant_wrapper<(_Tp::value /= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator%=(_Rp) const noexcept -> constant_wrapper<(_Tp::value %= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator&=(_Rp) const noexcept -> constant_wrapper<(_Tp::value &= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator|=(_Rp) const noexcept -> constant_wrapper<(_Tp::value |= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator^=(_Rp) const noexcept -> constant_wrapper<(_Tp::value ^= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator<<=(_Rp) const noexcept
    -> constant_wrapper<(_Tp::value <<= _Rp::value)>
  {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator>>=(_Rp) const noexcept
    -> constant_wrapper<(_Tp::value >>= _Rp::value)>
  {
    return {};
  }
};

template <const auto& _Callable, class... _Args>
concept __constexpr_callable = (__constexpr_param<remove_cvref_t<_Args>> && ...) && requires {
  typename constant_wrapper<::cuda::std::invoke(_Callable, remove_cvref_t<_Args>::value...)>;
};

#  if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
// template <const auto& _Obj, class... _Args>
// concept __constexpr_indexable = (__constexpr_param<remove_cvref_t<_Args>> && ...) && requires {
//   typename constant_wrapper<_Obj[remove_cvref_t<_Args>::value...]>;
// };
#  else
template <const auto& _Obj, class _Arg>
concept __constexpr_indexable = (__constexpr_param<remove_cvref_t<_Arg>>) && requires {
  typename constant_wrapper<_Obj[remove_cvref_t<_Arg>::value]>;
};
#  endif

template <class _Tp, _Tp _Value>
struct __cw_fixed_value_storage
{
  _Tp __data = _Value;
};

template <class _Tp, _Tp... _Values>
struct __cw_fixed_value_storage_array
{
  _Tp __data[sizeof...(_Values)]{_Values...};
};

template <auto _Value>
_CCCL_GLOBAL_CONSTANT auto __cw_storage = _Value;

template <class _Tp, _Tp... _Values>
_CCCL_GLOBAL_CONSTANT _Tp __cw_storage_array[]{_Values...};

template <auto _Value, size_t... _Is>
[[nodiscard]] consteval const auto& __make_cw_fixed_value_storage_array_helper(index_sequence<_Is...>) noexcept
{
  using _Tp = remove_cvref_t<remove_extent_t<decltype(_Value.__data[0])>>;
  return __cw_storage_array<_Tp, _Value.__data[_Is]...>;
}

template <auto _Value>
[[nodiscard]] consteval const auto& __make_fixed_value_storage() noexcept
{
  using _FixedValue = decltype(_Value);
  using _Tp         = typename _FixedValue::__type;

  if constexpr (is_array_v<_Tp>)
  {
    return ::cuda::std::__make_cw_fixed_value_storage_array_helper<_Value>(make_index_sequence<extent_v<_Tp>>{});
  }
  else
  {
    return __cw_storage<_Value>;
  }
}

template <class _Tp, class _Up>
consteval auto __cw_assign()
{
  return _Tp::value = _Up::value;
}

template <__cw_fixed_value _Xp, class>
struct constant_wrapper : __cw_operators
{
  static constexpr const auto& value = ::cuda::std::__make_fixed_value_storage<_Xp>();
  using type                         = constant_wrapper;
  using value_type                   = decltype(_Xp)::__type;

  template <__constexpr_param _Rp>
  [[nodiscard]] _CCCL_API constexpr auto operator=(_Rp) const noexcept
    -> constant_wrapper<__cw_assign<constant_wrapper, _Rp>>
  {
    return {};
  }

  _CCCL_API constexpr operator decltype(value)() const noexcept
  {
    return value;
  }

  template <class... _Args>
    requires __constexpr_callable<value, _Args...>
  [[nodiscard]]
  _CCCL_API constexpr constant_wrapper<::cuda::std::invoke(value, remove_cvref_t<_Args>::value...)>
  operator()(_Args&&...) const noexcept
  {
    return {};
  }

  template <class... _Args>
    requires(!__constexpr_callable<value, _Args...> && is_invocable_v<const value_type&, _Args && ...>)
  _CCCL_API constexpr decltype(auto) operator()(_Args&&... __args) const
    noexcept(noexcept(::cuda::std::invoke(value, ::cuda::std::forward<_Args>(__args)...)))
  {
    return ::cuda::std::invoke(value, ::cuda::std::forward<_Args>(__args)...);
  }
#  if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  template <class... _Args>
    requires __constexpr_indexable<value, _Args...>
  [[nodiscard]]
  _CCCL_API constexpr constant_wrapper<value[remove_cvref_t<_Args>::value...]> operator[](_Args&&...) const noexcept
  {
    return {};
  }

  template <class... _Args>
    requires(!__constexpr_indexable<value, _Args...> && requires { value[::cuda::std::declval<_Args>()...]; })
  _CCCL_API constexpr decltype(auto) operator[](_Args&&... __args) const
    noexcept(noexcept(value[::cuda::std::forward<_Args>(__args)...]))
  {
    return value[::cuda::std::forward<_Args>(__args)...];
  }
#  else
  template <class _Arg>
    requires __constexpr_indexable<value, _Arg>
  [[nodiscard]]
  _CCCL_API constexpr constant_wrapper<value[remove_cvref_t<_Arg>::value]> operator[](_Arg&&) const noexcept
  {
    return {};
  }

  template <class _Arg>
    requires(!__constexpr_indexable<value, _Arg> && requires { value[::cuda::std::declval<_Arg>()]; })
  _CCCL_API constexpr decltype(auto) operator[](_Arg&& __arg) const
    noexcept(noexcept(value[::cuda::std::forward<_Arg>(__arg)]))
  {
    return value[::cuda::std::forward<_Arg>(__arg)];
  }
#  endif
};

_CCCL_END_NAMESPACE_CUDA_STD

#endif // _CCCL_STD_VER >= 2020

#endif // _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H
