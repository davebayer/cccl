//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_ENTRY_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_ENTRY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/call_or.h>
#include <cuda/__functional/lazy_call_or.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__fwd/optional.h>
#include <cuda/std/__type_traits/always_false.h>

#include <cuda/experimental/__coop/scratch.cuh>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

_CCCL_BEGIN_NV_DIAG_SUPPRESS(342) // static call operator in earlier standard modes

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(static_member_operator_not_allowed)

namespace cuda::experimental::coop
{
template <bool _Broadcasted, class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn, class = void>
struct __reduce_impl
{
  static_assert(::cuda::std::__always_false_v<_Group>, "cudax::coop::reduce is not supported for the group");
};

_CCCL_BEGIN_NAMESPACE_CPO(__reduce)
struct __fn
{
  template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  __get_scratch_requirements(const _Group& __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn) noexcept
  {
    using _Impl = __reduce_impl<false, _Group, _Tp, _Np, _RedFn>;
    return __scratch_reqs<typename _Impl::_SmemScratch, typename _Impl::_GmemScratch>{};
  }

  template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  __get_scratch_requirements(broadcasted_t, const _Group& __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn) noexcept
  {
    using _Impl = __reduce_impl<true, _Group, _Tp, _Np, _RedFn>;
    return __scratch_reqs<typename _Impl::_SmemScratch, typename _Impl::_GmemScratch>{};
  }

  template <class _Impl, class _Env>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto __make_impl(_Env __env)
  {
    using _SmemScratch = typename _Impl::_SmemScratch;
    using _GmemScratch = typename _Impl::_GmemScratch;

    // Check that environment's smem and gmem scratch match the expected type.
    if constexpr (::cuda::std::execution::__queryable_with<_Env, __get_smem_scratch_t>)
    {
      using _QueryResult =
        ::cuda::std::remove_cvref_t<::cuda::std::execution::__query_result_t<_Env, __get_smem_scratch_t>>;
      using _EnvSmemScratch = typename _QueryResult::type;
      static_assert(::cuda::std::is_same_v<_EnvSmemScratch, _SmemScratch>, "Invalid shared memory scratch passed");
    }
    if constexpr (::cuda::std::execution::__queryable_with<_Env, __get_gmem_scratch_t>)
    {
      using _QueryResult =
        ::cuda::std::remove_cvref_t<::cuda::std::execution::__query_result_t<_Env, __get_gmem_scratch_t>>;
      using _EnvGmemScratch = typename _QueryResult::type;
      static_assert(::cuda::std::is_same_v<_EnvGmemScratch, _GmemScratch>, "Invalid global memory scratch passed");
    }
    else
    {
      static_assert(::cuda::std::is_same_v<_GmemScratch, __empty_gmem_scratch>,
                    "Algorithm can't allocate global memory for scratch by itself");
    }

    // Extract environment's scratch or allocate default scratch.
    auto& __smem_scratch =
      ::cuda::__lazy_call_or(
        __get_smem_scratch,
        [&]() {
          return ::cuda::experimental::coop::__make_smem_scratch<_SmemScratch>();
        },
        __env)
        .get();
    auto& __gmem_scratch =
      ::cuda::__call_or(__get_gmem_scratch, ::cuda::std::reference_wrapper{__empty_gmem_scratch_obj}, __env).get();

    return _Impl{__smem_scratch, __gmem_scratch};
  }

  template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
  _CCCL_STATIC_CALL_OPERATOR(const _Group& __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
  {
    using _Impl = __reduce_impl<false, _Group, _Tp, _Np, _RedFn>;
    return __make_impl<_Impl>(::cuda::std::execution::env{})(__group, __thread_data, __red_fn);
  }

  template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
  [[nodiscard]] _CCCL_DEVICE_API _Tp
  _CCCL_STATIC_CALL_OPERATOR(const broadcasted_t&, const _Group& __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
  {
    using _Impl = __reduce_impl<true, _Group, _Tp, _Np, _RedFn>;
    return __make_impl<_Impl>(::cuda::std::execution::env{})(__group, __thread_data, __red_fn);
  }
};
_CCCL_END_NAMESPACE_CPO

_CCCL_GLOBAL_CONSTANT auto reduce = __reduce::__fn{};
} // namespace cuda::experimental::coop

_CCCL_DIAG_POP

_CCCL_END_NV_DIAG_SUPPRESS()

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_ENTRY_CUH
