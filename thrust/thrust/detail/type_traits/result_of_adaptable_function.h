/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_traits.h>

#include <cuda/std/__type_traits/void_t.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// Sets `type` to the result of the specified Signature invocation. If the callable defines a `result_type` alias
// member, that type is used instead. Use invoke_result / result_of when FuncType::result_type is not defined.
template <typename Signature, typename Enable = void>
struct result_of_adaptable_function
{
private:
  template <typename Sig>
  struct impl;

  template <typename F, typename... Args>
  struct impl<F(Args...)>
  {
    using type = invoke_result_t<F, Args...>;
  };

public:
  using type = typename impl<Signature>::type;
};

// TODO(bgruber): remove this specialization eventually
// specialization for invocations which define result_type
_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename Functor, typename... ArgTypes>
struct result_of_adaptable_function<Functor(ArgTypes...), ::cuda::std::void_t<typename Functor::result_type>>
{
  using type = typename Functor::result_type;
};
_CCCL_SUPPRESS_DEPRECATED_POP

} // namespace detail
THRUST_NAMESPACE_END
