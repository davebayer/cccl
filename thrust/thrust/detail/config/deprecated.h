/*
 *  Copyright 2018-2024 NVIDIA Corporation
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

/*! \file deprecated.h
 *  \brief Defines Thrust's deprecation macros
 */

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/compiler.h>
#include <thrust/detail/config/cpp_dialect.h>

#if defined(CUB_IGNORE_DEPRECATED_API) && !defined(THRUST_IGNORE_DEPRECATED_API)
#  define THRUST_IGNORE_DEPRECATED_API
#endif

#ifdef THRUST_IGNORE_DEPRECATED_API
#  define THRUST_DEPRECATED
#  define THRUST_DEPRECATED_BECAUSE(MSG)
#else
#  define THRUST_DEPRECATED              _CCCL_DEPRECATED
#  define THRUST_DEPRECATED_BECAUSE(MSG) _CCCL_DEPRECATED(MSG)
#endif
