##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.hierarchy.sass)

if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
  message(STATUS "Skipping hierarchy SASS codegen tests without NVCC")
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(hierarchy_codegen_cuda_archs SASS)
libcudacxx_codegen_add_sass_tests(
  AGGREGATE_TARGET libcudacxx.test.hierarchy.sass
  TARGET_PREFIX hierarchy_codegen
  ARCHITECTURES ${hierarchy_codegen_cuda_archs}
  TESTS "${CMAKE_CURRENT_SOURCE_DIR}/cluster.cu" "${CMAKE_CURRENT_SOURCE_DIR}/queries.cu"
)
