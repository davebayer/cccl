##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

set(
  simd_codegen_load_store_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i8.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i32.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_bf16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f32.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i64.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f64.cu"
)

foreach (test_path IN LISTS simd_codegen_load_store_tests)
  simd_codegen_add_ptx_test("${test_path}")
endforeach()
