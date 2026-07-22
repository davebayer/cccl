##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.simd.ptx)
add_custom_target(libcudacxx.test.simd.sass)

if (NOT CCCL_ENABLE_CODEGEN_TESTING)
  return()
endif()

# GCC7 and NVCC 12.0 do not vectorize load/store.
if (
  CMAKE_CXX_COMPILER_ID STREQUAL GNU
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8
)
  message("-- Skipping simd codegen tests with GCC 7 host compiler")
  return()
endif()

if (
  CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1
)
  message("-- Skipping simd codegen tests with NVCC 12.0")
  return()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
  message("-- Skipping simd codegen tests with NVHPC host compiler")
  return()
endif()

if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  message("-- clang-cuda does not support the -dc simd codegen tests")
  return()
endif()

set_directory_properties(
  PROPERTIES INCLUDE_DIRECTORIES "${libcudacxx_SOURCE_DIR}/include"
)

set(simd_codegen_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")

set(simd_codegen_ptx_cuda_archs 80 90)
if (
  CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
  AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL Clang
)
  list(APPEND simd_codegen_ptx_cuda_archs 100 120)
endif()

set(simd_codegen_sass_cuda_archs 80 90)
if (
  CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8
  AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL Clang
)
  list(APPEND simd_codegen_sass_cuda_archs 100 120)
endif()

if (
  CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
  AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL Clang
)
  list(APPEND simd_codegen_sass_cuda_archs 103 120f)
endif()

function(simd_codegen_get_target_name out_var kind test_path)
  file(
    RELATIVE_PATH
    relative_test_path
    "${simd_codegen_source_dir}"
    "${test_path}"
  )
  cmake_path(
    REMOVE_EXTENSION relative_test_path
    LAST_ONLY
    OUTPUT_VARIABLE test_name
  )
  string(REGEX REPLACE "[^A-Za-z0-9_]+" "." test_name "${test_name}")
  set(${out_var} "libcudacxx.test.simd.${kind}.${test_name}" PARENT_SCOPE)
endfunction()

function(simd_codegen_get_sass_check_prefixes out_var test_contents arch)
  set(check_prefixes SMXX)
  set(arch_prefix "SM${arch}")

  string(
    REGEX MATCH
    "; ${arch_prefix}(:|-LABEL:|-NOT:|-NEXT:|-SAME:|-DAG:|-COUNT:|-EMPTY:)"
    has_arch_prefix
    "${test_contents}"
  )
  if (has_arch_prefix)
    list(APPEND check_prefixes "${arch_prefix}")
  endif()

  if (arch MATCHES "^1[0-9][0-9][af]?$")
    string(
      REGEX MATCH
      "; SM1XX(:|-LABEL:|-NOT:|-NEXT:|-SAME:|-DAG:|-COUNT:|-EMPTY:)"
      has_sm1xx_prefix
      "${test_contents}"
    )
    if (has_sm1xx_prefix)
      list(APPEND check_prefixes SM1XX)
    endif()
  endif()

  string(
    REGEX MATCHALL
    "; SM[0-9]+-PLUS(:|-[A-Z]+:)"
    plus_prefixes
    "${test_contents}"
  )
  string(REGEX REPLACE "^([0-9]+).*" "\\1" arch_number "${arch}")
  foreach (plus_prefix IN LISTS plus_prefixes)
    string(REGEX REPLACE ".*SM([0-9]+)-PLUS.*" "\\1" plus_arch "${plus_prefix}")
    if (arch_number GREATER_EQUAL plus_arch)
      list(APPEND check_prefixes "SM${plus_arch}-PLUS")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES check_prefixes)

  set(${out_var} "${check_prefixes}" PARENT_SCOPE)
endfunction()

function(simd_codegen_has_sass_arch_specific_checks out_var test_contents)
  string(
    REGEX MATCH
    "; SM([0-9]+[a-f]?|1XX)(:|-LABEL:|-NOT:|-NEXT:|-SAME:|-DAG:|-COUNT:|-EMPTY:)"
    has_arch_specific_checks
    "${test_contents}"
  )
  string(
    REGEX MATCH
    "; SM[0-9]+-PLUS(:|-[A-Z]+:)"
    has_plus_checks
    "${test_contents}"
  )
  if (has_arch_specific_checks OR has_plus_checks)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(simd_codegen_add_ptx_test test_path)
  simd_codegen_get_target_name(target_name ptx "${test_path}")
  cccl_add_codegen_test(
    ${target_name}
    SOURCE "${test_path}"
    ARCHS ${simd_codegen_ptx_cuda_archs}
    CHECK_PREFIXES SMXXX
    NO_METATARGETS
  )
  if (TARGET ${target_name})
    add_dependencies(libcudacxx.test.simd.ptx ${target_name})
  endif()
endfunction()

function(simd_codegen_add_sass_test test_path)
  file(READ "${test_path}" test_contents)

  set(test_archs)
  foreach (arch IN LISTS simd_codegen_sass_cuda_archs)
    simd_codegen_get_sass_check_prefixes(
      check_prefixes
      "${test_contents}"
      "${arch}"
    )
    if (NOT "${check_prefixes}" STREQUAL "SMXX")
      list(APPEND test_archs "${arch}")
    endif()
  endforeach()

  if (NOT test_archs)
    simd_codegen_has_sass_arch_specific_checks(
      has_arch_specific_checks
      "${test_contents}"
    )
    if (has_arch_specific_checks)
      message(
        STATUS
        "-- Skipping ${test_path}: requires an unsupported CUDA architecture"
      )
      return()
    endif()
    set(test_archs ${simd_codegen_sass_cuda_archs})
  endif()

  simd_codegen_get_target_name(target_name sass "${test_path}")
  cccl_add_codegen_test(
    ${target_name}
    SOURCE "${test_path}"
    ARCHS ${test_archs}
    NO_METATARGETS
  )
  if (TARGET ${target_name})
    add_dependencies(libcudacxx.test.simd.sass ${target_name})
  endif()
endfunction()

set(simd_codegen_sass_tests)
file(GLOB simd_codegen_sass_tests "floating_point/*.cu" "integer/*.cu")

add_subdirectory(load_store)

foreach (test_path IN LISTS simd_codegen_sass_tests)
  simd_codegen_add_sass_test("${test_path}")
endforeach()
