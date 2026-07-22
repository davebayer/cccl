##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

include_guard(GLOBAL)

function(_cccl_get_all_archs_list out_var)
  set(${out_var} "75;80;86;87;88;89;90;100;103;107;110;120;121" PARENT_SCOPE)
endfunction()

set(
  _cccl_codegen_test_code
  [=[
import sys

def main():
  return 0

if __name__ == "__main__":
  sys.exit(main())

]=]
)

function(_cccl_codegen_get_absolute_path out_var path)
  get_filename_component(
    absolute_path
    "${path}"
    ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set(${out_var} "${absolute_path}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_read_archs out_var source)
  file(READ "${source}" source_contents)
  string(
    REGEX MATCH
    "(^|\n)[ \t]*//[ \t]*ARCHS:[ \t]*([^\r\n]*)"
    archs_match
    "${source_contents}"
  )

  if (NOT archs_match)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  set(archs "${CMAKE_MATCH_2}")
  string(REGEX REPLACE "#.*$" "" archs "${archs}")
  string(STRIP "${archs}" archs)
  string(REGEX REPLACE "[ \t,]+" ";" archs "${archs}")
  list(FILTER archs EXCLUDE REGEX "^$")

  if (NOT archs)
    message(
      FATAL_ERROR
      "Missing CUDA architectures after '// ARCHS:' in '${source}'"
    )
  endif()

  set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_get_nvcc_supported_archs out_var)
  get_property(supported_archs GLOBAL PROPERTY CCCL_CODEGEN_SUPPORTED_ARCHS)
  if (supported_archs)
    set(${out_var} "${supported_archs}" PARENT_SCOPE)
    return()
  endif()

  cccl_get_cudatoolkit()
  set(nvcc_executable "${CUDAToolkit_NVCC_EXECUTABLE}")

  execute_process(
    COMMAND "${nvcc_executable}" --list-target
    RESULT_VARIABLE list_targets_result
    OUTPUT_VARIABLE list_targets_stdout
    ERROR_VARIABLE list_targets_stderr
  )

  if (NOT list_targets_result EQUAL 0)
    message(STATUS "nvcc --list-target failed; falling back to nvcc --help.")
    execute_process(
      COMMAND "${nvcc_executable}" --help
      RESULT_VARIABLE list_targets_result
      OUTPUT_VARIABLE list_targets_stdout
      ERROR_VARIABLE list_targets_stderr
    )
  endif()

  if (NOT list_targets_result EQUAL 0)
    message(STATUS "nvcc --help failed; falling back to nvcc --list-gpu-code.")
    execute_process(
      COMMAND "${nvcc_executable}" --list-gpu-code
      RESULT_VARIABLE list_targets_result
      OUTPUT_VARIABLE list_targets_stdout
      ERROR_VARIABLE list_targets_stderr
    )
  endif()

  if (NOT list_targets_result EQUAL 0)
    message(
      FATAL_ERROR
      "Failed to query supported CUDA architectures from '${nvcc_executable}'.\n"
      "stderr:\n${list_targets_stderr}"
    )
  endif()

  string(REGEX MATCHALL "sm_[0-9]+[a-z]?" nvcc_targets "${list_targets_stdout}")
  foreach (nvcc_target IN LISTS nvcc_targets)
    string(REGEX REPLACE "^sm_" "" arch "${nvcc_target}")
    list(APPEND supported_archs "${arch}")
  endforeach()
  list(REMOVE_DUPLICATES supported_archs)
  list(SORT supported_archs COMPARE NATURAL)

  if (NOT supported_archs)
    message(
      FATAL_ERROR
      "No CUDA architectures found in '${nvcc_executable}' target list."
    )
  endif()

  set_property(
    GLOBAL
    PROPERTY CCCL_CODEGEN_SUPPORTED_ARCHS "${supported_archs}"
  )
  set(${out_var} "${supported_archs}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_get_supported_archs out_var minimum_arch)
  _cccl_codegen_get_nvcc_supported_archs(supported_archs)
  set(archs)

  foreach (arch IN LISTS supported_archs)
    string(REGEX REPLACE "^([0-9]+).*" "\\1" arch_number "${arch}")
    if (arch_number GREATER_EQUAL minimum_arch)
      list(APPEND archs "${arch}")
    endif()
  endforeach()

  set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_get_supported_f_archs_for_major out_var minimum_arch)
  _cccl_codegen_get_nvcc_supported_archs(supported_archs)
  math(EXPR family_major "${minimum_arch} / 10")
  set(archs)

  foreach (arch IN LISTS supported_archs)
    if (arch MATCHES "^([0-9]+)f$")
      set(arch_number "${CMAKE_MATCH_1}")
      math(EXPR arch_major "${arch_number} / 10")
      if (
        arch_major EQUAL family_major
        AND arch_number GREATER_EQUAL minimum_arch
      )
        list(APPEND archs "${arch}")
      endif()
    endif()
  endforeach()

  set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_get_default_archs out_var)
  _cccl_codegen_get_supported_archs(archs 75)

  if (NOT archs)
    message(
      FATAL_ERROR
      "No sm_75+ CUDA architectures found in nvcc target list."
    )
  endif()

  set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_read_code_kind out_var source)
  file(READ "${source}" source_contents)
  string(
    REGEX MATCH
    "(^|\n)[ \t]*//[ \t]*CODE:[ \t]*([^\r\n]*)"
    code_match
    "${source_contents}"
  )

  if (NOT code_match)
    message(FATAL_ERROR "Missing '// CODE:' in '${source}'")
  endif()

  set(code_kind "${CMAKE_MATCH_2}")
  string(REGEX REPLACE "#.*$" "" code_kind "${code_kind}")
  string(STRIP "${code_kind}" code_kind)
  string(REGEX REPLACE "[ \t,]+" ";" code_kind "${code_kind}")
  list(FILTER code_kind EXCLUDE REGEX "^$")

  if (NOT code_kind)
    message(
      FATAL_ERROR
      "Missing generated code language after '// CODE:' in '${source}'"
    )
  endif()

  list(LENGTH code_kind num_code_kinds)
  if (NOT num_code_kinds EQUAL 1)
    message(
      FATAL_ERROR
      "cccl_add_codegen_test requires exactly one generated code language in "
      "'// CODE:' in '${source}'. Expected 'PTX' or 'SASS'."
    )
  endif()

  string(TOUPPER "${code_kind}" normalized_code_kind)
  if (NOT normalized_code_kind MATCHES "^(PTX|SASS)$")
    message(
      FATAL_ERROR
      "Unsupported cccl_add_codegen_test code language '${code_kind}'. "
      "Expected 'PTX' or 'SASS'."
    )
  endif()

  set(${out_var} "${normalized_code_kind}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_normalize_arch out_var arch)
  string(REGEX REPLACE "-real$" "" normalized_arch "${arch}")
  if (normalized_arch MATCHES "-virtual$")
    message(
      FATAL_ERROR
      "cccl_add_codegen_test requires SASS-capable architectures; "
      "'${arch}' is virtual-only."
    )
  endif()
  string(REGEX REPLACE "^sm_?" "" normalized_arch "${normalized_arch}")

  if (NOT normalized_arch MATCHES "^[0-9]+[af]?$")
    message(
      FATAL_ERROR
      "Unsupported cccl_add_codegen_test architecture '${arch}'. "
      "Expected forms like '80', 'sm80', '90', '120f', 'sm100+', or 'sm103f+'."
    )
  endif()

  set(${out_var} "${normalized_arch}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_expand_archs out_var)
  set(expanded_archs)

  foreach (arch IN LISTS ARGN)
    if ("${arch}" MATCHES "^sm_?([0-9]+)f\\+$")
      set(minimum_arch "${CMAKE_MATCH_1}")
      _cccl_codegen_get_supported_f_archs_for_major(
        range_archs
        "${minimum_arch}"
      )

      if (NOT range_archs)
        math(EXPR family_major "${minimum_arch} / 10")
        message(
          FATAL_ERROR
          "No f CUDA architectures found for major version ${family_major} "
          "in nvcc target list."
        )
      endif()

      list(APPEND expanded_archs ${range_archs})
    elseif ("${arch}" MATCHES "^sm_?([0-9]+)\\+$")
      set(minimum_arch "${CMAKE_MATCH_1}")
      _cccl_codegen_get_supported_archs(range_archs "${minimum_arch}")

      if (NOT range_archs)
        message(
          FATAL_ERROR
          "No sm_${minimum_arch}+ CUDA architectures found in nvcc target list."
        )
      endif()

      list(APPEND expanded_archs ${range_archs})
    else()
      _cccl_codegen_normalize_arch(normalized_arch "${arch}")
      list(APPEND expanded_archs "${normalized_arch}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES expanded_archs)
  set(${out_var} "${expanded_archs}" PARENT_SCOPE)
endfunction()

function(_cccl_codegen_set_sass_arch target_name arch)
  if (arch MATCHES "[af]$")
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(
      ${target_name}
      PRIVATE "--generate-code=arch=compute_${arch},code=sm_${arch}"
    )
  else()
    set_target_properties(
      ${target_name}
      PROPERTIES CUDA_ARCHITECTURES "${arch}-real"
    )
  endif()
endfunction()

function(_cccl_codegen_set_ptx_arch target_name arch)
  if (arch MATCHES "[af]$")
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(
      ${target_name}
      PRIVATE "--generate-code=arch=compute_${arch},code=compute_${arch}"
    )
  else()
    set_target_properties(
      ${target_name}
      PROPERTIES CUDA_ARCHITECTURES "${arch}-virtual"
    )
  endif()
endfunction()

function(_cccl_codegen_has_filecheck_prefix out_var reference_contents prefix)
  if (NOT prefix MATCHES "^[A-Za-z][A-Za-z0-9_-]*$")
    message(FATAL_ERROR "Invalid FileCheck prefix '${prefix}'")
  endif()

  string(
    REGEX MATCH
    "(^|\n)[^\r\n]*${prefix}(:|-(LABEL|NOT|NEXT|SAME|DAG|COUNT|EMPTY):)"
    has_prefix
    "${reference_contents}"
  )
  if (has_prefix)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(
  _cccl_codegen_append_prefix_if_used
  check_prefixes_var
  reference_contents
  prefix
)
  _cccl_codegen_has_filecheck_prefix(
    has_prefix
    "${reference_contents}"
    "${prefix}"
  )
  if (has_prefix)
    set(check_prefixes ${${check_prefixes_var}})
    list(APPEND check_prefixes "${prefix}")
    set(${check_prefixes_var} "${check_prefixes}" PARENT_SCOPE)
  endif()
endfunction()

function(
  _cccl_codegen_get_check_prefixes
  out_var
  source
  arch
  code_kind
  user_prefixes
)
  set(check_prefixes)
  file(READ "${source}" reference_contents)

  _cccl_codegen_append_prefix_if_used(
    check_prefixes
    "${reference_contents}"
    CHECK
  )
  _cccl_codegen_append_prefix_if_used(
    check_prefixes
    "${reference_contents}"
    "${code_kind}"
  )
  _cccl_codegen_append_prefix_if_used(
    check_prefixes
    "${reference_contents}"
    SMXX
  )
  _cccl_codegen_append_prefix_if_used(
    check_prefixes
    "${reference_contents}"
    "SM${arch}"
  )

  string(REGEX MATCH "^([0-9]+)" arch_number_match "${arch}")
  set(arch_number "${CMAKE_MATCH_1}")
  if (arch_number MATCHES "^1[0-9][0-9]$")
    _cccl_codegen_append_prefix_if_used(
      check_prefixes
      "${reference_contents}"
      SM1XX
    )
  endif()

  string(REGEX MATCHALL "SM[0-9]+-PLUS" plus_prefixes "${reference_contents}")
  foreach (plus_prefix IN LISTS plus_prefixes)
    string(REGEX REPLACE "SM([0-9]+)-PLUS" "\\1" plus_arch "${plus_prefix}")
    if (arch_number GREATER_EQUAL plus_arch)
      _cccl_codegen_append_prefix_if_used(
        check_prefixes
        "${reference_contents}"
        "${plus_prefix}"
      )
    endif()
  endforeach()

  foreach (user_prefix IN LISTS user_prefixes)
    _cccl_codegen_append_prefix_if_used(
      check_prefixes
      "${reference_contents}"
      "${user_prefix}"
    )
  endforeach()

  list(REMOVE_DUPLICATES check_prefixes)
  if (NOT check_prefixes)
    message(
      FATAL_ERROR
      "No FileCheck directives for ${code_kind} sm_${arch} found in '${source}'. "
      "Use CHECK, PTX, SASS, SMXX, SM${arch}, or an applicable SM<number>-PLUS prefix."
    )
  endif()

  list(JOIN check_prefixes "," check_prefixes)
  set(${out_var} "${check_prefixes}" PARENT_SCOPE)
endfunction()

# Add a compile-only CUDA codegen test.
#
# The source may contain an arch list:
#   // ARCHS: 80 90 120f sm100+ sm103f+
#
# If no ARCHS argument or source comment is provided, all sm_75+ architectures
# supported by nvcc are checked.
#
# The source must contain a generated code language:
#   // CODE: PTX
#
# FileCheck directives are read from SOURCE. The generic check prefixes are
# CHECK, PTX, and SASS. Architecture prefixes SMXX, SM<arch>, SM1XX, and
# SM<number>-PLUS are added when the source uses them and they apply to the
# tested architecture.
function(cccl_add_codegen_test target_name)
  set(options NO_METATARGETS)
  set(oneValueArgs SOURCE)
  set(multiValueArgs ARCHS CHECK_PREFIXES FILECHECK_OPTIONS)
  cmake_parse_arguments(
    _cccl
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (_cccl_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unrecognized arguments: ${_cccl_UNPARSED_ARGUMENTS}")
  endif()

  if (NOT DEFINED _cccl_SOURCE)
    message(FATAL_ERROR "cccl_add_codegen_test requires SOURCE argument")
  endif()

  if (NOT CCCL_ENABLE_CODEGEN_TESTING)
    return()
  endif()

  cccl_get_filecheck()

  _cccl_codegen_get_absolute_path(source "${_cccl_SOURCE}")
  if (NOT EXISTS "${source}")
    message(
      FATAL_ERROR
      "cccl_add_codegen_test source does not exist: ${source}"
    )
  endif()

  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")

  if (DEFINED _cccl_ARCHS)
    set(archs ${_cccl_ARCHS})
  else()
    _cccl_codegen_read_archs(archs "${source}")
    if (NOT archs)
      _cccl_codegen_get_default_archs(archs)
    endif()
  endif()
  _cccl_codegen_expand_archs(archs ${archs})
  if (NOT archs)
    message(
      FATAL_ERROR
      "cccl_add_codegen_test requires at least one CUDA architecture"
    )
  endif()
  _cccl_codegen_read_code_kind(code_kind "${source}")

  if (code_kind STREQUAL "SASS")
    cccl_get_cuobjdump()

    file(READ "${source}" source_contents)
    string(FIND "${source_contents}" "__device__" has_device_function)
    if (NOT has_device_function EQUAL -1)
      set(needs_relocatable_device_code ON)
    endif()
  endif()

  add_custom_target(${target_name})
  if (COMMAND cccl_ensure_metatargets AND NOT _cccl_NO_METATARGETS)
    cccl_ensure_metatargets(${target_name})
  endif()

  string(TOLOWER "${code_kind}" code_kind_lower)

  foreach (arch IN LISTS archs)
    _cccl_codegen_normalize_arch(normalized_arch "${arch}")

    set(codegen_target "${target_name}.sm${normalized_arch}.${code_kind_lower}")
    add_library(${codegen_target} OBJECT "${source}")
    add_dependencies(${target_name} ${codegen_target})

    cccl_configure_target(${codegen_target})
    if (code_kind STREQUAL "SASS")
      _cccl_codegen_set_sass_arch(${codegen_target} "${normalized_arch}")
      set_target_properties(
        ${codegen_target}
        PROPERTIES CUDA_CUBIN_COMPILATION ON
      )
      if (needs_relocatable_device_code)
        target_compile_options(${codegen_target} PRIVATE "-dc")
      endif()
    elseif (code_kind STREQUAL "PTX")
      _cccl_codegen_set_ptx_arch(${codegen_target} "${normalized_arch}")
      set_target_properties(
        ${codegen_target}
        PROPERTIES CUDA_PTX_COMPILATION ON
      )
    endif()
    target_compile_options(${codegen_target} PRIVATE "-Wno-comment")

    _cccl_codegen_get_check_prefixes(
      check_prefixes
      "${source}"
      "${normalized_arch}"
      "${code_kind}"
      "${_cccl_CHECK_PREFIXES}"
    )

    set(test_name "${codegen_target}")
    set(
      dump_file
      "${CMAKE_CURRENT_BINARY_DIR}/${codegen_target}.${code_kind_lower}"
    )
    add_test(
      NAME ${test_name}
      # gersemi: off
      COMMAND
        "${CMAKE_COMMAND}"
          "-DCODE_KIND=${code_kind}"
          "-DCUOBJDUMP_EXECUTABLE=${CCCL_CUOBJDUMP_EXECUTABLE}"
          "-DFILECHECK_EXECUTABLE=${CCCL_FILECHECK_EXECUTABLE}"
          "-DCODEGEN_FILE=$<TARGET_OBJECTS:${codegen_target}>"
          "-DSOURCE_FILE=${source}"
          "-DCHECK_PREFIXES=${check_prefixes}"
          "-DDUMP_FILE=${dump_file}"
          "-DFILECHECK_OPTIONS=${_cccl_FILECHECK_OPTIONS}"
          -P "${CCCL_SOURCE_DIR}/cmake/CCCLRunCodegenTest.cmake"
      # gersemi: on
    )
    set_tests_properties(${test_name} PROPERTIES LABELS codegen)
  endforeach()
endfunction()
