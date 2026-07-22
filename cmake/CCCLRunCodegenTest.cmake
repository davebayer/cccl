##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

foreach (
  required_var
  IN
  ITEMS
    CODE_KIND
    FILECHECK_EXECUTABLE
    CODEGEN_FILE
    SOURCE_FILE
    CHECK_PREFIXES
    DUMP_FILE
)
  if (NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "CCCLRunCodegenTest.cmake requires ${required_var}")
  endif()
endforeach()

get_filename_component(dump_dir "${DUMP_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${dump_dir}")

if (CODE_KIND STREQUAL "SASS")
  if (NOT DEFINED CUOBJDUMP_EXECUTABLE OR "${CUOBJDUMP_EXECUTABLE}" STREQUAL "")
    message(
      FATAL_ERROR
      "CCCLRunCodegenTest.cmake requires CUOBJDUMP_EXECUTABLE"
    )
  endif()

  execute_process(
    COMMAND "${CUOBJDUMP_EXECUTABLE}" --dump-sass "${CODEGEN_FILE}"
    RESULT_VARIABLE cuobjdump_result
    OUTPUT_FILE "${DUMP_FILE}"
    ERROR_VARIABLE cuobjdump_stderr
  )

  if (NOT cuobjdump_result EQUAL 0)
    message(
      FATAL_ERROR
      "cuobjdump failed for '${CODEGEN_FILE}' with exit code ${cuobjdump_result}:\n${cuobjdump_stderr}"
    )
  endif()
elseif (CODE_KIND STREQUAL "PTX")
  configure_file("${CODEGEN_FILE}" "${DUMP_FILE}" COPYONLY)
else()
  message(FATAL_ERROR "Unsupported code kind '${CODE_KIND}'")
endif()

set(filecheck_options --match-full-lines)
if (DEFINED FILECHECK_OPTIONS AND NOT "${FILECHECK_OPTIONS}" STREQUAL "")
  list(APPEND filecheck_options ${FILECHECK_OPTIONS})
endif()

execute_process(
  COMMAND
    "${FILECHECK_EXECUTABLE}" ${filecheck_options}
    "--check-prefixes=${CHECK_PREFIXES}" "${SOURCE_FILE}"
  INPUT_FILE "${DUMP_FILE}"
  RESULT_VARIABLE filecheck_result
  OUTPUT_VARIABLE filecheck_stdout
  ERROR_VARIABLE filecheck_stderr
)

if (NOT filecheck_result EQUAL 0)
  message(
    FATAL_ERROR
    "FileCheck failed for '${SOURCE_FILE}' with exit code ${filecheck_result}.\n"
    "Generated code: ${DUMP_FILE}\n"
    "stdout:\n${filecheck_stdout}\n"
    "stderr:\n${filecheck_stderr}"
  )
endif()
