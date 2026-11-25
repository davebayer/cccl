# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.

cccl_get_cudatoolkit()

# Meta target for all configs' header builds:
add_custom_target(cudax.all.headers)

function(cudax_tidy_headers target)
  if (CCCL_ENABLE_CLANG_TIDY)
    get_target_property(sources ${target} SOURCES)
    list(FILTER sources EXCLUDE REGEX "/__cuco/")
    cccl_tidy_add_target(SOURCES ${sources})
  endif()
endfunction()

function(cudax_add_header_test label definitions)
  ###################
  # Non-STF headers #
  set(headertest_target cudax.headers.${label}.no_stf)
  cccl_generate_header_tests(
    ${headertest_target}
    cudax/include
    # The cudax header template removes the check for the `small` macro.
    HEADER_TEMPLATE "${cudax_SOURCE_DIR}/cmake/header_test.in.cu"
    GLOBS
      "cuda/experimental/*.cuh"
      # These trees also provide .h and extensionless headers.
      "cuda/experimental/coop/*"
      "cuda/experimental/mgmn/*"
    EXCLUDES
      # The following internal headers are not required to compile independently:
      "cuda/experimental/__execution/prologue.cuh"
      "cuda/experimental/__execution/epilogue.cuh"
      # cuFile headers are compiled separately:
      "cuda/experimental/cufile.cuh"
      "cuda/experimental/__cufile/*"
      # Places headers are compiled separately:
      "cuda/experimental/places.cuh"
      "cuda/experimental/__places/*"
      # STF headers are compiled separately:
      "cuda/experimental/stf.cuh"
      "cuda/experimental/__stf/*"
  )
  target_link_libraries(${headertest_target} PUBLIC cudax.compiler_interface)
  cudax_tidy_headers(${headertest_target})

  if (cudax_ENABLE_CUFILE)
    ###############
    # cuFile headers #
    set(headertest_target cudax.headers.${label}.cufile)
    cccl_generate_header_tests(
      ${headertest_target}
      cudax/include
      HEADER_TEMPLATE "${cudax_SOURCE_DIR}/cmake/header_test.in.cu"
      GLOBS #
        "cuda/experimental/cufile.cuh"
        "cuda/experimental/__cufile/*.cuh"
    )
    target_link_libraries(${headertest_target} PUBLIC cudax.compiler_interface)
    cudax_tidy_headers(${headertest_target})
  endif()

  # FIXME: Enable MSVC
  if (cudax_ENABLE_PLACES AND NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    ##################
    # Places headers #
    set(headertest_target cudax.headers.${label}.places)
    cccl_generate_header_tests(
      ${headertest_target}
      cudax/include
      GLOBS #
        "cuda/experimental/places.cuh"
        "cuda/experimental/__places/*.cuh"
      HEADER_TEMPLATE "${cudax_SOURCE_DIR}/cmake/header_test.in.cu"
    )
    target_link_libraries(${headertest_target} PUBLIC cudax.compiler_interface)
    cudax_tidy_headers(${headertest_target})
    target_compile_options(
      ${headertest_target}
      PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
    )
  endif()

  # FIXME: Enable MSVC
  if (cudax_ENABLE_CUDASTF AND NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    ###############
    # STF headers #
    set(headertest_target cudax.headers.${label}.stf)
    cccl_generate_header_tests(
      ${headertest_target}
      cudax/include
      GLOBS #
        "cuda/experimental/stf.cuh"
        "cuda/experimental/__stf/*.cuh"
      # FIXME: The cudax header template removes the check for the `small` macro.
      # cuda/experimental/__stf/utility/memory.cuh defines functions named `small`.
      # These should be renamed to avoid conflicts with windows system headers, and
      # the following line removed:
      HEADER_TEMPLATE "${cudax_SOURCE_DIR}/cmake/header_test.in.cu"
    )
    target_link_libraries(
      ${headertest_target}
      PUBLIC cudax.compiler_interface CUDA::cuda_driver
    )
    target_compile_options(
      ${headertest_target}
      PRIVATE
        # Required by stf headers:
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
        # FIXME: We should be able to refactor away from needing this by
        # using _CCCL_HOST_DEVICE and friends + `::cuda::std` utilities where
        # necessary.
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
    )
  endif()
endfunction()

cudax_add_header_test(basic "")
