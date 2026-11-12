cccl_generate_install_rules(
  CUB
  ${CCCL_TOPLEVEL_PROJECT}
  HEADERS_INCLUDE "*.cuh"
  PACKAGE
)

if (CUB_ENABLE_INSTALL_RULES)
  # Cooperative groups are an implementation dependency of CUB's lookahead
  # kernels, including in installations without the experimental libraries.
  set(coop_source_dir "${CCCL_SOURCE_DIR}/cudax/include/cuda/experimental/coop")
  set(coop_dest_dir "${CMAKE_INSTALL_INCLUDEDIR}/cuda/experimental/coop")
  install(DIRECTORY "${coop_source_dir}/__group" DESTINATION "${coop_dest_dir}")
  install(FILES "${coop_source_dir}/group" DESTINATION "${coop_dest_dir}")
endif()
