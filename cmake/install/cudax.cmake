# CUB's WarpSpeed kernels include the experimental cooperative groups.
set(cudax_install_default "${CCCL_ENABLE_CUDAX}")
if (CUB_ENABLE_INSTALL_RULES)
  set(cudax_install_default ON)
endif()

cccl_generate_install_rules(
  cudax
  "${cudax_install_default}"
  HEADERS_SUBDIRS "include/cuda"
  HEADERS_INCLUDE "*.cuh" "group" "algorithm"
  PACKAGE
)
unset(cudax_install_default)
