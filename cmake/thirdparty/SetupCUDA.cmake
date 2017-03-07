################################
# CUDA
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(CUDA REQUIRED)

message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")

blt_register_library(
  NAME CUDA
  INCLUDES ${CUDA_INCLUDE_DIRECTORIES}
  LIBRARIES ${CUDA_INCLUDE_DIRECTORIES})
