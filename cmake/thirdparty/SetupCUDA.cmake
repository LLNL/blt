################################
# CUDA
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(CUDA REQUIRED)

message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")

# don't propagate host flags - too easy to break stuff!
set (CUDA_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
  set (CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else ()
  set (CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
endif ()

# depend on 'cuda', if you need to use cuda
# headers, link to cuda libs, and need to run your source
# through a cuda compiler (nvcc)
blt_register_library(NAME cuda
                     INCLUDES ${CUDA_INCLUDE_DIRS}
                     LIBRARIES ${CUDA_LIBRARIES}
                     DEFINES USE_CUDA)

# depend on 'cuda_runtime', if you only need to use cuda
# headers or link to cuda libs, but don't need to run your source
# through a cuda compiler (nvcc)
blt_register_library(NAME cuda_runtime
                     INCLUDES ${CUDA_INCLUDE_DIRS}
                     LIBRARIES ${CUDA_LIBRARIES}
                     DEFINES USE_CUDA)

