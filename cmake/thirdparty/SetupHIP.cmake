################################
# HIP
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(HIP REQUIRED)

message(STATUS "HIP version:      ${HIP_VERSION_STRING}")
message(STATUS "HIP Include Path: ${HIP_INCLUDE_DIRS}")
message(STATUS "HIP Libraries:    ${HIP_LIBRARIES}")

# don't propagate host flags - too easy to break stuff!
set (HIP_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
  set (HIP_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else ()
  set (HIP_HOST_COMPILER ${CMAKE_C_COMPILER})
endif ()

# depend on 'hip', if you need to use hip
# headers, link to hip libs, and need to run your source
# through a hip compiler (nvcc)
blt_register_library(NAME hip
                     INCLUDES ${HIP_INCLUDE_DIRS}
                     LIBRARIES ${HIP_LIBRARIES})

# depend on 'hip_runtime', if you only need to use hip
# headers or link to hip libs, but don't need to run your source
# through a hip compiler (hipcc)
blt_register_library(NAME hip_runtime
                     INCLUDES ${HIP_INCLUDE_DIRS}
                     LIBRARIES ${HIP_LIBRARIES})
