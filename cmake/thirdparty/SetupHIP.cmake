################################
# HIP
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(HIP REQUIRED)

message(STATUS "HIP version:      ${HIP_VERSION_STRING}")
message(STATUS "HIP platform:     ${HIP_PLATFORM}")
#message(STATUS "HIP Include Path: ${HIP_INCLUDE_DIRS}")
#message(STATUS "HIP Libraries:    ${HIP_LIBRARIES}")

# depend on 'hip', if you need to use hip
# headers, link to hip libs, and need to run your source
# through a hip compiler (hipcc)
blt_register_library(NAME hip
                     INCLUDES ${HIP_INCLUDE_DIRS}
                     LIBRARIES ${HIP_LIBRARIES}
                     DEFINES USE_HIP)

# depend on 'hip_runtime', if you only need to use hip
# headers or link to hip libs, but don't need to run your source
# through a hip compiler (hipcc)
blt_register_library(NAME hip_runtime
                     INCLUDES ${HIP_INCLUDE_DIRS}
                     LIBRARIES ${HIP_LIBRARIES}
                     DEFINES USE_HIP)
