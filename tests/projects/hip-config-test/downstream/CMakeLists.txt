cmake_minimum_required(VERSION 3.14)

project(hip-config-test-user LANGUAGES CXX C)

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Calling find_package sets up the TPL targets needed by the project
# in its argument.
find_package(hip-config-test REQUIRED
  NO_DEFAULT_PATH
  PATHS ${base_install_dir}/lib/cmake/hip-config-test
  ${base_install_dir})


if (NOT TARGET blt_hip)
  message(FATAL_ERROR "Target blt_hip was not configured successfully during"
                      "downstream project configuration.")
endif()

if (NOT TARGET blt_hip_runtime)
  message(FATAL_ERROR "Target blt_hip_runtime was not configured successfully during"
                      "downstream project configuration.")
endif()

blt_add_executable(
    NAME hello-hip
    SOURCES hip-config-test-user.cpp
    DEPENDS_ON hip-config-test
)

target_include_directories(hello-hip PUBLIC ${base_install_dir}/include)
