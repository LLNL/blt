cmake_minimum_required(VERSION 3.14)

project(cuda-clang-cpp-user LANGUAGES CXX)

set(ENABLE_CLANG_CUDA On CACHE BOOL "")
set(BLT_CLANG_CUDA_ARCH "sm_70" CACHE STRING "")
set(CMAKE_CUDA_SEPARABLE_COMPILATION On CACHE BOOL "")
# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

set(BLT_CXX_STD c++14 CACHE STRING "")
# Calling find_package sets up the TPL targets needed by the project
# in its argument.
find_package(cuda-clang-cpp REQUIRED
  NO_DEFAULT_PATH
  PATHS ${base_install_dir}/lib/cmake/cuda-clang-cpp
  ${base_install_dir})

if (NOT TARGET blt::cuda)
  message(FATAL_ERROR "Target cuda was not configured successfully during"
                      "downstream project configuration.")
endif()

# Check that the cuda target created during the `find_package` call contains the expected 
# user provided flags for clang cuda.
get_target_property(cuda_compile_flags blt::cuda INTERFACE_COMPILE_OPTIONS)

message(STATUS "CUDA target created with compile flags set to: ${cuda_compile_flags}")

string(FIND "${cuda_compile_flags}" "-x" clang_cuda_flag)
string(FIND "${cuda_compile_flags}" "sm_70" clang_cuda_arch)
string(FIND "${cuda_compile_flags}" "/usr/tce/packages/cuda/cuda-11.1.0" expected_cuda_toolkit_dir)

if (clang_cuda_flag EQUAL -1)
    message(FATAL_ERROR "Expected downstream project CLANG CUDA config flags to override upstream project's.")
endif()

if(clang_cuda_arch EQUAL -1)
  message(FATAL_ERROR "Expected downstream project CLANG CUDA cuda arch config flags to override upstream project's.")
endif()

if (expected_cuda_toolkit_dir EQUAL -1)
    message(FATAL_ERROR "Expected downstream project config flags to inherit upstream project's when alternative is not specified.")
endif()

# Compile a basic example to test correctness of link and compile flags.
blt_add_executable(
  NAME hello-cuda
  SOURCES cuda-clang-cpp-user.cpp
  DEPENDS_ON cuda-clang-cpp blt::cuda
)

target_include_directories(hello-cuda PUBLIC ${base_install_dir}/include)
