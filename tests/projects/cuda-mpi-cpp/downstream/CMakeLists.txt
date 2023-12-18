cmake_minimum_required(VERSION 3.14)

project(cuda-mpi-cpp-user LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_MPI On CACHE BOOL "")

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Calling find_package sets up the TPL targets needed by the project
# in its argument.
find_package(cuda-mpi-cpp REQUIRED 
             NO_DEFAULT_PATH
             PATHS ${base_install_dir}/lib/cmake/cuda-mpi-cpp
             ${base_install_dir})

if (NOT TARGET cuda)
  message(FATAL_ERROR "Target cuda was not configured successfully during"
                      "downstream project configuration.")
endif()

# The downstream project also introduces mpi, so this target must 
# exist as well.
if (NOT TARGET mpi)
  message(FATAL_ERROR "Target MPI was not configured successfully during"
                      "downstream project configuration.")
endif()

# Verify that MPI is using the generator expression dependent on CUDA compilation support
# in its compile flags.
get_target_property(mpi_compile_flags mpi INTERFACE_COMPILE_OPTIONS)
SET(expected_mpi_compile_flags $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>)
if (NOT "${mpi_compile_flags}" STREQUAL "${expected_mpi_compile_flags}")
  message(FATAL_ERROR "Expected MPI compile flags to contain generator expression ${expected_mpi_compile_flags}, "
                      "but got ${mpi_compile_flags}. If the compile flag generator expression inside BLTSetupMPI.cmake"
                      "was changed, please update this test.  Otherwise, this test has failed.")
endif()

blt_add_executable(
    NAME hello-mpi-cuda
    SOURCES cuda-mpi-cpp-user.cpp
    DEPENDS_ON mpi cuda-mpi-cpp
)

target_include_directories(hello-mpi-cuda PUBLIC ${base_install_dir}/include)
