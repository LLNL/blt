# Clear variables set in the host-config so we can test that they are inherited
# from the upstream project
foreach(_option MPI CUDA HIP OPENMP)
    unset(ENABLE_${_option} CACHE)
    unset(ENABLE_${_option})
endforeach()

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

# Verify that MPI keeps CXX and CUDA compile options language scoped. The
# precise option payload is owned by the CMake FindMPI version in use.
get_target_property(mpi_compile_flags mpi INTERFACE_COMPILE_OPTIONS)
foreach(_mpi_language CXX CUDA)
  if(NOT "${mpi_compile_flags}" MATCHES "COMPILE_LANGUAGE:${_mpi_language}")
    message(FATAL_ERROR
            "Expected language-scoped ${_mpi_language} MPI compile flags, "
            "but got '${mpi_compile_flags}'")
  endif()
endforeach()

if("${mpi_compile_flags}" MATCHES "COMPILE_LANGUAGE:C>")
  message(FATAL_ERROR
          "Unexpected C MPI compile flags in this CXX/CUDA-only project: "
          "'${mpi_compile_flags}'")
endif()

blt_add_executable(
    NAME hello-mpi-cuda
    SOURCES cuda-mpi-cpp-user.cpp
    DEPENDS_ON mpi cuda-mpi-cpp
)

target_include_directories(hello-mpi-cuda PUBLIC ${base_install_dir}/include)
