cmake_minimum_required(VERSION 3.14)

project(openmp-mpi-cpp-user LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_MPI On CACHE BOOL "")

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Calling find_package sets up the TPL targets needed by the project
# in its argument.
find_package(openmp-mpi-cpp REQUIRED 
             NO_DEFAULT_PATH
             PATHS ${base_install_dir}/lib/cmake/openmp-mpi-cpp
             ${base_install_dir})

if (NOT TARGET openmp)
  message(FATAL_ERROR "Target OpenMP was not configured successfully during"
                      "downstream project configuration.")
endif()

# The downstream project also introduces mpi, so this target must 
# exist as well.
if (NOT TARGET mpi)
  message(FATAL_ERROR "Target MPI was not configured successfully during"
                      "downstream project configuration.")
endif()

blt_add_library(
    NAME openmp-mpi-cpp-user
    SOURCES openmp-mpi-cpp-user.cpp
    DEPENDS_ON mpi openmp-mpi-cpp)

blt_add_executable(
    NAME hello-mpi-openmp
    SOURCES openmp-mpi-cpp-user.cpp
    DEPENDS_ON openmp-mpi-cpp-user
)
