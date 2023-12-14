cmake_minimum_required(VERSION 3.14)

project(cuda-mpi-cpp LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_CUDA On CACHE BOOL "")

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Add third-party library setup files to the project's installation
# directory.
blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

# Add the base library with its single source file and header file.
blt_add_library(
  NAME cuda-mpi-cpp
  SOURCES cuda-mpi-cpp.cu
  HEADERS cuda-mpi-cpp.cuh
  DEPENDS_ON cuda)

# Add the header file to the installation.
install(FILES cuda-mpi-cpp.cuh DESTINATION include)

# Install the header along with the necessary cmake files.
install(TARGETS cuda-mpi-cpp
  EXPORT cuda-mpi-cpp-targets)

# Install the *-config.cmake file, which will include `BLTSetupTargets` 
# when a downstream project calls `find_package` on this library.
install(FILES
  ${PROJECT_SOURCE_DIR}/cuda-mpi-cpp-config.cmake
  DESTINATION lib/cmake/cuda-mpi-cpp)

install(EXPORT cuda-mpi-cpp-targets
    DESTINATION  lib/cmake/cuda-mpi-cpp)
