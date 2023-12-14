cmake_minimum_required(VERSION 3.14)

project(openmp-mpi-cpp LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_OPENMP On CACHE BOOL "")

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Add third-party library setup files to the project's installation
# directory.
blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

# Add the base library with its single source file and header file.
blt_add_library(
  NAME openmp-mpi-cpp
  SOURCES openmp-mpi-cpp.cpp
  HEADERS openmp-mpi-cpp.hpp
  DEPENDS_ON openmp)

# Add the header file to the installation.
install(FILES openmp-mpi-cpp.hpp DESTINATION include)

# Install the header along with the necessary cmake files.
install(TARGETS openmp-mpi-cpp
  EXPORT openmp-mpi-cpp-targets)

# Install the *-config.cmake file, which will include `BLTSetupTargets` 
# when a downstream project calls `find_package` on this library.
install(FILES
  ${PROJECT_SOURCE_DIR}/openmp-mpi-cpp-config.cmake
  DESTINATION lib/cmake/openmp-mpi-cpp)

install(EXPORT openmp-mpi-cpp-targets
    DESTINATION  lib/cmake/openmp-mpi-cpp)
