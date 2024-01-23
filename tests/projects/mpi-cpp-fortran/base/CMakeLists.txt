cmake_minimum_required(VERSION 3.14)

project(mpi-cpp-fortran LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_MPI On CACHE BOOL "")

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Add third-party library setup files to the project's installation
# directory.
blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

blt_add_library(
    NAME mpi-cpp-fortran
    SOURCES mpi-cpp-fortran.cpp
    DEPENDS_ON mpi)

# Install the header along with the necessary cmake files.
install(TARGETS mpi-cpp-fortran
  EXPORT mpi-cpp-fortran-targets)

# Install the *-config.cmake file, which will include `BLTSetupTargets` 
# when a downstream project calls `find_package` on this library.
install(FILES
  ${PROJECT_SOURCE_DIR}/mpi-cpp-fortran-config.cmake
  DESTINATION lib/cmake/mpi-cpp-fortran)

install(EXPORT mpi-cpp-fortran-targets
    DESTINATION  lib/cmake/mpi-cpp-fortran)
