cmake_minimum_required(VERSION 3.14)

project(mpi-cpp-fortran-missing-wrapper LANGUAGES CXX Fortran)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_MPI ON CACHE BOOL "")
set(BLT_ALLOW_MISSING_MPI_WRAPPER ON CACHE BOOL "")

# Ensure this is treated as missing, even if the host-config provides it.
set(MPI_Fortran_COMPILER "" CACHE FILEPATH "" FORCE)

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Add third-party library setup files to the project's installation directory.
blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

blt_add_library(
    NAME mpi-cpp-fortran-missing-wrapper
    SOURCES mpi-cpp-fortran-missing-wrapper.cpp
    DEPENDS_ON blt::mpi)

blt_add_executable(
    NAME non-mpi-fortran-example
    SOURCES non_mpi_fortran_example.f90)

install(TARGETS mpi-cpp-fortran-missing-wrapper
  EXPORT mpi-cpp-fortran-missing-wrapper-targets)

install(FILES
  ${PROJECT_SOURCE_DIR}/mpi-cpp-fortran-missing-wrapper-config.cmake
  DESTINATION lib/cmake/mpi-cpp-fortran-missing-wrapper)

install(EXPORT mpi-cpp-fortran-missing-wrapper-targets
    DESTINATION  lib/cmake/mpi-cpp-fortran-missing-wrapper)
