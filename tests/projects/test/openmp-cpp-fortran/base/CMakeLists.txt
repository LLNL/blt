cmake_minimum_required(VERSION 3.14)

project(openmp-cpp-fortran LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_OPENMP On CACHE BOOL "")

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Add third-party library setup files to the project's installation
# directory.
blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

blt_add_library(
    NAME openmp-cpp-fortran
    SOURCES openmp-cpp-fortran.cpp
    DEPENDS_ON openmp)

# Install library
install(TARGETS openmp-cpp-fortran
  EXPORT openmp-cpp-fortran-targets)

install(EXPORT openmp-cpp-fortran-targets
  DESTINATION  lib/cmake/openmp-cpp-fortran)

# Install the *-config.cmake file, which will include `BLTSetupTargets` 
# when a downstream project calls `find_package` on this library.
install(FILES
  ${PROJECT_SOURCE_DIR}/openmp-cpp-fortran-config.cmake
  DESTINATION lib/cmake/openmp-cpp-fortran)

add_executable(openmp-cpp-fortran-print openmp-cpp-fortran.cpp)
target_link_libraries(openmp-cpp-fortran-print PUBLIC openmp-cpp-fortran)
