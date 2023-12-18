cmake_minimum_required(VERSION 3.14)

project(hip-config-test LANGUAGES CXX C)

set(ENABLE_MPI OFF)
set(ENABLE_FORTRAN OFF)
set(ENABLE_HIP ON)

# Load BLT
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# Add third-party library setup files to the project's installation
# directory.
blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

# Add the base library with its single source file and header file.
blt_add_library(
  NAME hip-config-test
  SOURCES hip-library.cpp
  HEADERS hip-library.hpp
  DEPENDS_ON blt::hip)

# Add the header file to the installation.
install(FILES hip-library.hpp DESTINATION include)

# Install the header along with the necessary cmake files.
install(TARGETS hip-config-test
  EXPORT hip-config-test-targets)

# Install the *-config.cmake file, which will include `BLTSetupTargets` 
# when a downstream project calls `find_package` on this library.
install(FILES
  ${PROJECT_SOURCE_DIR}/hip-config-test-config.cmake
  DESTINATION lib/cmake/hip-config-test)

install(EXPORT hip-config-test-targets
    DESTINATION  lib/cmake/hip-config-test)

