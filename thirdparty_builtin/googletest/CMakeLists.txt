# Note: CMake support is community-based. The maintainers do not use CMake
# internally.

# TODO: Per patches/gtest-2020-01-07-remove-minimum-cmake.patch
# The below rule may in fact stay here, it looks unlikely that we build with
# an older version of CMake.
cmake_minimum_required(VERSION 3.5)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

if (POLICY CMP0069)
  cmake_policy(SET CMP0069 NEW)
endif (POLICY CMP0069)

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif (POLICY CMP0077)

project(googletest-distribution)
set(GOOGLETEST_VERSION 1.13.0)

# TODO: Per patches/gtest-2020-01-07-remove-cxx-std-logic.patch
# it looks like the below section should be removed.
#if(NOT CYGWIN AND NOT MSYS AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL QNX)
#  set(CMAKE_CXX_EXTENSIONS OFF)
#endif()

enable_testing()

include(CMakeDependentOption)
include(GNUInstallDirs)

#Note that googlemock target already builds googletest
option(BUILD_GMOCK "Builds the googlemock subproject" ON)
option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" ON)
option(GTEST_HAS_ABSL "Use Abseil and RE2. Requires Abseil and RE2 to be separately added to the build." OFF)

if(BUILD_GMOCK)
  add_subdirectory( googlemock )
elseif(BUILD_GTEST)
  add_subdirectory( googletest )
endif()
