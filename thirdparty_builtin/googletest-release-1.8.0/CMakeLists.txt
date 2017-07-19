cmake_minimum_required(VERSION 2.6.2)

##########################################################
# BLT FIX:
# Add policy setting until we upgrade to gtest
# with a fix for:
#   https://github.com/google/googletest/issues/709
#
# As of 7/14/17 -- there is an open PR to add this line
#   https://github.com/google/googletest/pull/782/files
##########################################################
# avoid warnings for project commands w/o VERSION
cmake_policy(SET CMP0048 NEW)

project( googletest-distribution )

##########################################################
# BLT FIX:
# Set CMAKE_MACOSX_RPATH until we upgrade to gtest
# with a fix for CMake CMP0042 warnings about 
# MACOSX_RPATH.
#
# As of 7/18/17 -- there is an open PR to add this line
#   https://github.com/google/googletest/pull/829
##########################################################
# Enable MACOSX_RPATH support for gtest and gmock
# to avoid CMP0042 warnings
set(CMAKE_MACOSX_RPATH 1)

enable_testing()

option(BUILD_GTEST "Builds the googletest subproject" OFF)

#Note that googlemock target already builds googletest
option(BUILD_GMOCK "Builds the googlemock subproject" ON)

if(BUILD_GMOCK)
  add_subdirectory( googlemock )
elseif(BUILD_GTEST)
  add_subdirectory( googletest )
endif()
