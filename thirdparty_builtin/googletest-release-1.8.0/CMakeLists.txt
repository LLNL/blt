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

enable_testing()


##########################################################
# BLT: 
# Use our option (ENABLE_GMOCK) to set gtest and gmock 
# options. 
##########################################################

#Note that googlemock target already builds googletest

if(ENABLE_GMOCK)
    set(BUILD_GMOCK ON)
    set(BUILD_GTEST OFF)
    add_subdirectory( googlemock )
else()
    set(BUILD_GMOCK OFF)
    set(BUILD_GTEST ON)
    add_subdirectory( googletest )
endif()
