###############################################################################
# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-725085
#
# All rights reserved.
#
# This file is part of BLT.
#
# For additional details, please also read BLT/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

################################
# BLT Internal Testing Project
################################

################################
cmake_minimum_required(VERSION 3.1)

project(blt-example LANGUAGES C CXX)

###############################################################################
# Setup BLT
###############################################################################
# Set BLT_SOURCE_DIR to default location, if not set by user 
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../..")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

if(WIN32 AND BUILD_SHARED_LIBS)
    add_definitions("-DWIN32_SHARED_LIBS")
endif()


################################
# Project options
################################
option(TEST_GIT_MACROS 
       "Toggle smoke tests for git macros. Off by default since it requires a modified git repo." 
       OFF)


################################
# Add a library
################################

blt_add_library( NAME example
                 SOURCES "src/Example.cpp"
                 HEADERS "src/Example.hpp"
                 )

if(ENABLE_GTEST)
    ################################
    # Add an executable
    # (which happens to be a test)
    ################################
    blt_add_executable(
             NAME t_example_smoke
             SOURCES "src/t_example_smoke.cpp"
             DEPENDS_ON example gtest)

    ################################
    # Register our test w/ ctest
    ################################
    blt_add_test(NAME t_example_smoke
                 COMMAND t_example_smoke)


    ####################
    #  Header-only test
    ####################

    blt_add_library(NAME blt_header_only
                    HEADERS "src/HeaderOnly.hpp")

    # This executable depends on the header-only library

    blt_add_executable(
      NAME t_header_only_smoke
      SOURCES "src/t_header_only_smoke.cpp"
      DEPENDS_ON blt_header_only gtest)

    blt_add_test(NAME t_header_only_smoke
                 COMMAND t_header_only_smoke)

    ####################################################
    # Tests blt_combine_static_libs macro
    # Note: Does not currently work on Windows or BGQ
    ####################################################
    set(_try_combine_static_libs ON)

    if(CMAKE_HOST_WIN32)
        set(_try_combine_static_libs OFF)
    endif()

    if ("$ENV{SYS_TYPE}" STREQUAL "bgqos_0")
        set(_try_combine_static_libs OFF)
    endif()

    if ( ${_try_combine_static_libs} )
        add_subdirectory( src/combine_static_library_test )
    endif ()

    ####################
    # Git Macros test
    ####################
    if ( GIT_FOUND AND TEST_GIT_MACROS)

      blt_is_git_repo( OUTPUT_STATE is_git_repo
                       SOURCE_DIR ${PROJECT_SOURCE_DIR} )

      if ( ${is_git_repo} )

        ## get the latest tag from the master branch
        blt_git_tag( OUTPUT_TAG blt_tag
                     RETURN_CODE rc
                     ON_BRANCH master
                     SOURCE_DIR ${PROJECT_SOURCE_DIR}
                     )
        if ( NOT ${rc} EQUAL 0 )
          message(FATAL_ERROR "blt_git_tag failed!")
        endif()

        ## get the name of the current (i.e., checked out) branch
        blt_git_branch( BRANCH_NAME blt_branch
                        RETURN_CODE rc
                        SOURCE_DIR ${PROJECT_SOURCE_DIR}
                        )
        if ( NOT ${rc} EQUAL 0 )
          message(FATAL_ERROR "blt_git_branch failed!" )
        endif()

        ## get sha1 at the tip of the current branch
        blt_git_hashcode ( HASHCODE blt_sha1
                           RETURN_CODE rc
                           SOURCE_DIR ${PROJECT_SOURCE_DIR}
                           )
        if ( NOT ${rc} EQUAL 0 )
          message(FATAL_ERROR "blt_git_hashcode failed!")
        endif()

        set(BLT_TEST_TAG ${blt_tag})
        set(BLT_TEST_SHA1 ${blt_sha1})
        set(BLT_TEST_BRANCH ${blt_branch})

        configure_file( src/t_git_macros_smoke.cpp.in
                        ${CMAKE_BINARY_DIR}/t_git_macros_smoke.cpp )

        blt_add_executable(
          NAME t_git_macros_smoke
          SOURCES "${CMAKE_BINARY_DIR}/t_git_macros_smoke.cpp"
          DEPENDS_ON gtest
          )

        blt_add_test( NAME t_git_macros_smoke
                      COMMAND t_git_macros_smoke )

      endif() # endif is_git_repo

    endif() # endif Git_FOUND

    ################
    # CUDA tests
    #################
    if (ENABLE_CUDA)
        add_subdirectory(src/test_cuda_device_call_from_kernel)
    endif()
    
    ########################################################
    # Tests blt_add_target_definitions macro
    # Four variants of a test with a list of two definitions
    ########################################################
    set(_variant_1 A=1 B)         # neither use '-D'
    set(_variant_2 -DA=1 -DB)     # both uses '-D'
    set(_variant_3 "A=1;-DB")     # list passed in as string
    set(_variant_4 " " "-DA=1;B") # list can contain empty strings
    foreach(i RANGE 1 4)
        set(_casename "_variant_${i}")
        set(_testname "t_example_compile_definitions_test${_casename}")

        blt_add_executable( 
            NAME ${_testname}
            SOURCES src/t_example_compile_definitions.cpp 
            DEPENDS_ON gtest)

        blt_add_target_definitions(
            TO ${_testname}
            TARGET_DEFINITIONS ${${_casename}})

        blt_add_test( 
            NAME ${_testname}
            COMMAND ${_testname})
    endforeach()

endif() # endif ENABLE_GTEST

message(STATUS "Exercising blt_print_target_properties macro on some targets and non-targets.")
message(STATUS "")
foreach(_target gtest example t_example_smoke not-a-target blt_header_only mpi)
    blt_print_target_properties(TARGET ${_target})
endforeach()

