###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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

if(ENABLE_TESTS)
    include(CTest)

    if(WIN32 AND BUILD_SHARED_LIBS)
        add_definitions(-DGTEST_CREATE_SHARED_LIBRARY=1)
        list(APPEND gtest_defines "-DGTEST_LINKED_AS_SHARED_LIBRARY=1")
    endif()

    # Explicitly enable/disable death tests
    if(ENABLE_GTEST_DEATH_TESTS)
        add_definitions(-DGTEST_HAS_DEATH_TEST=1)
        list(APPEND gtest_defines "-DGTEST_HAS_DEATH_TEST=1")
    else()
        add_definitions(-DGTEST_HAS_DEATH_TEST=0)
        list(APPEND gtest_defines "-DGTEST_HAS_DEATH_TEST=0")
    endif()

    #
    # Use BLT's ENABLE_GMOCK option to set the build options for 
    # gtest's cmake project
    #
    
    if(ENABLE_GMOCK)
        set(BUILD_GMOCK ON  CACHE BOOL "")
        set(BUILD_GTEST OFF CACHE BOOL "")
    else()
        set(BUILD_GMOCK OFF CACHE BOOL "")
        set(BUILD_GTEST ON  CACHE BOOL "")        
    endif()

    message(STATUS "Google Test Support is ${ENABLE_GTEST}")
    message(STATUS "Google Mock Support is ${ENABLE_GMOCK}")
    
    #
    # Guard of googletest w/ ENABLE_GTEST
    # In BTL ENABLE_GTEST is also required when using ENABLE_GMOCK
    #
    
    if(ENABLE_GTEST)
        # Enable builtin google test 
        add_subdirectory(googletest-release-1.8.0)

        blt_register_library(NAME gtest
                             INCLUDES ${gtest_SOURCE_DIR}/include
                             LIBRARIES gtest_main gtest
                             DEFINES  ${gtest_defines})
         if(ENABLE_GMOCK)
             blt_register_library(NAME gmock
                                  INCLUDES ${gmock_SOURCE_DIR}/include
                                  LIBRARIES gmock_main gmock
                                  DEFINES  ${gtest_defines})
         endif()
    endif()

    # Enable Fruit (FortRan UnIT testing) support
    if (ENABLE_FORTRAN)
        message(STATUS "Fruit Support is ${ENABLE_FRUIT}")
        if(ENABLE_FRUIT)
            add_subdirectory(fruit-3.4.1)
        endif()
    endif()

    if(ENABLE_BENCHMARKS)
        if(WIN32 AND BUILD_SHARED_LIBS)
          message(FATAL_ERROR "Benchmarks cannot be built when BUILD_SHARED_LIBS=On")
        endif()

        ## google benchmark support
        add_subdirectory(gbenchmark-master-2017-05-19)

        if (UNIX AND NOT APPLE)
          find_library(RT_LIBRARIES rt)
        endif()

        blt_register_library(NAME gbenchmark
                             INCLUDES ${benchmark_SOURCE_DIR}/include ${benchmark_SOURCE_DIR}
                             LIBRARIES benchmark ${RT_LIBRARIES}
                             )

        # This sets up a target to run the benchmarks
        add_custom_target(run_benchmarks 
                          COMMAND ctest -C Benchmark -VV -R benchmark
                          WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                          )
    endif()
endif()
