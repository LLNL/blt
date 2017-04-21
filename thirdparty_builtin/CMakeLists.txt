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
        set(gtest_defines "-DGTEST_LINKED_AS_SHARED_LIBRARY=1")
    endif()

    # Enable builtin google test 
    add_subdirectory(googletest-release-1.8.0)

    blt_register_library(NAME gtest
                         INCLUDES ${gtest_SOURCE_DIR}/include
                         LIBRARIES gtest_main gtest
                         DEFINES  ${gtest_defines})

    blt_register_library(NAME gmock
                         INCLUDES ${gmock_SOURCE_DIR}/include
                         LIBRARIES gmock_main gmock
                         DEFINES  ${gtest_defines})

    # Enable Fruit (FortRan UnIT testing) support
    if (ENABLE_FORTRAN)
        add_subdirectory(fruit-3.4.1)
    endif()

    if(ENABLE_BENCHMARKS)
        ## google benchmark support
        add_subdirectory(gbenchmark-release-1.1.0)

        blt_register_library(NAME gbenchmark
                             INCLUDES ${benchmark_SOURCE_DIR}/include ${benchmark_SOURCE_DIR}
                             LIBRARIES benchmark
                             )

        # This sets up a target to run the benchmarks
        add_custom_target(run_benchmarks 
                          COMMAND ctest -C Benchmark -VV -R benchmark
                          WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                          )
    endif()
endif()
