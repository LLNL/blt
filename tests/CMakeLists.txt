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

#
# Smoke tests for BLT TPLs
#

################
# gtest test
################
blt_add_executable(NAME blt_gtest_smoke
                   SOURCES blt_gtest_smoke.cpp
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest)

blt_add_test( NAME blt_gtest_smoke
              COMMAND blt_gtest_smoke)

################
# gmock test
################
blt_add_executable(NAME blt_gmock_smoke
                 SOURCES blt_gmock_smoke.cpp
                 OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                 DEPENDS_ON gtest gmock)

blt_add_test( NAME blt_gmock_smoke
              COMMAND blt_gmock_smoke)


################
# gbenchmark test
################
if (ENABLE_BENCHMARKS)
    blt_add_executable(NAME blt_gbenchmark_smoke
                       SOURCES blt_gbenchmark_smoke.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON gbenchmark)

    blt_add_benchmark( NAME blt_gbenchmark_smoke 
                       COMMAND  blt_gbenchmark_smoke "--benchmark_min_time=0.0001")
endif()


################
# fruit test
################
if (ENABLE_FORTRAN)
    blt_add_executable(NAME blt_fruit_smoke
                       SOURCES blt_fruit_smoke.f
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON fruit)

    blt_add_test(NAME blt_fruit_smoke
                 COMMAND blt_fruit_smoke)
endif()


################
# OpenMP test
################
if (ENABLE_OPENMP)
    blt_add_executable(NAME blt_openmp_smoke 
                       SOURCES blt_openmp_smoke.cpp 
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON openmp)

    blt_add_test(NAME blt_openmp_smoke
                 COMMAND blt_openmp_smoke)
endif()


################
#  MPI test
################
if (ENABLE_MPI)
    blt_add_executable(NAME blt_mpi_smoke
                       SOURCES blt_mpi_smoke.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY} )

    blt_add_test(NAME blt_mpi_smoke
                 COMMAND blt_mpi_smoke
                 NUM_PROCS 2)
endif()
