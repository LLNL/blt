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
# Build Targets
################################
option(ENABLE_DOCS       "Enables documentation" ON)
option(ENABLE_EXAMPLES   "Enables examples" ON)
option(ENABLE_TESTS      "Enables tests" ON)
option(ENABLE_BENCHMARKS "Enables benchmarks" OFF)
option(ENABLE_COVERAGE   "Enables code coverage support" OFF)


################################
# TPL Executable Options
################################
option(ENABLE_CPPCHECK     "Enables Cppcheck support" ON)
option(ENABLE_DOXYGEN      "Enables Doxygen support" ON)
option(ENABLE_GIT          "Enables Git support" ON)
option(ENABLE_SPHINX       "Enables Sphinx support" ON)
option(ENABLE_UNCRUSTIFY   "Enables Uncrustify support" ON)
option(ENABLE_ASTYLE       "Enables AStyle support" ON)
option(ENABLE_VALGRIND     "Enables Valgrind support" ON)


################################
# Build Options
################################
get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(_languages MATCHES "Fortran")
    set(_fortran_already_enabled TRUE)
else()
    set(_fortran_already_enabled FALSE)
endif()
option(ENABLE_FORTRAN      "Enables Fortran compiler support" ${_fortran_already_enabled})

option(ENABLE_MPI          "Enables MPI support" OFF)
option(ENABLE_OPENMP       "Enables OpenMP compiler support" OFF)
option(ENABLE_CUDA         "Enable CUDA support" OFF)
option(ENABLE_CLANG_CUDA   "Enable Clang's native CUDA support" OFF)
mark_as_advanced(ENABLE_CLANG_CUDA)
set(BLT_CLANG_CUDA_ARCH "sm_30" CACHE STRING "Compute architecture to use when generating CUDA code with Clang")
mark_as_advanced(BLT_CLANG_CUDA_ARCH)
option(ENABLE_ROCM         "Enable ROCM support" OFF)
set(BLT_ROCM_ARCH "gfx900" CACHE STRING "gfx architecture to use when generating ROCm code")

# Options that control if Google Test, Google Mock, and Fruit are built 
# and available for use. 
#
# If ENABLE_TESTS=OFF, no testing support is built and these option are ignored.
#
# Google Mock requires and always builds Google Test, so ENABLE_GMOCK=ON
# implies ENABLE_GTEST=ON.
#
get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(_languages MATCHES "CXX")
  set(_CXX_enabled ON)
else()
  set(_CXX_enabled OFF)
endif()
option(ENABLE_GTEST        "Enable Google Test testing support (if ENABLE_TESTS=ON)" ${_CXX_enabled})
option(ENABLE_GMOCK        "Enable Google Mock testing support (if ENABLE_TESTS=ON)" OFF)
option(ENABLE_FRUIT        "Enable Fruit testing support (if ENABLE_TESTS=ON and ENABLE_FORTRAN=ON)" ON)

if( (NOT _CXX_enabled) AND ENABLE_GTEST )
  message( FATAL_ERROR
    "You must have CXX enabled in your project to use GTEST!" )
endif()

################################
# Compiler Options
################################
option(ENABLE_ALL_WARNINGS       "Enables all compiler warnings on all build targets" ON)
option(ENABLE_WARNINGS_AS_ERRORS "Enables treating compiler warnings as errors on all build targets" OFF)

set(BLT_CXX_STD "c++11" CACHE STRING "Version of C++ standard")
set_property(CACHE BLT_CXX_STD PROPERTY STRINGS c++98 c++11 c++14)

if( NOT ( ( BLT_CXX_STD STREQUAL "c++98" ) 
       OR ( BLT_CXX_STD STREQUAL "c++11" ) 
       OR ( BLT_CXX_STD STREQUAL "c++14" ) ) )
    message(FATAL_ERROR "${BLT_CXX_STD} is an invalid entry for BLT_CXX_STD.
Valid Options are ( c++98, c++11, c++14 )")
endif()


################################
# Generator Options
################################
option(ENABLE_FOLDERS "Organize projects using folders (in generators that support this)" OFF)


################################
# Advanced configuration options
################################

option(ENABLE_FIND_MPI     "Enables CMake's Find MPI support (Turn off when compiling with the mpi wrapper directly)" ON)

option(
    ENABLE_GTEST_DEATH_TESTS
    "Enables tests that assert application failure. Only valid when tests are enabled"
    OFF )

option(
    ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC
    "Option to ensure that all tests are invoked through mpiexec. Required on some platforms, like IBM's BG/Q."
    OFF )

# All advanced options should be marked as advanced
mark_as_advanced(
    ENABLE_FIND_MPI
    ENABLE_GTEST_DEATH_TESTS
    ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC )
       
if (DEFINED ENABLE_SHARED_LIBS)
    message(FATAL_ERROR "ENABLE_SHARED_LIBS is a deprecated BLT option."
                        "Use the standard CMake option, BUILD_SHARED_LIBS, instead.")
endif()