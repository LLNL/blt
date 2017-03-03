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

set( BLT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE )

################################
# Prevent in-source builds
################################
include(${BLT_ROOT_DIR}/cmake/PreventInSourceBuilds.cmake)

################################
# Setup build options and their default values
################################
include(${BLT_ROOT_DIR}/cmake/BLTOptions.cmake)

################################
# Enable ctest support 
################################
if(ENABLE_TESTS)
    enable_testing()
endif()

################################
# Macros
################################
include(${BLT_ROOT_DIR}/cmake/BLTMacros.cmake)

################################
# Standard TPL support
################################
include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupThirdParty.cmake)

################################
# Setup docs targets
################################
include(${BLT_ROOT_DIR}/cmake/SetupDocs.cmake)

################################
# Setup source checks
################################
include(${BLT_ROOT_DIR}/cmake/SetupCodeChecks.cmake)

################################
# Standard Build Layout
################################

# Defines the layout of the build directory. Namely,
# it indicates the location where the various header files should go,
# where to store libraries (static or shared), the location of the
# bin directory for all executables and the location for fortran modules.

# Set the path where all the headers will be stored
set(HEADER_INCLUDES_DIRECTORY
    ${PROJECT_BINARY_DIR}/include/
    CACHE PATH
    "Directory where all headers will go in the build tree"
    )
include_directories(${HEADER_INCLUDES_DIRECTORY})

# Set the path where all the libraries will be stored
set(LIBRARY_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/lib
    CACHE PATH
    "Directory where compiled libraries will go in the build tree"
    )

# Set the path where all the installed executables will go
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin
    CACHE PATH
    "Directory where executables will go in the build tree"
    )

# Set the path were all test executables will go
set(TEST_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/tests
    CACHE PATH
    "Directory where test executables will go in the build tree"
    )

# Set the path were all example test executables will go
set(EXAMPLE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/examples
    CACHE PATH
    "Directory where example executables will go in the build tree"
    )

# Set the Fortran module directory
set(CMAKE_Fortran_MODULE_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib/fortran
    CACHE PATH
    "Directory where all Fortran modules will go in the build tree"
    )

# Mark as advanced
mark_as_advanced(
     LIBRARY_OUTPUT_PATH
     CMAKE_RUNTIME_OUTPUT_DIRECTORY
     CMAKE_Fortran_MODULE_DIRECTORY
     )

################################
# Setup compiler options
# (must be included after HEADER_INCLUDES_DIRECTORY and MPI variables are set)
################################
include(${BLT_ROOT_DIR}/cmake/SetupCompilerOptions.cmake)

################################
# Setup code metrics -
# profiling, code coverage, etc.
# (must be included after SetupCompilerOptions)
################################
include(${BLT_ROOT_DIR}/cmake/SetupCodeMetrics.cmake)

################################
# builtin third party libs used by BLT
################################
add_subdirectory(${BLT_ROOT_DIR}/thirdparty_builtin)

################################
# BLT smoke tests
################################
if(ENABLE_TESTS)
    add_subdirectory(${BLT_ROOT_DIR}/tests)
endif()
