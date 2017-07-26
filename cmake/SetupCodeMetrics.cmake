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

#################################################
# Setup code metrics - coverage, profiling, etc
#################################################

########################################
# Enable code coverage via gcov
# Note: Only supported for gnu or clang.
########################################
if (ENABLE_COVERAGE)
    ##########################################################################
    # Setup coverage compiler flags 
    ##########################################################################
    # Set the actual flags for coverage in the COVERAGE_FLAGS variable 
    # Note: For gcc '--coverage' is equivalent to 
    # '-fprofile-arcs -ftest-coverage' for compiling and '-lgcov' for linking
    # Additional flags that might be useful: 
    #       " -fno-inline -fno-inline-small-functions -fno-default-inline"
    blt_append_custom_compiler_flag(FLAGS_VAR   COVERAGE_FLAGS 
                                    DEFAULT " "
                                    GNU     "--coverage"
                                    CLANG   "--coverage")
        
    SET( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${COVERAGE_FLAGS}" )
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}" )
    SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_FLAGS}" )
    
    if(ENABLE_FORTRAN)
        SET( CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} ${COVERAGE_FLAGS}" )
    endif()

    ######################################
    # Setup Code Coverage Report Targets
    ######################################
    include(blt/cmake/SetupCodeCoverageReports.cmake)
    
endif()

if (VALGRIND_FOUND)
    set(MEMORYCHECK_COMMAND ${VALGRIND_EXECUTABLE} CACHE PATH "")
    set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" CACHE PATH "")
endif()
