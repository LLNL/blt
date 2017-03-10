###############################################################################
# Copyright (c) 2012 - 2015, Lars Bilke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#
# 2012-01-31, Lars Bilke
# - Enable Code Coverage
#
# 2013-09-17, Joakim Soderberg
# - Added support for Clang.
# - Some additional usage instructions.
#
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
# 2015-07-06, Aaron Black
# - Modified for use by ASC toolkit.
# 2016-04-04, Kenny Weiss
# - Removed support for clang; simplified flags
#
# USAGE:

# 0. (Mac only) If you use Xcode 5.1 make sure to patch geninfo as described here:
#      http://stackoverflow.com/a/22404544/80480
#
# 1. Copy this file into your cmake modules path.
#
# 2. Add the following line to your CMakeLists.txt:
#      INCLUDE(CodeCoverage)
#
#    [The script] sets compiler flags to turn off optimization and enable coverage:
#
# 3. Use the function add_code_coverage_target to create a custom make target
#    which runs your test executable and produces a lcov code coverage report:
#    Example:
#	 add_code_coverage_target(
#				my_coverage_target  # Name for custom target -- and for the output directory
#				test_driver         # Name of the test driver executable that runs the tests.
#									# NOTE! This should always have a ZERO as exit code
#									# otherwise the coverage generation will not complete.
#				)
#
# 4. Build a Debug build:
#	 cmake -DCMAKE_BUILD_TYPE=Debug ..
#	 make
#	 make my_coverage_target
#
#

# Check requirements

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
   MESSAGE(FATAL_ERROR "Code coverage requires 'Debug' build type. "
					   "Current build type is ${CMAKE_BUILD_TYPE}.")
endif()

if ( NOT COMPILER_FAMILY_IS_GNU ) 
   MESSAGE(FATAL_ERROR "Code coverage only supported on gnu compiler. "
					   "Either set ENABLE_COVERAGE=FALSE or change your compiler to gnu. "
					   "Current CXX compiler is ${CMAKE_CXX_COMPILER_ID}.")
endif()


# Verify lcov found.  Will attempt to use lcov in uberenv first, then check for one in path.
IF(NOT EXISTS ${LCOV_PATH})
   MESSAGE(STATUS "Code coverage: LCOV_PATH is not set, attempting to find lcov in your path...")
   FIND_PROGRAM( LCOV_PATH lcov )
   IF(NOT EXISTS ${LCOV_PATH})
      MESSAGE(FATAL_ERROR "Code coverage: Unable to find lcov, try setting LCOV_PATH in your host-config file.")
   ENDIF()
ENDIF()

# Verify genhtml found.  Will attempt to use genhtml in uberenv first, then check for one in path.
IF(NOT EXISTS ${GENHTML_PATH})
   MESSAGE(STATUS "Code coverage: GENHTML_PATH is not set, attempting to find genhtml in your path...")
   FIND_PROGRAM( GENHTML_PATH genhtml )
   IF(NOT EXISTS ${GENHTML_PATH})
      MESSAGE(FATAL_ERROR "Code coverage: Unable to find genhtml, try setting GENHTML_PATH in your host-config file.")
   ENDIF()
ENDIF()

# Verify gcov path was provided in host config file.  Do not try to locate it in path, as different versions of gcov and gcc do not work together.
IF(NOT EXISTS ${GCOV_PATH})
   MESSAGE( FATAL_ERROR "Code coverage: GCOV_PATH is not set.  This must be set in your host-config file.")
ENDIF()


# Set the actual flags for coverage in the COVERAGE_FLAGS variable 
# Note: '--coverage' is equivalent to '-fprofile-arcs -ftest-coverage' for compiling and '-lgcov' for linking
# Additional flags that might be useful: 
#       " -fno-inline -fno-inline-small-functions -fno-default-inline"
blt_append_custom_compiler_flag(
        FLAGS_VAR   COVERAGE_FLAGS 
        DEFAULT     "--coverage"        
        )

SET(CMAKE_CXX_FLAGS_COVERAGE
    "-g -O0 ${COVERAGE_FLAGS}"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )
SET(CMAKE_C_FLAGS_COVERAGE
    "-g -O0 ${COVERAGE_FLAGS}"
    CACHE STRING "Flags used by the C compiler during coverage builds."
    FORCE )
    
if(ENABLE_FORTRAN)
    SET(CMAKE_Fortran_FLAGS_COVERAGE
        "-g -O0 ${COVERAGE_FLAGS}"
        CACHE STRING "Flags used by the Fortran compiler during coverage builds."
        FORCE )
    MARK_AS_ADVANCED(CMAKE_Fortran_FLAGS_COVERAGE)
endif()

SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE
    "${COVERAGE_FLAGS}"
    CACHE STRING "Flags used for linking binaries during coverage builds."
    FORCE )
    
SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
    ""
    CACHE STRING "Flags used by the shared libraries linker during coverage builds."
    FORCE )
    
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_EXE_LINKER_FLAGS_COVERAGE
    CMAKE_SHARED_LINKER_FLAGS_COVERAGE )


# Param _targetname     The name of new the custom make target and output file name.
# Param _testrunner     The name of the target which runs the tests.
#						MUST return ZERO always, even on errors.
#						If not, no coverage report will be created!
# Optional fourth parameter is passed as arguments to _testrunner
#   Pass them in list form, e.g.: "-j;2" for -j 2
FUNCTION(add_code_coverage_target _targetname _testrunner)

	# Setup target
	ADD_CUSTOM_TARGET(${_targetname}

		# Cleanup lcov
		${LCOV_PATH} --no-external --gcov-tool ${GCOV_PATH} --directory ${CMAKE_BINARY_DIR} --directory ${CMAKE_SOURCE_DIR}/components --zerocounters

		# Run tests
		COMMAND ${_testrunner} ${ARGV2}

		# Capturing lcov counters and generating report
		COMMAND ${LCOV_PATH} --no-external --gcov-tool ${GCOV_PATH} --directory ${CMAKE_BINARY_DIR} --directory ${CMAKE_SOURCE_DIR}/components --capture --output-file ${_targetname}.info
		COMMAND ${LCOV_PATH} --no-external --gcov-tool ${GCOV_PATH} --directory ${CMAKE_BINARY_DIR} --directory ${CMAKE_SOURCE_DIR}/components --remove ${_targetname}.info '/usr/include/*' --output-file ${_targetname}.info.cleaned
		COMMAND ${GENHTML_PATH} -o ${_targetname} ${_targetname}.info.cleaned
		COMMAND ${CMAKE_COMMAND} -E remove ${_targetname}.info ${_targetname}.info.cleaned

		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		COMMENT "Resetting code coverage counters to zero.\nProcessing code coverage counters and generating report."
	)

	# Show info where to find the report
	ADD_CUSTOM_COMMAND(TARGET ${_targetname} POST_BUILD
		COMMAND ;
		COMMENT "Open ./${_targetname}/index.html in your browser to view the coverage report."
	)

ENDFUNCTION()

# Add code coverage target
add_code_coverage_target(coverage make test)
SET( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_COVERAGE}" )
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_COVERAGE}" )
SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_COVERAGE}" )

if(ENABLE_FORTRAN)
    SET( CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_COVERAGE}" )
endif()

MESSAGE(STATUS "Code coverage: enabled via gcov.")
