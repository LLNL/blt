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
# (Original LCOV + GENHTML CMake Macro) from Lars Bilke + Joakim Soderberg
#
# 2015-07-06, Aaron Black
# - Modified for use by BLT.
#
# 2016-04-04, Kenny Weiss
# - Removed support for clang; simplified flags
#
# 2017-07-25, Cyrus Harrison
# - Refactored to only include report gen logic, not coverage flags
#
###############################################################################

set(BLT_CODE_COVERAGE_REPORTS ON)

# Check for lcov
if(NOT EXISTS ${LCOV_EXECUTABLE})
    message(STATUS "Code coverage: Unable to find lcov, disabling code coverage reports.")
    set(BLT_CODE_COVERAGE_REPORTS OFF)
endif()

# Check for genthml
if(NOT EXISTS ${GENHTML_EXECUTABLE})
    message(STATUS "Code coverage: Unable to find genhtml, disabling code coverage reports.")
    set(BLT_CODE_COVERAGE_REPORTS OFF)
endif()

# Check for gcov
if(NOT EXISTS ${GCOV_EXECUTABLE})
   message(STATUS "Code coverage: GCOV_EXECUTABLE is not set, disabling code coverage reports")
   set(BLT_CODE_COVERAGE_REPORTS OFF)
endif()
    
mark_as_advanced(BLT_CODE_COVERAGE_REPORTS)


######################################################################
# Function that adds target that generates code coverage reports
#####################################################################
# Param _targetname     The name of new the custom make target and output file name.
# Param _testrunner     The name of the target which runs the tests.
#                        MUST return ZERO always, even on errors.
#                        If not, no coverage report will be created!
# Optional fourth parameter is passed as arguments to _testrunner
#   Pass them in list form, e.g.: "-j;2" for -j 2
function(add_code_coverage_target _targetname _testrunner)

    # Setup target
    add_custom_target(${_targetname}

        # Cleanup lcov
        ${LCOV_EXECUTABLE} --no-external --gcov-tool ${GCOV_EXECUTABLE} --directory ${CMAKE_BINARY_DIR} --directory ${CMAKE_SOURCE_DIR}/components --zerocounters

        # Run tests
        COMMAND ${_testrunner} ${ARGV2}

        # Capture lcov counters and generating report
        COMMAND ${LCOV_EXECUTABLE} --no-external --gcov-tool ${GCOV_EXECUTABLE} --directory ${CMAKE_BINARY_DIR} --directory ${CMAKE_SOURCE_DIR}/components --capture --output-file ${_targetname}.info
        COMMAND ${LCOV_EXECUTABLE} --no-external --gcov-tool ${GCOV_EXECUTABLE} --directory ${CMAKE_BINARY_DIR} --directory ${CMAKE_SOURCE_DIR}/components --remove ${_targetname}.info '/usr/include/*' --output-file ${_targetname}.info.cleaned
        COMMAND ${GENHTML_EXECUTABLE} -o ${_targetname} ${_targetname}.info.cleaned
        COMMAND ${CMAKE_COMMAND} -E remove ${_targetname}.info ${_targetname}.info.cleaned

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Resetting code coverage counters to zero.\nProcessing code coverage counters and generating report."
    )

    # Show info where to find the report
    add_custom_command(TARGET ${_targetname} POST_BUILD
        COMMAND ;
        COMMENT "Open ./${_targetname}/index.html in your browser to view the coverage report."
    )
endfunction()


if(BLT_CODE_COVERAGE_REPORTS)
        # Add code coverage target
        add_code_coverage_target(coverage make test)
        message(STATUS "Code coverage: reports enabled via lcov, genthml, and gcov.")
endif()

