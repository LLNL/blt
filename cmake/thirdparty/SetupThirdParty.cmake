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

####################################
# BLT 3rd Party Lib Support
####################################

################################
# Git
################################
if (ENABLE_GIT)
    find_package(Git)
    if (Git_FOUND)
        message(STATUS "Git Support is ON")
        set(GIT_FOUND TRUE)
        message(STATUS "Git Executable: " ${GIT_EXECUTABLE} )
        message(STATUS "Git Version: " ${GIT_VERSION_STRING} )
    else()
        message(STATUS "Git Support is OFF")
    endif()
else()
    message(STATUS "Git Support is OFF")
endif()

################################
# MPI
################################
message(STATUS "MPI Support is ${ENABLE_MPI}")
if (ENABLE_MPI)
    include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupMPI.cmake)
endif()

################################
# CUDA
################################
message(STATUS "CUDA Support is ${ENABLE_CUDA}")
if (ENABLE_CUDA)
  include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupCUDA.cmake)
endif()

################################
# ROCM
################################
message(STATUS "ROCM Support is ${ENABLE_ROCM}")
if (ENABLE_ROCM)
  include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupROCm.cmake)
endif()

################################
# Documentation Packages
################################
if (ENABLE_DOXYGEN)
    find_package(Doxygen)
endif()

blt_find_executable(NAME        Sphinx
                    EXECUTABLES sphinx-build sphinx-build2)

################################
# Valgrind
################################
blt_find_executable(NAME        Valgrind
                    EXECUTABLES valgrind)

################################
# linting
################################
blt_find_executable(NAME        Uncrustify
                    EXECUTABLES uncrustify)

blt_find_executable(NAME        AStyle
                    EXECUTABLES astyle)

################################
# Static analysis via Cppcheck
################################
blt_find_executable(NAME        Cppcheck
                    EXECUTABLES cppcheck)
