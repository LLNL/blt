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
# ROCM
################################

if (ENABLE_ROCM)
    set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
    find_package(ROCm REQUIRED)

    if (ROCM_FOUND)
        message(STATUS "ROCM  Compile Flags:  ${ROCM_CXX_COMPILE_FLAGS}")
        message(STATUS "ROCM  Include Path:   ${ROCM_INCLUDE_PATH}")
        message(STATUS "ROCM  Link Flags:     ${ROCM_CXX_LINK_FLAGS}")
        message(STATUS "ROCM  Libraries:      ${ROCM_CXX_LIBRARIES}")
        message(STATUS "ROCM  Device Arch:    ${ROCM_ARCH}")

        if (ENABLE_FORTRAN)
             message(ERROR "ROCM does not support Fortran at this time")
        endif()
    else()
        message(ERROR "ROCM Executable not found")
    endif()
endif()



# register ROCM with blt
blt_register_library(NAME rocm
                     INCLUDES ${ROCM_CXX_INCLUDE_PATH}  
                     LIBRARIES ${ROCM_CXX_LIBRARIES}  
                     COMPILE_FLAGS ${ROCM_CXX_COMPILE_FLAGS}
                     LINK_FLAGS    ${ROCM_CXX_LINK_FLAGS} 
                     DEFINES USE_ROCM)


