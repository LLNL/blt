###############################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

find_path(ROCM_PATH
     NAMES bin/hcc
     PATHS /opt/rocm
#     PATH_SUFFIXES hcc
     DOC "Path to ROCm hcc executable")


if(ROCM_PATH)
    message(STATUS "ROCM_PATH:  ${ROCM_PATH}")
    set(CMAKE_C_COMPILER_ID "HCC")
    set(CMAKE_CXX_COMPILER_ID "HCC")

    set(HSA_PATH "${ROCM_PATH}/hsa")

    set(ROCM_COMPILE_FLAGS "-hc")
    set(ROCM_ARCH_FLAG "-amdgpu-target=${BLT_ROCM_ARCH}")
    set(ROCM_LIBRARIES "-lhc_am")

#    set(ROCM_C_COMPILE_FLAGS ${ROCM_COMPILE_FLAGS})
#    set(ROCM_C_INCLUDE_PATH "${ROCM_PATH}/hcc/include")
#    set(ROCM_C_LIBRARIES ${ROCM_LIBRARIES})
#    set(ROCM_C_LIBRARY_PATH "${ROCM_PATH}/hcc/lib")
#    set(ROCM_C_LINK_FLAGS "${ROCM_C_LIBRARIES} -L${ROCM_C_LIBRARY_PATH} ${ROCM_ARCH_FLAG}")

    set(ROCM_CXX_COMPILE_FLAGS ${ROCM_COMPILE_FLAGS})
    set(ROCM_CXX_INCLUDE_PATH "${ROCM_PATH}/hcc/include")
    set(ROCM_CXX_LIBRARIES ${ROCM_LIBRARIES})
    set(ROCM_CXX_LIBRARY_PATH "${ROCM_PATH}/hcc/lib")
    set(ROCM_CXX_LINK_FLAGS "${ROCM_C_LIBRARIES} -L${ROCM_C_LIBRARY_PATH} ${ROCM_ARCH_FLAG}")


    set(HCC_COMPILER ${ROCM_PATH}/bin/hcc)


#    set(CMAKE_C_INCLUDE_PATH ${ROCM_C_INCLUDE_PATH})
#    set(CMAKE_C_LINK_EXECUTABLE ${HCC_COMPILER})
    set(CMAKE_CXX_INCLUDE_PATH ${ROCM_CXX_INCLUDE_PATH})
    set(CMAKE_CXX_LINK_EXECUTABLE ${HCC_COMPILER})
#    set(CMAKE_C_LINK_FLAGS ${ROCM_C_LINK_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS ${ROCM_CXX_LINK_FLAGS})

#    set(CMAKE_C_COMPILER "${ROCM_PATH}/bin/hcc" CACHE FILEPATH "HCC compiler" FORCE)
#    set(CMAKE_C_FLAGS "${ROCM_C_COMPILE_FLAGS}" CACHE STRING "HCC compiler flags" FORCE)
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hcc" CACHE FILEPATH "HCC compiler" FORCE)
    set(CMAKE_CXX_FLAGS "${ROCM_CXX_COMPILE_FLAGS}" CACHE STRING "HCC compiler flags" FORCE)
    set(CMAKE_LINKER "${ROCM_PATH}/bin/hcc" CACHE FILEPATH "HCC linker" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS ${ROCM_CXX_LINK_FLAGS} CACHE STRING "HCC link flags" FORCE)
#    set( ${})

    set(ROCM_FOUND TRUE)

else()
    set(ROCM_FOUND FALSE)
    message(WARNING "ROCm hcc executable not found")
endif()
