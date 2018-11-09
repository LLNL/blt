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
# CUDA
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")




if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )

  get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

  if ( NOT CMAKE_CUDA_HOST_COMPILER )
  
    if("CUDA" IN_LIST LANGUAGES )
      message( FATAL_ERROR 
               "CUDA Enabled prior to setting CMAKE_CUDA_HOST_COMPILER. Please set \
                CMAKE_CUDA_HOST_COMPILER prior to ENABLE_LANGUAGE(CUDA) or PROJECT(.. LANGUAGES CUDA) ")
    endif()    
  
    if ( CMAKE_CXX_COMPILER )
      set ( CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
    else ()
      set ( CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
    endif ()
  endif ()
  message(STATUS "CMAKE_CUDA_HOST_COMPILER:      ${CMAKE_CUDA_HOST_COMPILER}")
endif ()


enable_language(CUDA)


############################################################
# Map Legacy FindCUDA variables to native cmake variables
############################################################
# if we are linking with NVCC, define the link rule here
# Note that some mpi wrappers might have things like -Wl,-rpath defined, which when using 
# FindMPI can break nvcc. In that case, you should set ENABLE_FIND_MPI to Off and control
# the link using CMAKE_CUDA_LINK_FLAGS. -Wl,-rpath, equivalent would be -Xlinker -rpath -Xlinker
if (CUDA_LINK_WITH_NVCC)
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CUDA_FLAG "-Xlinker -rpath -Xlinker")
  set(CMAKE_CUDA_LINK_EXECUTABLE
    "${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS>  <FLAGS>  <LINK_FLAGS>  <OBJECTS> -o <TARGET>  <LINK_LIBRARIES>")
  # do a no-op for the device links - for some reason the device link library dependencies are only a subset of the 
  # executable link dependencies so the device link fails if there are any missing CUDA library dependencies. Since
  # we are doing a link with the nvcc compiler, the device link step is unnecessary .
  # Frustratingly, nvcc-link errors out if you pass it an empty file, so we have to first compile the empty file. 
  set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "touch <TARGET>.cu ; ${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS> -std=c++11 -dc <TARGET>.cu -o <TARGET>")
  set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "touch <TARGET>.cu ; ${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS> -std=c++11 -dc <TARGET>.cu -o <TARGET>")
endif()

find_package(CUDA REQUIRED)

message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")

# don't propagate host flags - too easy to break stuff!
set (CUDA_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
  set (CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else ()
  set (CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
endif ()

if (ENABLE_CLANG_CUDA)
  set (clang_cuda_flags "-x cuda --cuda-gpu-arch=${BLT_CLANG_CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

  blt_register_library(NAME cuda
                       COMPILE_FLAGS ${clang_cuda_flags}
                       INCLUDES ${CUDA_INCLUDE_DIRS}
                       LIBRARIES ${CUDA_LIBRARIES}
                       DEFINES USE_CUDA)
else ()
  # depend on 'cuda', if you need to use cuda
  # headers, link to cuda libs, and need to run your source
  # through a cuda compiler (nvcc)
  blt_register_library(NAME cuda
                       INCLUDES ${CUDA_INCLUDE_DIRS}
                       LIBRARIES ${CUDA_LIBRARIES}
                       DEFINES USE_CUDA)

endif ()

# depend on 'cuda_runtime', if you only need to use cuda
# headers or link to cuda libs, but don't need to run your source
# through a cuda compiler (nvcc)
blt_register_library(NAME cuda_runtime
                     INCLUDES ${CUDA_INCLUDE_DIRS}
                     LIBRARIES ${CUDA_LIBRARIES}
                     DEFINES USE_CUDA)
