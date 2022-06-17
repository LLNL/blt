# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# Sanity Checks
################################

# Rare case of two flags being incompatible
if (DEFINED CMAKE_SKIP_BUILD_RPATH AND DEFINED BLT_CUDA_LINK_WITH_NVCC)
    if (NOT CMAKE_SKIP_BUILD_RPATH AND BLT_CUDA_LINK_WITH_NVCC)
        message( FATAL_ERROR
                         "CMAKE_SKIP_BUILD_RPATH (FALSE) and BLT_CUDA_LINK_WITH_NVCC (TRUE) "
                         "are incompatible when linking explicit shared libraries. Set "
                         "CMAKE_SKIP_BUILD_RPATH to TRUE.")
    endif()
endif()

# CUDA_HOST_COMPILER was changed in 3.9.0 to CMAKE_CUDA_HOST_COMPILER and
# needs to be set prior to enabling the CUDA language
get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )
    if ( NOT CMAKE_CUDA_HOST_COMPILER )
        if("CUDA" IN_LIST _languages )
            message( FATAL_ERROR 
                 "CUDA language enabled prior to setting CMAKE_CUDA_HOST_COMPILER. "
                 "Please set CMAKE_CUDA_HOST_COMPILER prior to "
                 "ENABLE_LANGUAGE(CUDA) or PROJECT(.. LANGUAGES CUDA)")
        endif()    
  
        if ( CMAKE_CXX_COMPILER )
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
        else()
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
        endif()
    endif()
else()
   if (NOT CUDA_HOST_COMPILER)
        if("CUDA" IN_LIST _languages )
            message( FATAL_ERROR 
                 "CUDA language enabled prior to setting CUDA_HOST_COMPILER. "
                 "Please set CUDA_HOST_COMPILER prior to "
                 "ENABLE_LANGUAGE(CUDA) or PROJECT(.. LANGUAGES CUDA)")
        endif()    

        if ( CMAKE_CXX_COMPILER )
            set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
        else()
            set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
        endif()
    endif()
endif()

# Override rpath link flags for nvcc
if (BLT_CUDA_LINK_WITH_NVCC)
    set(CMAKE_SHARED_LIBRARY_RUNTIME_CUDA_FLAG "-Xlinker -rpath -Xlinker " CACHE STRING "")
    set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CUDA_FLAG "-Xlinker -rpath -Xlinker " CACHE STRING "")
endif()


############################################################
# Basics
############################################################
enable_language(CUDA)

if(CMAKE_CUDA_STANDARD STREQUAL "17")
    if(NOT DEFINED CMAKE_CUDA_COMPILE_FEATURES OR (NOT "cuda_std_17" IN_LIST CMAKE_CUDA_COMPILE_FEATURES))
        message(FATAL_ERROR "CMake's CUDA_STANDARD does not support C++17.")
    endif()
endif()

if(CMAKE_CUDA_STANDARD STREQUAL "20")
    if(NOT DEFINED CMAKE_CUDA_COMPILE_FEATURES OR (NOT "cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES))
        message(FATAL_ERROR "CMake's CUDA_STANDARD does not support C++20.")
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

# Append the path to the NVIDIA SDK to the link flags
if(IS_DIRECTORY "${CUDAToolkit_LIBRARY_DIR}" )
    list(APPEND CMAKE_CUDA_LINK_FLAGS "-L${CUDAToolkit_LIBRARY_DIR}" )
endif()

message(STATUS "CUDA Version:       ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Bin Dir: ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDA Compiler:      ${CUDAToolkit_NVCC_EXECUTABLE}")
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )
    message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
else()
    message(STATUS "CUDA Host Compiler: ${CUDA_HOST_COMPILER}")
endif()
message(STATUS "CUDA Include Path:  ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Compile Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA Link Flags:    ${CMAKE_CUDA_LINK_FLAGS}")
message(STATUS "CUDA Separable Compilation:  ${CUDA_SEPARABLE_COMPILATION}")
message(STATUS "CUDA Link with NVCC:         ${BLT_CUDA_LINK_WITH_NVCC}")
message(STATUS "CUDA Implicit Link Libraries:   ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}")
message(STATUS "CUDA Implicit Link Directories: ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

# don't propagate host flags - too easy to break stuff!
set (CUDA_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
endif()

# Set PIE options to empty for PGI since it doesn't understand -fPIE This
# option is set in the CUDA toolchain file so must be unset after
# enable_language(CUDA)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  set(CMAKE_CUDA_COMPILE_OPTIONS_PIE "")
endif()


# cuda targets must be global for aliases when created as imported targets
set(_blt_cuda_is_global On)
if (${BLT_EXPORT_THIRDPARTY})
    set(_blt_cuda_is_global Off)
endif ()

blt_import_library(NAME          blt_cuda
                   LINK_FLAGS    "${CMAKE_CUDA_LINK_FLAGS}"
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY}
                   GLOBAL        ${_blt_cuda_is_global})

# Hard-copy inheritable properties instead of depending on CUDA::cudart so that we can export all required
# information in our target blt_cuda
blt_inherit_target_info(TO blt_cuda FROM CUDA::cudart OBJECT FALSE)

add_library(blt::cuda ALIAS blt_cuda)

blt_import_library(NAME          blt_cuda_runtime
                   TREAT_INCLUDES_AS_SYSTEM ON
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY}
                   GLOBAL        ${_blt_cuda_is_global})

blt_inherit_target_info(TO blt_cuda_runtime FROM CUDA::cudart OBJECT FALSE)

add_library(blt::cuda_runtime ALIAS blt_cuda_runtime)
