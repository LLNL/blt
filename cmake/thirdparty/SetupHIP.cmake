# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Author: Noel Chalmers @ Advanced Micro Devices, Inc.
# Date: March 11, 2019

################################
# HIP
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(HIP REQUIRED)

message(STATUS "HIP version:      ${HIP_VERSION_STRING}")
message(STATUS "HIP platform:     ${HIP_PLATFORM}")

set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
if(${HIP_PLATFORM} STREQUAL "hcc")
	set(HIP_RUNTIME_DEFINES "-D__HIP_PLATFORM_HCC__")
    find_library(HIP_RUNTIME_LIBRARIES NAMES hip_hcc libhip_hcc
                PATHS ${HIP_ROOT_DIR}/lib
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH)
    set(HIP_RUNTIME_LIBRARIES "${HIP_ROOT_DIR}/lib/libhip_hcc.so")
elseif(${HIP_PLATFORM} STREQUAL "clang" OR ${HIP_PLATFORM} STREQUAL "amd")
    set(HIP_RUNTIME_DEFINES "-D__HIP_PLATFORM_HCC__;-D__HIP_ROCclr__;-D__HIP_PLATFORM_AMD__")
    find_library(HIP_RUNTIME_LIBRARIES NAMES amdhip64 libamdhip64
                PATHS ${HIP_ROOT_DIR}/lib
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH)
elseif(${HIP_PLATFORM} STREQUAL "nvcc" OR ${HIP_PLATFORM} STREQUAL "nvidia")
    set(HIP_RUNTIME_DEFINES "-D__HIP_PLATFORM_NVCC__;-D__HIP_PLATFORM_NVIDIA__")
    if (${CMAKE_VERSION} VERSION_LESS "3.17.0")
        find_package(CUDA)
        find_library(HIP_RUNTIME_LIBRARIES NAMES cudart libcudart
            PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH)
        set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_RUNTIME_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS}")
    else()
        find_package(CUDAToolkit)
        find_library(HIP_RUNTIME_LIBRARIES NAMES cudart libcudart
            PATHS ${CUDAToolkit_LIBRARY_DIR}
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH)
        set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_RUNTIME_INCLUDE_DIRS};${CUDAToolkit_INCLUDE_DIR}")
    endif()
endif()

if ( IS_DIRECTORY "${HIP_ROOT_DIR}/hcc/include" ) # this path only exists on older rocm installs
    set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_ROOT_DIR}/include;${HIP_ROOT_DIR}/hcc/include" CACHE STRING "")
else()
    set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" CACHE STRING "")
endif()
set(HIP_RUNTIME_COMPILE_FLAGS "${HIP_RUNTIME_COMPILE_FLAGS};-Wno-unused-parameter")

set(_hip_compile_flags " ")
if (ENABLE_CLANG_HIP)
    if (NOT (${HIP_PLATFORM} STREQUAL "clang"))
        message(FATAL_ERROR "ENABLE_CLANG_HIP requires HIP_PLATFORM=clang")
    endif()
    set(_hip_compile_flags -x;hip)
    # Using clang HIP, we need to construct a few CPP defines and compiler flags
    foreach(_arch ${BLT_CLANG_HIP_ARCH})
        string(TOUPPER ${_arch} _UPARCH)
        string(TOLOWER ${_arch} _lowarch)
        list(APPEND _hip_compile_flags "--offload-arch=${_lowarch}")
        set(_hip_compile_defines "${HIP_RUNTIME_DEFINES};-D__HIP_ARCH_${_UPARCH}__=1")
    endforeach(_arch)
    
    # We need to pass rocm path as well, for certain bitcode libraries.
    # First see if we were given it, then see if it exists in the environment.
    # If not, don't try to guess but print a warning and hope the compiler knows where it is.
    if (NOT ROCM_PATH)
        find_path(ROCM_PATH
            bin/rocminfo
            ENV ROCM_DIR
            ENV ROCM_PATH
            ${HIP_ROOT_DIR}/../
            ${ROCM_ROOT_DIR}
            /opt/rocm)
    endif()

    if(DEFINED ROCM_PATH)
        list(APPEND _hip_compile_flags "--rocm-path=${ROCM_PATH}")
    else()
        message(WARN "ROCM_PATH not set or found! This is typically required for Clang HIP Compilation")
    endif()

    message(STATUS "Clang HIP Enabled. Clang flags for HIP compilation: ${_hip_compile_flags}")
    message(STATUS "Defines for HIP compilation: ${_hip_compile_defines}")

    blt_import_library(NAME             hip
                       DEFINES          ${_hip_compile_defines}
                       COMPILE_FLAGS    ${_hip_compile_flags}
                       DEPENDS_ON       ${HIP_RUNTIME_LIBRARIES})
else()

# depend on 'hip', if you need to use hip
# headers, link to hip libs, and need to run your source
# through a hip compiler (hipcc)
# This is currently used only as an indicator for blt_add_hip* -- FindHIP/hipcc will handle resolution
# of all required HIP-related includes/libraries/flags.
    blt_import_library(NAME      hip)
endif()


# depend on 'hip_runtime', if you only need to use hip
# headers or link to hip libs, but don't need to run your source
# through a hip compiler (hipcc)
blt_import_library(NAME          hip_runtime
                   INCLUDES      ${HIP_RUNTIME_INCLUDE_DIRS}
                   DEFINES       ${HIP_RUNTIME_DEFINES}
                   COMPILE_FLAGS ${HIP_RUNTIME_COMPILE_FLAGS}
                   DEPENDS_ON    ${HIP_RUNTIME_LIBRARIES}
                   TREAT_INCLUDES_AS_SYSTEM ON
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY})
