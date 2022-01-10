# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Author: Noel Chalmers @ Advanced Micro Devices, Inc.
# Date: March 11, 2019

################################
# HIP
################################
find_package(hip REQUIRED)

message(STATUS "HIP version:      ${hip_VERSION")
message(STATUS "HIP platform:     ${HIP_PLATFORM}")

if (NOT ROCM_PATH)
    find_path(ROCM_PATH
        hip
        ENV ROCM_DIR
        ENV ROCM_PATH
        ${HIP_ROOT_DIR}/../
        ${ROCM_ROOT_DIR}
        /opt/rocm)
endif()

# AMDGPU_TARGETS should be defined in the hip-config.cmake that gets "included" via find_package(hip)
# This file is also what hardcodes the --offload-arch flags we're removing here
if(DEFINED AMDGPU_TARGETS)
    # If we haven't selected a particular architecture via CMAKE_HIP_ARCHITECTURES,
    # we want to remove the unconditionally added compile/link flags from the hip::device target.
    # FIXME: This may cause problems for targets whose HIP_ARCHITECTURES property differs
    # from CMAKE_HIP_ARCHITECTURES - this only happens when a user manually modifies
    # the property after it is initialized
    get_target_property(_hip_compile_options hip::device INTERFACE_COMPILE_OPTIONS)
    get_target_property(_hip_link_libs hip::device INTERFACE_LINK_LIBRARIES)

    foreach(_target ${AMDGPU_TARGETS})
        if (NOT "${CMAKE_HIP_ARCHITECTURES}" MATCHES "${_target}")
            list(REMOVE_ITEM _hip_compile_options "--offload-arch=${_target}")
            list(REMOVE_ITEM _hip_link_libs "--offload-arch=${_target}")
        endif()
    endforeach()
    
    set_property(TARGET hip::device PROPERTY INTERFACE_COMPILE_OPTIONS ${_hip_compile_options})
    set_property(TARGET hip::device PROPERTY INTERFACE_LINK_LIBRARIES ${_hip_link_libs})

    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
    endif()
endif()

blt_import_library(NAME       blt_hip
                   COMPILE_FLAGS "--rocm-path=${ROCM_PATH}"
                   EXPORTABLE ${BLT_EXPORT_THIRDPARTY})

blt_inherit_target_info(TO blt_hip FROM hip::device OBJECT FALSE)

add_library(blt::hip ALIAS blt_hip)

blt_import_library(NAME          blt_hip_runtime
                   INCLUDES ${HIP_INCLUDE_DIRS}
                   TREAT_INCLUDES_AS_SYSTEM ON
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY})

blt_inherit_target_info(TO blt_hip_runtime FROM hip::host OBJECT FALSE)

add_library(blt::hip_runtime ALIAS blt_hip_runtime)
