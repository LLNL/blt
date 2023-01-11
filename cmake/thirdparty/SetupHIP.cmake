# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# HIP
################################

if(NOT ROCM_PATH)
    find_path(ROCM_PATH
        hip
        ENV{ROCM_DIR}
        ENV{ROCM_PATH}
        ENV{HIP_PATH}
        ${HIP_PATH}/..
        ${HIP_ROOT_DIR}/../
        ${ROCM_ROOT_DIR}
        /opt/rocm)
endif()

# Update CMAKE_PREFIX_PATH to make sure all the configs that hip depends on are
# found.
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${ROCM_PATH}")

find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH})

message(STATUS "ROCM path:        ${ROCM_PATH}")
message(STATUS "HIP version:      ${hip_VERSION}")

if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21.0" )
   enable_language(HIP)
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
        if(NOT "${CMAKE_HIP_ARCHITECTURES}" MATCHES "${_target}")
            set(_flag "--offload-arch=${_target}")
            set(_generator_compile_flag "$<$<COMPILE_LANGUAGE:CXX>:SHELL:${_flag}>")
            set(_generator_link_flag "$<$<LINK_LANGUAGE:CXX>:${_flag}>")

            list(REMOVE_ITEM _hip_compile_options ${_generator_compile_flag})
            list(REMOVE_ITEM _hip_compile_options ${_flag})
            list(REMOVE_ITEM _hip_link_libs ${_generator_link_flag})
            list(REMOVE_ITEM _hip_link_libs ${_flag})
        endif()
    endforeach()

    if(BLT_BUILD_HIP_WITH_HIPCC)
        list(REMOVE_ITEM _hip_compile_options "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-mllvm")
        list(REMOVE_ITEM _hip_compile_options "-amdgpu-early-inline-all=true")
        list(REMOVE_ITEM _hip_compile_options "-mllvm")
        list(REMOVE_ITEM _hip_compile_options "-amdgpu-function-calls=false>")
        get_target_property(_hip_lang_compile_options hip-lang::device INTERFACE_COMPILE_OPTIONS)
        set(_flag "-mllvm-amdgpu-early-inline-all=true-mllvm-amdgpu-function-calls=false")
        set(_generator_compile_flag "$<$<COMPILE_LANGUAGE:HIP>:SHELL:${_flag}>")
        list(REMOVE_ITEM _hip_lang_compile_options "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-mllvm")
        list(REMOVE_ITEM _hip_lang_compile_options "-amdgpu-early-inline-all=true")
        list(REMOVE_ITEM _hip_lang_compile_options "-mllvm")
        list(REMOVE_ITEM _hip_lang_compile_options "-amdgpu-function-calls=false>")
        set_property(TARGET hip-lang::device PROPERTY INTERFACE_COMPILE_OPTIONS ${_hip_lang_compile_options})
    endif()

    set_property(TARGET hip::device PROPERTY INTERFACE_COMPILE_OPTIONS ${_hip_compile_options})
    set_property(TARGET hip::device PROPERTY INTERFACE_LINK_LIBRARIES ${_hip_link_libs})

    if(BLT_BUILD_HIP_WITH_HIPCC)
        #   remove interface include directories - these are bugged in ROCM 5.2.1
        set_property(TARGET hip-lang::device PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
        set_property(TARGET hip::device PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
        set_property(TARGET hip::host PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
        set_property(TARGET hip-lang::device PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
        set_property(TARGET hip::device PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
        set_property(TARGET hip::host PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
    endif()

    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
    endif()
endif()

# adds -x hip to the CL for hip-lang::device
if(NOT BLT_BUILD_HIP_WITH_HIPCC)
   get_target_property(_hip_lang_compile_options hip-lang::device INTERFACE_COMPILE_OPTIONS)
   message(${_hip_lang_compile_options})
   list(REMOVE_ITEM _hip_lang_compile_options "-mllvm")
   list(REMOVE_ITEM _hip_lang_compile_options "-amdgpu-function-calls=false>")
   list(APPEND _hip_lang_compile_options "-x")
   list(APPEND _hip_lang_compile_options "hip")
   list(APPEND _hip_lang_compile_options "-mllvm")
   list(APPEND _hip_lang_compile_options "-amdgpu-function-calls=false>")
   set_property(TARGET hip-lang::device PROPERTY INTERFACE_COMPILE_OPTIONS ${_hip_lang_compile_options})
endif()

# hip targets must be global for aliases when created as imported targets
set(_blt_hip_is_global On)
if(${BLT_EXPORT_THIRDPARTY})
    set(_blt_hip_is_global Off)
endif()

set(BLT_ADD_ROCM_PATH_COMPILER_FLAG CACHE BOOL ON "")
if (BLT_ADD_ROCM_PATH_COMPILER_FLAG)
# Guard against `--rocm-path` being added to crayftn less than version 15.0.0 due to
# invalid command line option error
if(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray" AND CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 15.0.0)
    set(_blt_hip_compile_flags "$<$<COMPILE_LANGUAGE:CXX>:SHELL:--rocm-path=${ROCM_PATH}>")
else()
    set(_blt_hip_compile_flags "--rocm-path=${ROCM_PATH}")
endif()
else()
   set(_blt_hip_compile_flags "")
endif()

blt_import_library(NAME          blt_hip
                   COMPILE_FLAGS ${_blt_hip_compile_flags}
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY}
                   GLOBAL        ${_blt_hip_is_global})

# Hard-copy inheritable properties instead of depending on hip::device so that we can export all required
# information in our target blt_hip
blt_inherit_target_info(TO blt_hip FROM hip::device OBJECT FALSE)

add_library(blt::hip ALIAS blt_hip)

blt_import_library(NAME          blt_hip_runtime
                   INCLUDES      ${HIP_INCLUDE_DIRS}
                   TREAT_INCLUDES_AS_SYSTEM ON
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY}
                   GLOBAL        ${_blt_hip_is_global})

blt_inherit_target_info(TO blt_hip_runtime FROM hip::host OBJECT FALSE)

add_library(blt::hip_runtime ALIAS blt_hip_runtime)
