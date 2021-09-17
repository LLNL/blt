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

message(STATUS "HIP version:      ${HIP_VERSION_STRING}")
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

blt_import_library(NAME       blt::hip
                   DEPENDS_ON hip::device
                   COMPILE_FLAGS "--rocm-path=${ROCM_PATH}"
                   EXPORTABLE ${BLT_EXPORT_THIRDPARTY})

# depend on 'hip_runtime', if you only need to use hip
# headers or link to hip libs, but don't need to run your source
# through a hip compiler (hipcc)
blt_import_library(NAME          blt::hip_runtime
                   DEPENDS_ON    hip::host
                   TREAT_INCLUDES_AS_SYSTEM ON
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY})
