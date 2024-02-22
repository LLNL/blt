# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#################################################
# OpenMP
# (OpenMP support is provided by the compiler)
#################################################

find_package(OpenMP REQUIRED)

# check if the openmp flags used for C/C++ are different from the openmp flags
# used by the Fortran compiler
set(_flags_differ FALSE)
if(BLT_ENABLE_FORTRAN)
    string(COMPARE NOTEQUAL "${OpenMP_CXX_FLAGS}" "${OpenMP_Fortran_FLAGS}"
           _flags_differ)
endif()
set(BLT_OPENMP_FLAGS_DIFFER ${_flags_differ} CACHE BOOL "")

set(_flags ${OpenMP_CXX_FLAGS})

if(NOT COMPILER_FAMILY_IS_MSVC)
    if(BLT_ENABLE_CUDA AND BLT_OPENMP_FLAGS_DIFFER)
        set(_flags
            $<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<NOT:$<COMPILE_LANGUAGE:Fortran>>>:${OpenMP_CXX_FLAGS}> 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
            $<$<COMPILE_LANGUAGE:Fortran>:${OpenMP_Fortran_FLAGS}>)
    elseif(BLT_ENABLE_CUDA)
        set(_flags
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${OpenMP_CXX_FLAGS}> 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    elseif(BLT_OPENMP_FLAGS_DIFFER)
        set(_flags
            $<$<NOT:$<COMPILE_LANGUAGE:Fortran>>:${OpenMP_CXX_FLAGS}>
            $<$<COMPILE_LANGUAGE:Fortran>:${OpenMP_Fortran_FLAGS}>)
    endif()
endif()

set(_compile_flags ${_flags})
set(_link_flags ${_flags})

# Allow user to override
if (BLT_OPENMP_COMPILE_FLAGS)
    set(_compile_flags ${BLT_OPENMP_COMPILE_FLAGS})
endif()
if (BLT_OPENMP_LINK_FLAGS)
    set(_link_flags ${BLT_OPENMP_LINK_FLAGS})
endif()


message(STATUS "BLT OpenMP Compile Flags: ${_compile_flags}")
message(STATUS "BLT OpenMP Link Flags:    ${_link_flags}")

blt_import_library(NAME          openmp
                   COMPILE_FLAGS ${_compile_flags}
                   LINK_FLAGS    ${_link_flags}
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY})

add_library(blt::openmp ALIAS openmp)
