# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#################################################
# OpenMP
# (OpenMP support is provided by the compiler)
#################################################

find_package(OpenMP REQUIRED)

# avoid generator expressions if possible, as generator expressions can be 
# passed as flags to downstream projects that might not be using the same
# languages. See https://github.com/LLNL/blt/issues/205
if (ENABLE_FORTRAN AND NOT OpenMP_CXX_FLAGS STREQUAL OpenMP_Fortran_FLAGS)
   set(ESCAPE_FORTRAN ON)
else()
   set(ESCAPE_FORTRAN OFF)
endif()

set(_compile_flags ${OpenMP_CXX_FLAGS})
set(_link_flags  ${OpenMP_CXX_FLAGS})

if(NOT COMPILER_FAMILY_IS_MSVC AND ENABLE_CUDA AND ESCAPE_FORTRAN)
    set(_compile_flags
        $<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<NOT:$<COMPILE_LANGUAGE:Fortran>>>:${OpenMP_CXX_FLAGS}> 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:Fortran>:${OpenMP_Fortran_FLAGS}>)
elseif(NOT COMPILER_FAMILY_IS_MSVC AND ENABLE_CUDA)
    set(_compile_flags
        $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${OpenMP_CXX_FLAGS}> 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
elseif(NOT COMPILER_FAMILY_IS_MSVC AND ESCAPE_FORTRAN)
    set(_compile_flags
        $<$<NOT:$<COMPILE_LANGUAGE:Fortran>>:${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:Fortran>:${OpenMP_Fortran_FLAGS}>)
endif()


# Allow user to override
if (BLT_OPENMP_COMPILE_FLAGS)
    set(_compile_flags ${BLT_OPENMP_COMPILE_FLAGS})
endif()
if (BLT_OPENMP_LINK_FLAGS)
    set(_link_flags ${BLT_OPENMP_LINK_FLAGS})
endif()


message(STATUS "OpenMP Compile Flags: ${_compile_flags}")
message(STATUS "OpenMP Link Flags:    ${_link_flags}")

blt_register_library(NAME openmp
                     COMPILE_FLAGS ${_compile_flags} 
                     LINK_FLAGS    ${_link_flags})
