# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#####################################################
# Set some variables to simplify determining compiler
# Compiler string list from:
#   https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
####################################################3

# use CMAKE_BUILD_TOOL to identify visual studio
# and CMAKE_CXX_COMPILER_ID for all other cases

if("${CMAKE_BUILD_TOOL}" MATCHES "(msdev|devenv|nmake|MSBuild)")
    set(COMPILER_FAMILY_IS_MSVC 1)
    message(STATUS "Compiler family is MSVC")

    if(CMAKE_GENERATOR_TOOLSET AND "${CMAKE_GENERATOR_TOOLSET}" MATCHES "Intel")
        set(COMPILER_FAMILY_IS_MSVC_INTEL 1) 
        message(STATUS "Toolset is ${CMAKE_GENERATOR_TOOLSET}")
    endif()
else()
    #Determine C/C++ compiler family. 
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(C_COMPILER_FAMILY_IS_GNU 1)
        message(STATUS "C Compiler family is GNU")

    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") # For Clang or AppleClang
        set(C_COMPILER_FAMILY_IS_CLANG 1)
        message(STATUS "C Compiler family is Clang")

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
        set(C_COMPILER_FAMILY_IS_XL 1)
        message(STATUS "C Compiler family is XL")

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(C_COMPILER_FAMILY_IS_INTEL 1)
        message(STATUS "C Compiler family is Intel")

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
        set(C_COMPILER_FAMILY_IS_INTELLLVM 1)
        message(STATUS "C Compiler family is IntelLLVM")

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
        set(C_COMPILER_FAMILY_IS_PGI 1)
        message(STATUS "C Compiler family is PGI")

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
        set(C_COMPILER_FAMILY_IS_CRAY 1)
        message(STATUS "C Compiler family is Cray")

    else()
        message(STATUS "C Compiler family not set!!!")
    endif()
    # Determine Fortran compiler family 
    if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
        set(Fortran_COMPILER_FAMILY_IS_GNU 1)
        message(STATUS "Fortran Compiler family is GNU")

    elseif("${CMAKE_Fortran_COMPILER_ID}" MATCHES "Clang") # For Clang or AppleClang
        set(Fortran_COMPILER_FAMILY_IS_CLANG 1)
        message(STATUS "Fortran Compiler family is Clang")

    elseif("${CMAKE_Fortran_COMPILER_ID}" MATCHES "Flang") # For Flang compilers
        set(Fortran_COMPILER_FAMILY_IS_CLANG 1 CACHE BOOL "")
        message(STATUS "Fortran Compiler family is Clang")

    elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "XL")
        set(Fortran_COMPILER_FAMILY_IS_XL 1)
        message(STATUS "Fortran Compiler family is XL")

    elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
        set(Fortran_COMPILER_FAMILY_IS_INTEL 1)
        message(STATUS "Fortran Compiler family is Intel")

    elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "IntelLLVM")
        set(Fortran_COMPILER_FAMILY_IS_INTELLLVM 1)
        message(STATUS "Fortran Compiler family is IntelLLVM")

    elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
        set(Fortran_COMPILER_FAMILY_IS_PGI 1)
        message(STATUS "Fortran Compiler family is PGI")

    elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray")
        set(Fortran_COMPILER_FAMILY_IS_CRAY 1)
        message(STATUS "Fortran Compiler family is Cray")

    elseif(ENABLE_FORTRAN)
        message(STATUS "Fortran Compiler family not set!!!")
    endif()
endif()
