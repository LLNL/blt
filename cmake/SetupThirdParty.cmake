# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

include("${CMAKE_CURRENT_LIST_DIR}/BLTSetupTargets.cmake")

#------------------------------------
# Git
#------------------------------------
if (ENABLE_GIT)
    find_package(Git)
    if (Git_FOUND)
        message(STATUS "Git Support is ON")
        set(GIT_FOUND TRUE)
        message(STATUS "Git Executable: " ${GIT_EXECUTABLE} )
        message(STATUS "Git Version: " ${GIT_VERSION_STRING} )
    else()
        message(STATUS "Git Support is OFF")
    endif()
else()
    message(STATUS "Git Support is OFF")
endif()


#------------------------------------
# Documentation Packages
#------------------------------------
if (ENABLE_DOXYGEN)
    find_package(Doxygen)
endif()

blt_find_executable(NAME        Sphinx
                    EXECUTABLES sphinx-build sphinx-build2)


#------------------------------------
# Valgrind
#------------------------------------
blt_find_executable(NAME        Valgrind
                    EXECUTABLES valgrind)


#------------------------------------
# linting
#------------------------------------
blt_find_executable(NAME        AStyle
                    EXECUTABLES astyle)

blt_find_executable(NAME        ClangFormat
                    EXECUTABLES clang-format)

blt_find_executable(NAME        Uncrustify
                    EXECUTABLES uncrustify)

blt_find_executable(NAME        Yapf
                    EXECUTABLES yapf)

blt_find_executable(NAME        CMakeFormat
                    EXECUTABLES cmake-format)


#------------------------------------
# Static analysis via Cppcheck
#------------------------------------
blt_find_executable(NAME        Cppcheck
                    EXECUTABLES cppcheck)


#------------------------------------
# Static analysis via clang-query and clang-tidy
#------------------------------------
if(CMAKE_GENERATOR STREQUAL "Unix Makefiles" OR CMAKE_GENERATOR STREQUAL "Ninja")
    blt_find_executable(NAME        ClangQuery
                        EXECUTABLES clang-query)

    blt_find_executable(NAME        ClangTidy
                        EXECUTABLES clang-tidy)

    blt_find_executable(NAME        ClangApplyReplacements
                        EXECUTABLES clang-apply-replacements)
endif()

#------------------------------------
# Code coverage
#------------------------------------
if (ENABLE_COVERAGE)
    # Attempt to find the executables associated with gcov, lcov and genhtml.
    # This requires that the associated features are enabled.
    set(ENABLE_GCOV ON CACHE BOOL "")
    set(ENABLE_LCOV ON CACHE BOOL "")
    set(ENABLE_GENHTML ON CACHE BOOL "")
    blt_find_executable(NAME        gcov
                        EXECUTABLES gcov)

    blt_find_executable(NAME        lcov
                        EXECUTABLES lcov)

    blt_find_executable(NAME        genhtml
                        EXECUTABLES genhtml)
endif()
