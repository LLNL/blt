# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

####################################
# BLT 3rd Party Lib Support
####################################

################################
# Git
################################
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

################################
# MPI
################################
message(STATUS "MPI Support is ${ENABLE_MPI}")
if (ENABLE_MPI)
    include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupMPI.cmake)
endif()

################################
# CUDA
################################
message(STATUS "CUDA Support is ${ENABLE_CUDA}")
if (ENABLE_CUDA)
  include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupCUDA.cmake)
endif()

################################
# HIP
################################
message(STATUS "HIP Support is ${ENABLE_HIP}")
if (ENABLE_HIP)
  include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupHIP.cmake)
endif()

################################
# HCC
################################
message(STATUS "HCC Support is ${ENABLE_HCC}")
if (ENABLE_HCC)
  include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupROCm.cmake)
endif()

################################
# Documentation Packages
################################
if (ENABLE_DOXYGEN)
    find_package(Doxygen)
endif()

blt_find_executable(NAME        Sphinx
                    EXECUTABLES sphinx-build sphinx-build2)

################################
# Valgrind
################################
blt_find_executable(NAME        Valgrind
                    EXECUTABLES valgrind)

################################
# linting
################################

blt_find_executable(NAME        Uncrustify
                    EXECUTABLES uncrustify)

blt_find_executable(NAME        AStyle
                    EXECUTABLES astyle)

################################
# Static analysis via Cppcheck
################################
blt_find_executable(NAME        Cppcheck
                    EXECUTABLES cppcheck)

################################
# Static analysis via clang-query
################################
if(CMAKE_GENERATOR STREQUAL "Unix Makefiles" OR CMAKE_GENERATOR STREQUAL "Ninja")
    blt_find_executable(NAME        ClangQuery
                        EXECUTABLES clang-query)
endif()
