# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

# This file is intended to be included in the *-config.cmake files of
# any project using a third-party library.  The macro 
# `blt_install_tpl_setups(DESTINATION <dir>)`  installs this file
# into the destination specified by the argument <dir>.

# BLTInstallableMacros provides helper macros for setting up and creating
# third-party library targets.
include("${CMAKE_CURRENT_LIST_DIR}/BLTInstallableMacros.cmake")
# If the generated TPL config file exists, include it here.
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/BLT-TPL-config.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/BLT-TPL-config.cmake")
endif()

# Detect if Fortran has been introduced.
get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(_languages MATCHES "Fortran")
    set(_fortran_already_enabled TRUE)
else()
    set(_fortran_already_enabled FALSE)
endif()

# Only update ENABLE_FORTRAN if it is a new requirement, don't turn 
# the flag off if required by an upstream dependency.
if (NOT ENABLE_FORTRAN)
  set(ENABLE_FORTRAN ${_fortran_already_enabled})
endif()

#------------------------------------
# MPI
#------------------------------------
message(STATUS "MPI Support is ${ENABLE_MPI}")
if (ENABLE_MPI AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupMPI.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupMPI.cmake")
endif()


#------------------------------------
# OpenMP
#------------------------------------
message(STATUS "OpenMP Support is ${ENABLE_OPENMP}")
if (ENABLE_OPENMP AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupOpenMP.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupOpenMP.cmake")
endif()


#------------------------------------
# CUDA
#------------------------------------
message(STATUS "CUDA Support is ${ENABLE_CUDA}")
if (ENABLE_CUDA AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupCUDA.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupCUDA.cmake")
endif()


#------------------------------------
# HIP
#------------------------------------
message(STATUS "HIP Support is ${ENABLE_HIP}")
if (ENABLE_HIP AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupHIP.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/thirdparty/BLTSetupHIP.cmake")
endif()
