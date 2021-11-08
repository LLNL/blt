# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for the rznevada cluster at LLNL
#------------------------------------------------------------------------------
# This file provides CMake with paths / details for:
#  C,C++, & Fortran compilers + MPI & HIP
# using ROCM compilers
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# ROCM@4.2.0 compilers
#------------------------------------------------------------------------------
set(ROCM_HOME "/usr/tce/packages/rocmcc-tce/rocmcc-4.2.0")
set(CMAKE_C_COMPILER   "${ROCM_HOME}/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${ROCM_HOME}/bin/hipcc" CACHE PATH "")

# Fortran support
set(CCE_HOME "/usr/tce/packages/cce-tce/cce-12.0.1")
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(CMAKE_Fortran_COMPILER "${CCE_HOME}/bin/crayftn" CACHE PATH "")

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME "/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-rocmcc-4.2.0")
set(MPI_C_COMPILER "${MPI_HOME}/bin/mpicc" CACHE PATH "")
set(MPI_CXX_COMPILER "${MPI_HOME}/bin/mpicxx" CACHE PATH "")
set(MPI_Fortran_COMPILER "${MPI_HOME}/bin/mpif90" CACHE PATH "")

#------------------------------------------------------------------------------
# HIP support
#------------------------------------------------------------------------------
set(ENABLE_HIP ON CACHE BOOL "")
set(ROCM_PATH "/opt/rocm-4.2.0/" CACHE PATH "")
set(CMAKE_CUDA_ARCHITECTURES "gfx908" CACHE STRING "")
