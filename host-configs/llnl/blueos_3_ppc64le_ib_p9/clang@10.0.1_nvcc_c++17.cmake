# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for the blue_os cluster at LLNL, specifically Lassen
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C/C++:   Clang with GCC 8.3.1 toolchain
#  Cuda
#  MPI
# 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------

set(CLANG_HOME "/usr/tce/packages/clang/clang-ibm-10.0.1-gcc-8.3.1")
set(CMAKE_C_COMPILER   "${CLANG_HOME}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${CLANG_HOME}/bin/clang++" CACHE PATH "")
set(BLT_CXX_STD "c++17" CACHE STRING "")

# Disable Fortran
set(ENABLE_FORTRAN OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME               "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1")
set(MPI_C_COMPILER         "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER       "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")

set(MPIEXEC                "${MPI_HOME}/bin/mpirun"  CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np"     CACHE PATH "")
set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE PATH "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
# _blt_tutorial_cuda_config_start
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.2.0" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda -G" CACHE STRING "")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )

# _blt_tutorial_cuda_config_end

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

# Very specific fix for working around CMake adding implicit link directories returned by the BlueOS
# compilers to link CUDA executables 
set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")
