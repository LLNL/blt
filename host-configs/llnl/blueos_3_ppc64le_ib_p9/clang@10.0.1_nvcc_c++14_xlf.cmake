# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for the blue_os cluster at LLNL
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C/C++:   Clang
#  Fortran: XLF
#  MPI
#  Cuda
# 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------

# Use Clang compilers for C/C++
set(CLANG_HOME "/usr/tce/packages/clang/clang-ibm-10.0.1-gcc-8.3.1")
set(CMAKE_C_COMPILER   "${CLANG_HOME}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${CLANG_HOME}/bin/clang++" CACHE PATH "")
set(BLT_CXX_STD "c++14" CACHE STRING "")

# Use XL compiler for Fortran
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(XL_VERSION "xl-2019.06.12")
set(XL_HOME "/usr/tce/packages/xl/${XL_VERSION}")
set(CMAKE_Fortran_COMPILER "${XL_HOME}/bin/xlf2003" CACHE PATH "")

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME               "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1")
set(MPI_C_COMPILER         "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER       "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")

set(MPI_Fortran_HOME       "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1")
set(MPI_Fortran_COMPILER   "${MPI_Fortran_HOME}/bin/mpif90" CACHE PATH "")

set(MPIEXEC                "${MPI_HOME}/bin/mpirun"  CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np"     CACHE PATH "")
set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE PATH "")

#------------------------------------------------------------------------------
# Other machine specifics
#------------------------------------------------------------------------------

set(CMAKE_Fortran_COMPILER_ID "XL" CACHE PATH "All of BlueOS compilers report clang due to nvcc, override to proper compiler family")
set(BLT_FORTRAN_FLAGS "-WF,-C!" CACHE PATH "Converts C-style comments to Fortran style in preprocessed files")

set(BLT_CMAKE_IMPLICIT_LINK_LIBRARIES_EXCLUDE "xlomp_ser" CACHE STRING "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
#_blt_tutorial_useful_cuda_flags_start
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.2.0" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda -G" CACHE STRING "")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")
#_blt_tutorial_useful_cuda_flags_end

# Very specific fix for working around CMake adding implicit link directories returned by the BlueOS
# compilers to link CUDA executables 
set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")
