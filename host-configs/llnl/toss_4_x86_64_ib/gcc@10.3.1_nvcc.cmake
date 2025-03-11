# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for a cluster on a toss4 platform (e.g. quartz) at LLNL
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C,C++, & Fortran compilers + MPI
# 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# gcc@10.3.1 compilers
#------------------------------------------------------------------------------

# _blt_pascal_compiler_config_start
set(GCC_VERSION "gcc-10.3.1")
set(GCC_HOME "/usr/tce/packages/gcc/${GCC_VERSION}")

set(CMAKE_C_COMPILER "${GCC_HOME}/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${GCC_HOME}/bin/g++" CACHE PATH "")

# Fortran support
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(CMAKE_Fortran_COMPILER "${GCC_HOME}/bin/gfortran" CACHE PATH "")
# _blt_pascal_compiler_config_end

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME             "/usr/tce/packages/mvapich2/mvapich2-2.3.6-${GCC_VERSION}" CACHE PATH "")

set(MPI_C_COMPILER       "${MPI_HOME}/bin/mpicc" CACHE PATH "")
set(MPI_CXX_COMPILER     "${MPI_HOME}/bin/mpicxx" CACHE PATH "")
set(MPI_Fortran_COMPILER "${MPI_HOME}/bin/mpif90" CACHE PATH "")

set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-12.2.2" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda -G" CACHE STRING "")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
