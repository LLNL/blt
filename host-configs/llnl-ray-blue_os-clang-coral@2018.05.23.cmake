#------------------------------------------------------------------------------
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-725085
#
# All rights reserved.
#
# This file is part of BLT.
#
# For additional details, please also read BLT/LICENSE.
#------------------------------------------------------------------------------
# Example host-config file for the blue_os ray cluster at LLNL
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C,C++, MPI and Cuda
# 
#------------------------------------------------------------------------------

set(COMPILER_HOME "/usr/tce/packages/clang/clang-coral-2018.05.23")

# c compiler
set(CMAKE_C_COMPILER "${COMPILER_HOME}/bin/clang" CACHE PATH "")
# cpp compiler
set(CMAKE_CXX_COMPILER "${COMPILER_HOME}/bin/clang++" CACHE PATH "")

set(ENABLE_FORTRAN OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME               "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-coral-2018.05.23")
set(MPI_C_COMPILER         "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER       "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")

set(MPIEXEC                "mpirun"  CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np"     CACHE PATH "")
set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE PATH "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-9.2.148" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")

set (CUDA_ARCH "sm_60" CACHE PATH "")
set (CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} -std=c++11 --expt-extended-lambda -G" CACHE STRING "" )
set (CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})

set (CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
set (CUDA_LINK_WITH_NVCC ON CACHE BOOL "")
# set the link flags manually since nvcc will link (and not have the wrappers knowledge)
# on ray - can figure out your equivalant flags by doing mpicc -vvvv
set (SPECTRUM_ROLLING "/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release")
set (CMAKE_CUDA_LINK_FLAGS "-Xlinker -rpath -Xlinker ${SPECTRUM_ROLLING}/lib -Xlinker -rpath -Xlinker ${COMPILER_HOME}/ibm/lib:/usr/tce/packages/gcc/gcc-4.9.3/lib64 -L${SPECTRUM_ROLLING}/lib/ -lmpi_ibm" CACHE STRING "")

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

