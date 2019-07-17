# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for the blue_os cluster at LLNL
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C/C++, OpenMP, MPI, and Cuda
# 
# It demonstrates a more complex build, where the code is linked with nvcc.
#------------------------------------------------------------------------------
# Warning: This host-config does not currently work with CMake@3.13 or above
#          due to a linking error. 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------

set(CLANG_VERSION "clang-upstream-2019.03.26" CACHE STRING "")
set(CLANG_HOME "/usr/tce/packages/clang/${CLANG_VERSION}")

set(CMAKE_C_COMPILER   "${CLANG_HOME}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${CLANG_HOME}/bin/clang++" CACHE PATH "")

set(BLT_CXX_STD "c++11" CACHE STRING "")
set(ENABLE_FORTRAN OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME               "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-${CLANG_VERSION}")
set(MPI_C_COMPILER         "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER       "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")

set(MPIEXEC                "${MPI_HOME}/bin/mpirun"  CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np"     CACHE PATH "")
set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE PATH "")
set(BLT_MPI_LINK_FLAGS     "-Xlinker -rpath -Xlinker ${MPI_HOME}/lib" CACHE STRING "")

#------------------------------------------------------------------------------
# OpenMP support
#------------------------------------------------------------------------------
set(ENABLE_OPENMP ON CACHE BOOL "")

# Override default link flags because linking with nvcc
set(OMP_HOME ${CLANG_HOME}/ibm/omprtl)
set(BLT_OPENMP_LINK_FLAGS "-Xlinker -rpath -Xlinker ${OMP_HOME}/lib -L${OMP_HOME}/lib -lomp -lomptarget-nvptx" CACHE STRING "")


#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-9.2.148" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER} CACHE PATH "")

set (_cuda_arch "sm_60")
set (CMAKE_CUDA_FLAGS "-restrict -arch ${_cuda_arch} -std=c++11 --expt-extended-lambda -G" CACHE STRING "" )

set (CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
set (CUDA_LINK_WITH_NVCC ON CACHE BOOL "")
# set the link flags manually since nvcc will link (and not have the wrappers knowledge)
# on ray - can figure out your equivalant flags by doing mpicc -vvvv
set (CMAKE_CUDA_LINK_FLAGS "-Xlinker -rpath -Xlinker ${MPI_HOME}/lib -Xlinker -rpath -Xlinker ${CLANG_HOME}/ibm/lib:/usr/tce/packages/gcc/gcc-4.9.3/lib64 -L${MPI_HOME}/lib/ -lmpi_ibm" CACHE STRING "")


# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

