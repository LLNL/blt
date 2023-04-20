# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for using PGI and CUDA
#------------------------------------------------------------------------------
set(CMAKE_CXX_COMPILER "/usr/tce/packages/pgi/pgi-20.4/bin/pgc++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/pgi/pgi-20.4/bin/pgcc" CACHE PATH "")

set(ENABLE_FORTRAN OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# Extra options and flags
#------------------------------------------------------------------------------

set(ENABLE_OPENMP OFF CACHE BOOL "")
set(ENABLE_MPI OFF CACHE BOOL "")


#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------

set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.2.0" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda -G" CACHE STRING "")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

# Very specific fix for working around CMake adding implicit link directories returned by the BlueOS
# compilers to link CUDA executables 
set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")
