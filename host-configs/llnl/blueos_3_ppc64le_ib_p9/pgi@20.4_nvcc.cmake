# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for using PGI and CUDA
#------------------------------------------------------------------------------
set(CMAKE_CXX_COMPILER "/usr/tce/packages/pgi/pgi-20.4/bin/pgc++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/pgi/pgi-20.4/bin/pgcc" CACHE PATH "")

set(ENABLE_FORTRAN OFF CACHE BOOL "")
set(ENABLE_MPI OFF CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.1.1" CACHE PATH "")
set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE BOOL "")
