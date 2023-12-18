# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Host-config file for executing tests of BLT's configuration of CUDA targets on Lassen.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------

set(BLT_CXX_STD "c++14" CACHE STRING "")
set(CLANG_HOME "/usr/tce/packages/clang/clang-16.0.6")
set(CMAKE_C_COMPILER   "${CLANG_HOME}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${CLANG_HOME}/bin/clang++" CACHE PATH "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.1.0" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "60" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda -G" CACHE STRING "")
