# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for the blue_os cluster at LLNL
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C/C++:   Clang with GCC 8.3.1 toolchain with C++14 support
#  Cuda
# 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
set(COMPILER_BASE "/usr/tce/packages/clang/clang-ibm-14.0.5" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/clang++" CACHE PATH "")

set(GCC_HOME "/usr/tce/packages/gcc/gcc-8.3.1" CACHE PATH "")
set(CMAKE_C_FLAGS "--gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "--gcc-toolchain=${GCC_HOME}" CACHE STRING "")

set(BLT_CXX_STD "c++14" CACHE STRING "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA")

set(CMAKE_CUDA_FLAGS "-Xcompiler=--gcc-toolchain=${GCC_HOME}" CACHE STRING "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.7.0" CACHE PATH "Path to CUDA")

set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")
