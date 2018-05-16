###########################################################
# Example host-config file for the surface cluster at LLNL
###########################################################
#
# This file provides CMake with paths / details for:
#  C,C++, & Fortran compilers + MPI & CUDA
###########################################################

###########################################################
# gcc@4.9.3 compilers
###########################################################

# c compiler
set(CMAKE_C_COMPILER "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")

# cpp compiler
set(CMAKE_CXX_COMPILER "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

# fortran support
set(ENABLE_FORTRAN ON CACHE BOOL "")

# fortran compiler
set(CMAKE_Fortran_COMPILER "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")

###########################################################
# MPI Support
###########################################################
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_C_COMPILER "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/local/tools/mvapich2-gnu-2.0/bin/mpif90" CACHE PATH "")

###########################################################
# CUDA support
###########################################################
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/opt/cudatoolkit-8.0" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "/opt/cudatoolkit-8.0/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")
set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")

