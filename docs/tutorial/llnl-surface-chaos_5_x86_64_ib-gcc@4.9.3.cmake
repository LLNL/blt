###########################################################
# Example host-config file for the surface cluster at LLNL
###########################################################
#
# This file provides CMake with paths to:
#  C,C++, and Fortran compilers
#  MPI, and CUDA.
###########################################################

###########################################################
# gcc@4.9.3 compilers
###########################################################

# c compiler
set("CMAKE_C_COMPILER" "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")

# cpp compiler
set("CMAKE_CXX_COMPILER" "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

# fortran support
set("ENABLE_FORTRAN" "ON" CACHE PATH "")

# fortran compiler
set("CMAKE_Fortran_COMPILER" "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")

###########################################################
# MPI Support
###########################################################
set("ENABLE_MPI" "ON" CACHE PATH "")

set("MPI_C_COMPILER" "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set("MPI_CXX_COMPILER" "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set("MPI_Fortran_COMPILER" "/usr/local/tools/mvapich2-gnu-2.0/bin/mpif90" CACHE PATH "")

###########################################################
# CUDA support
###########################################################
set("ENABLE_CUDA" "ON" CACHE PATH "")

set("CUDA_BIN_DIR" "/opt/cudatoolkit-8.0/bin" CACHE PATH "")
