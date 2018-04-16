###########################################################
# Example host-config file for the quartz cluster at LLNL
###########################################################
#
# This file provides CMake with paths / details for:
#  C,C++, & Fortran compilers + MPI
# 
###########################################################

###########################################################
# gcc@4.9.3 compilers
###########################################################

# c compiler
set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/gcc" CACHE PATH "")

# cpp compiler
set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/g++" CACHE PATH "")

# fortran support
set(ENABLE_FORTRAN ON CACHE BOOL "")

# fortran compiler
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/gfortran" CACHE PATH "")

###########################################################
# MPI Support
###########################################################
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_C_COMPILER       "/usr/tce/packages/mvapich2/mvapich2-2.2-gcc-4.9.3/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER     "/usr/tce/packages/mvapich2/mvapich2-2.2-gcc-4.9.3/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.2-gcc-4.9.3/bin/mpif90" CACHE PATH "")

set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")
