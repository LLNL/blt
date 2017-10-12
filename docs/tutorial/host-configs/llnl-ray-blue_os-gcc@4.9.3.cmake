##############################################################
# Example host-config file for the blue_os ray cluster at LLNL
##############################################################
#
# This file provides CMake with paths / details for:
#  C,C++, MPI and Cuda
# 
###########################################################

###########################################################
# gcc@4.9.3 compilers
###########################################################

# c compiler used by spack
set(CMAKE_C_COMPILER "/usr/tcetmp/packages/gcc/gcc-4.9.3/bin/gcc" CACHE PATH "")

# cpp compiler used by spack
set(CMAKE_CXX_COMPILER "/usr/tcetmp/packages/gcc/gcc-4.9.3/bin/g++" CACHE PATH "")


###########################################################
# MPI Support
###########################################################
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME                 "/usr/tcetmp/packages/spectrum-mpi/spectrum-mpi-2017.04.03-gcc-4.9.3" CACHE PATH "")
set(MPI_C_COMPILER           "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER         "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER     "${MPI_HOME}/bin/mpif90" CACHE PATH "")

set(MPIEXEC              "mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE PATH "")


###########################################################
# CUDA support
###########################################################
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0" CACHE PATH "")
set(CUDA_BIN_DIR "/usr/local/cuda-8.0/bin" CACHE PATH "")

set (CUDA_ARCH "sm_60" CACHE PATH "")
set (NVCC_FLAGS -restrict; -arch; ${CUDA_ARCH}; -std c++11; --expt-extended-lambda; )

#  options for findCUDA
set (CUDA_NVCC_FLAGS ${NVCC_FLAGS} CACHE LIST "" )
set (CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
set (CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
