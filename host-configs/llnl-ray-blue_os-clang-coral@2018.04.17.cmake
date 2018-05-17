##############################################################
# Example host-config file for the blue_os ray cluster at LLNL
##############################################################
#
# This file provides CMake with paths / details for:
#  C,C++, MPI and Cuda
# 
###########################################################




###########################################################
# MPI Support
###########################################################
set(ENABLE_MPI ON CACHE BOOL "")
set(ENABLE_FIND_MPI OFF CACHE BOOL "")

set(MPI_HOME                 "/usr/tce/packages/spectrum-mpi/spectrum-mpi-2017.04.03-clang-coral-2018.04.17" CACHE PATH "")
set(MPI_C_COMPILER           "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER         "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER     "${MPI_HOME}/bin/mpif90" CACHE PATH "")

set(MPIEXEC              "mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE PATH "")

###########################################################
# clang-coral-2018.04.17 compilers
###########################################################
# c compiler
set(CMAKE_C_COMPILER "${MPI_C_COMPILER}" CACHE PATH "")
# cpp compiler
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}" CACHE PATH "")


###########################################################
# CUDA support
###########################################################
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-9.2.64" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")

set (CUDA_ARCH "sm_60" CACHE PATH "")
set (CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} -std=c++11 --expt-extended-lambda -G" CACHE STRING "" )
set (CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})

set (CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
set (CUDA_LINK_WITH_NVCC ON CACHE BOOL "")
# set the link flags manually since nvcc will link (and not have the wrappers knowledge)
# on ray - can figure out your equavilant flags by doing mpicc -vvvv
set (CMAKE_CUDA_LINK_FLAGS "-Xlinker -rpath -Xlinker /usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2017.04.03/lib -Xlinker -rpath -Xlinker /usr/tce/packages/clang/clang-coral-2017.10.13/ibm/lib:/usr/tce/packages/gcc/gcc-4.9.3/lib64 -L/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2017.04.03/lib/ -lmpi_ibm" CACHE STRING "")
