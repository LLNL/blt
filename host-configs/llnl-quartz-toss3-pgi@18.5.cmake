#------------------------------------------------------------------------------
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-725085
#
# All rights reserved.
#
# This file is part of BLT.
#
# For additional details, please also read BLT/LICENSE.
#------------------------------------------------------------------------------
# Example pgi@18.5 host-config for LLNL toss3 machines
#------------------------------------------------------------------------------

set(COMPILER_HOME "/usr/tce/packages/pgi/pgi-18.5")

# c compiler
set(CMAKE_C_COMPILER "${COMPILER_HOME}/bin/pgcc" CACHE PATH "")

# cpp compiler
set(CMAKE_CXX_COMPILER "${COMPILER_HOME}/bin/pgc++" CACHE PATH "")

# fortran support
set(ENABLE_FORTRAN ON CACHE BOOL "")

# fortran support
set(CMAKE_Fortran_COMPILER "${COMPILER_HOME}/bin/pgfortran" CACHE PATH "")

#------------------------------------------------------------------------------
# Extra options and flags
#------------------------------------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")


#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------

set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME             "/usr/tce/packages/mvapich2/mvapich2-2.2-pgi-18.5")

set(MPI_C_COMPILER       "${MPI_HOME}/bin/mpicc" CACHE PATH "")
set(MPI_CXX_COMPILER     "${MPI_HOME}/bin/mpicxx" CACHE PATH "")
set(MPI_Fortran_COMPILER "${MPI_HOME}/bin/mpif90" CACHE PATH "")
set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")
