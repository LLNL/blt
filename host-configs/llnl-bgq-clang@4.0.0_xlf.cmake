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
# Example host-config file for clang C/C++ compiler 
# paired with xlf Fortran compiler on LLNL's bgq machine
#------------------------------------------------------------------------------

set(CLANG_HOME "/collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable")
set(CMAKE_C_COMPILER "${CLANG_HOME}/llnl/bin/mpiclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${CLANG_HOME}/llnl/bin/mpiclang++11" CACHE PATH "")

# Set Fortran compiler
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(CMAKE_Fortran_COMPILER "/opt/ibmcmp/xlf/bg/14.1/bin/bgxlf2003" CACHE PATH "")


#------------------------------------------------------------------------------
# Extra options and flags
#------------------------------------------------------------------------------

set(ENABLE_DOCS    OFF CACHE BOOL "")
set(CMAKE_SKIP_RPATH TRUE CACHE BOOL "")

# Use clang's libc++ instead of libstdc++
set(BLT_CXX_FLAGS "-stdlib=libc++" CACHE STRING "")
set(gtest_defines "-DGTEST_HAS_CXXABI_H_=0" CACHE STRING "")

# Converts C-style comments to Fortran style in preprocessed files
set(BLT_FORTRAN_FLAGS "-WF,-C!" CACHE STRING "")


#------------------------------------------------------------------------------
# MPI Support
# Note: On BGQ, CMake uses the wrong linker flags when using FindMPI.
# Disable FindMPI to use LLNL wrapper scripts via CMake compiler variables.
#------------------------------------------------------------------------------

set(ENABLE_MPI      ON CACHE BOOL "")
set(ENABLE_FIND_MPI OFF CACHE BOOL "")

# Pass in an explicit path to help find mpif.h
set(MPI_Fortran_INCLUDE_PATH "/usr/local/tools/deg/drivers/V1R2M0/ppc64/comm/gcc/include" CACHE PATH "")

set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")

# Ensures that tests will be wrapped with srun to run on the backend nodes
set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC TRUE CACHE BOOL "Run tests on backend nodes")
