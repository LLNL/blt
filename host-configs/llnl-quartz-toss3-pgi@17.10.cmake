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
# Example pgi@17.10 host-config for LLNL toss3 machines
#------------------------------------------------------------------------------

set(PGI_HOME "/usr/tce/packages/pgi/pgi-17.10")

# c compiler
set(CMAKE_C_COMPILER "${PGI_HOME}/bin/pgcc" CACHE PATH "")

# cpp compiler
set(CMAKE_CXX_COMPILER "${PGI_HOME}/bin/pgc++" CACHE PATH "")

# fortran support
set(ENABLE_FORTRAN ON CACHE BOOL "")

# fortran support
set(CMAKE_Fortran_COMPILER "${PGI_HOME}/bin/pgfortran" CACHE PATH "")

