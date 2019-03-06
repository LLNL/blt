# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# MPI
################################

if (ENABLE_FIND_MPI)
    find_package(MPI REQUIRED)

    message(STATUS "MPI C Compile Flags:  ${MPI_C_COMPILE_FLAGS}")
    message(STATUS "MPI C Include Path:   ${MPI_C_INCLUDE_PATH}")
    message(STATUS "MPI C Link Flags:     ${MPI_C_LINK_FLAGS}")
    message(STATUS "MPI C Libraries:      ${MPI_C_LIBRARIES}")

    message(STATUS "MPI CXX Compile Flags: ${MPI_CXX_COMPILE_FLAGS}")
    message(STATUS "MPI CXX Include Path:  ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI CXX Link Flags:    ${MPI_CXX_LINK_FLAGS}")
    message(STATUS "MPI CXX Libraries:     ${MPI_CXX_LIBRARIES}")
endif()

message(STATUS "MPI Executable:       ${MPIEXEC}")
message(STATUS "MPI Num Proc Flag:    ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "MPI Command Append:   ${BLT_MPI_COMMAND_APPEND}")

if (ENABLE_FORTRAN)
    # Determine if we should use fortran mpif.h header or fortran mpi module
    find_path(mpif_path
        NAMES "mpif.h"
        PATHS ${MPI_Fortran_INCLUDE_PATH}
        NO_DEFAULT_PATH
        )
    
    if(mpif_path)
        set(MPI_Fortran_USE_MPIF ON CACHE PATH "")
        message(STATUS "Using MPI Fortran header: mpif.h")
    else()
        set(MPI_Fortran_USE_MPIF OFF CACHE PATH "")
        message(STATUS "Using MPI Fortran module: mpi.mod")
    endif()
endif()

# register MPI with blt
blt_register_library(NAME mpi
                     INCLUDES ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH} ${MPI_Fortran_INCLUDE_PATH}
                     TREAT_INCLUDES_AS_SYSTEM ON
                     LIBRARIES ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${MPI_Fortran_LIBRARIES}
                     COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}"
                     LINK_FLAGS    "${MPI_C_COMPILE_FLAGS} ${MPI_Fortran_LINK_FLAGS}")


