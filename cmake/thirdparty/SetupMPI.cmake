# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# MPI
################################

# Handle CMake changing MPIEXEC to MPIEXEC_EXECUTABLE
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
    if (NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
        set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE PATH "" FORCE)
    endif()
else()
    if (MPIEXEC_EXECUTABLE AND NOT MPIEXEC)
        set(MPIEXEC ${MPIEXEC_EXECUTABLE} CACHE PATH "" FORCE)
    endif()
endif()

set(_mpi_compile_flags)
set(_mpi_includes)
set(_mpi_libraries)
set(_mpi_link_flags)

if (ENABLE_FIND_MPI)
    find_package(MPI REQUIRED)

    #-------------------
    # Merge found MPI info and remove duplication
    #-------------------
    # Compile flags
    list(APPEND _mpi_compile_flags ${MPI_C_COMPILE_FLAGS})
    if (NOT "${MPI_C_COMPILE_FLAGS}" STREQUAL "${MPI_CXX_COMPILE_FLAGS}")
        list(APPEND _mpi_compile_flags ${MPI_CXX_LINK_FLAGS})
    endif()
    if (ENABLE_FORTRAN)
        if (NOT "${MPI_C_COMPILE_FLAGS}" STREQUAL "${MPI_Fortran_COMPILE_FLAGS}")
            list(APPEND _mpi_compile_flags ${MPI_CXX__COMPILE_FLAGS})
        endif()
    endif()

    # Include paths
    list(APPEND _mpi_includes ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    if (ENABLE_FORTRAN)
        list(APPEND _mpi_includes ${MPI_Fortran_INCLUDE_PATH})
    endif()
    list(REMOVE_DUPLICATES _mpi_includes)

    # Link flags
    set(_mpi_link_flags ${MPI_C_LINK_FLAGS})
    if (NOT "${MPI_C_LINK_FLAGS}" STREQUAL "${MPI_CXX_LINK_FLAGS}")
        list(APPEND _mpi_link_flags ${MPI_CXX_LINK_FLAGS})
    endif()
    if (ENABLE_FORTRAN)
        if (NOT "${MPI_C_LINK_FLAGS}" STREQUAL "${MPI_Fortran_LINK_FLAGS}")
            list(APPEND _mpi_link_flags ${MPI_CXX_LINK_FLAGS})
        endif()
    endif()

    # Libraries
    set(_mpi_libraries ${MPI_C_LIBRARIES}
                       ${MPI_CXX_LIBRARIES})
    if (ENABLE_FORTRAN)
        list(APPEND _mpi_libraries ${MPI_Fortran_LIBRARIES})
    endif()
    list(REMOVE_DUPLICATES _mpi_libraries)
endif()

message(STATUS "BLT MPI Compile Flags:  ${_mpi_compile_flags}")
message(STATUS "BLT MPI Include Paths:   ${_mpi_includes}")
message(STATUS "BLT MPI Libraries:      ${_mpi_libraries}")
message(STATUS "BLT MPI Link Flags:     ${_mpi_link_flags}")

if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
    message(STATUS "MPI Executable:       ${MPIEXEC_EXECUTABLE}")
else()
    message(STATUS "MPI Executable:       ${MPIEXEC}")
endif()
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

# Create the registered library
blt_register_library(NAME          mpi
                     INCLUDES      ${_mpi_includes}
                     TREAT_INCLUDES_AS_SYSTEM ON
                     LIBRARIES     ${_mpi_libraries}
                     COMPILE_FLAGS ${_mpi_compile_flags}
                     LINK_FLAGS    ${_mpi_link_flags} )


