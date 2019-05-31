# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# MPI
################################

# CMake changed some of the output variables that we use from Find(MPI)
# in 3.10+.  This toggles the variables based on the CMake version
# the user is running.
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
    if (NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
        set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE PATH "" FORCE)
    endif()

    set(_mpi_includes_suffix "INCLUDE_DIRS")
    set(_mpi_compile_flags_suffix "COMPILE_OPTIONS")
else()
    if (MPIEXEC_EXECUTABLE AND NOT MPIEXEC)
        set(MPIEXEC ${MPIEXEC_EXECUTABLE} CACHE PATH "" FORCE)
    endif()

    set(_mpi_includes_suffix "INCLUDE_PATH")
    set(_mpi_compile_flags_suffix "COMPILE_FLAGS")
endif()

set(_mpi_compile_flags )
set(_mpi_includes )
set(_mpi_libraries )
set(_mpi_link_flags )

message(STATUS "Enable FindMPI:  ${ENABLE_FIND_MPI}")

if (ENABLE_FIND_MPI)
    find_package(MPI REQUIRED)

    #-------------------
    # Merge found MPI info and remove duplication
    #-------------------
    # Compile flags
    list(APPEND _mpi_compile_flags ${MPI_C_${_mpi_compile_flags_suffix}})
    if (NOT "${MPI_C_${_mpi_compile_flags_suffix}}" STREQUAL
             "${MPI_CXX_${_mpi_compile_flags_suffix}}")
        list(APPEND _mpi_compile_flags ${MPI_CXX_${_mpi_compile_flags_suffix}})
    endif()
    if (ENABLE_FORTRAN)
        if (NOT "${MPI_C_${_mpi_compile_flags_suffix}}" STREQUAL
                "${MPI_Fortran_${_mpi_compile_flags_suffix}}")
            list(APPEND _mpi_compile_flags ${MPI_Fortran_${_mpi_compile_flags_suffix}})
        endif()
    endif()

    # Include paths
    list(APPEND _mpi_includes ${MPI_C_${_mpi_includes_suffix}}
                              ${MPI_CXX_${_mpi_includes_suffix}})
    if (ENABLE_FORTRAN)
        list(APPEND _mpi_includes ${MPI_Fortran_${_mpi_includes_suffix}})
    endif()
    blt_list_remove_duplicates(TO _mpi_includes)

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
    set(_mpi_libraries ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    if (ENABLE_FORTRAN)
        list(APPEND _mpi_libraries ${MPI_Fortran_LIBRARIES})
    endif()
    blt_list_remove_duplicates(TO _mpi_libraries)
endif()

# Allow users to override CMake's FindMPI
if (BLT_MPI_COMPILE_FLAGS)
    set(_mpi_compile_flags ${BLT_MPI_COMPILE_FLAGS})
endif()
if (BLT_MPI_INCLUDES)
    set(_mpi_includes ${BLT_MPI_INCLUDES})
endif()
if (BLT_MPI_LIBRARIES)
    set(_mpi_libraries ${BLT_MPI_LIBRARIES})
endif()
if (BLT_MPI_LINK_FLAGS)
    set(_mpi_link_flags ${BLT_MPI_LINK_FLAGS})
endif()


# Output all MPI information
message(STATUS "BLT MPI Compile Flags:  ${_mpi_compile_flags}")
message(STATUS "BLT MPI Include Paths:  ${_mpi_includes}")
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
        PATHS ${_mpi_includes}
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
# When using blt with RAJA, this fix is needed to compile RAJA with MPI and CUDA enabled smoothly.
if(RAJA_EXISTS AND ENABLE_FIND_MPI AND ENABLE_CUDA)
blt_register_library(NAME mpi
                     INCLUDES ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH} ${MPI_Fortran_INCLUDE_PATH}
                     TREAT_INCLUDES_AS_SYSTEM ON
                     LIBRARIES ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${MPI_Fortran_LIBRARIES}
                     COMPILE_FLAGS "-Xcompiler=${MPI_C_COMPILE_FLAGS}"
                     LINK_FLAGS    "${MPI_C_COMPILE_FLAGS} ${MPI_Fortran_LINK_FLAGS}")
else()
blt_register_library(NAME mpi
                     INCLUDES ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH} ${MPI_Fortran_INCLUDE_PATH}
                     TREAT_INCLUDES_AS_SYSTEM ON
                     LIBRARIES ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${MPI_Fortran_LIBRARIES}
                     COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}"
                     LINK_FLAGS    "${MPI_C_COMPILE_FLAGS} ${MPI_Fortran_LINK_FLAGS}")

endif()
