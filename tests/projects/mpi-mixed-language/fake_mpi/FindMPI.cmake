# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(_fake_mpi_root "${CMAKE_CURRENT_LIST_DIR}")

set(MPI_FOUND TRUE)
set(MPI_C_FOUND TRUE)
set(MPI_CXX_FOUND TRUE)
set(MPI_Fortran_FOUND TRUE)
set(MPI_Fortran_HAVE_F77_HEADER TRUE)

set(MPIEXEC_EXECUTABLE "${CMAKE_COMMAND}")
set(MPIEXEC "${CMAKE_COMMAND}")
set(MPIEXEC_NUMPROC_FLAG "-E")

set(MPI_C_INCLUDE_DIRS "${_fake_mpi_root}/include/c")
set(MPI_CXX_INCLUDE_DIRS "${_fake_mpi_root}/include/cxx")
set(MPI_Fortran_INCLUDE_DIRS "${_fake_mpi_root}/include/fortran")

set(MPI_C_INCLUDE_PATH "${MPI_C_INCLUDE_DIRS}")
set(MPI_CXX_INCLUDE_PATH "${MPI_CXX_INCLUDE_DIRS}")
set(MPI_Fortran_INCLUDE_PATH "${MPI_Fortran_INCLUDE_DIRS}")

set(MPI_C_COMPILE_OPTIONS "-DBLT_FAKE_MPI_C_COMPILE_OPTION")
set(MPI_CXX_COMPILE_OPTIONS
    "-DBLT_FAKE_MPI_CXX_COMPILE_OPTION"
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler >-DBLT_FAKE_MPI_CXX_WRAPPED_OPTION")
set(MPI_Fortran_COMPILE_OPTIONS "-DBLT_FAKE_MPI_FORTRAN_COMPILE_OPTION")

set(MPI_C_COMPILE_DEFINITIONS "BLT_FAKE_MPI_C_COMPILE_DEFINITION")
set(MPI_CXX_COMPILE_DEFINITIONS "BLT_FAKE_MPI_CXX_COMPILE_DEFINITION")
set(MPI_Fortran_COMPILE_DEFINITIONS "BLT_FAKE_MPI_FORTRAN_COMPILE_DEFINITION")

set(MPI_C_COMPILE_FLAGS "${MPI_C_COMPILE_OPTIONS}")
set(MPI_CXX_COMPILE_FLAGS "${MPI_CXX_COMPILE_OPTIONS}")
set(MPI_Fortran_COMPILE_FLAGS "${MPI_Fortran_COMPILE_OPTIONS}")

set(MPI_C_LINK_FLAGS "")
set(MPI_CXX_LINK_FLAGS "")
set(MPI_Fortran_LINK_FLAGS "")
set(MPI_C_LIBRARIES "")
set(MPI_CXX_LIBRARIES "")
set(MPI_Fortran_LIBRARIES "")

foreach(_fake_mpi_language IN LISTS MPI_FIND_COMPONENTS)
    if(NOT TARGET MPI::MPI_${_fake_mpi_language})
        add_library(MPI::MPI_${_fake_mpi_language} INTERFACE IMPORTED)
        set_target_properties(MPI::MPI_${_fake_mpi_language} PROPERTIES
            INTERFACE_COMPILE_OPTIONS
                "${MPI_${_fake_mpi_language}_COMPILE_OPTIONS}"
            INTERFACE_COMPILE_DEFINITIONS
                "${MPI_${_fake_mpi_language}_COMPILE_DEFINITIONS}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${MPI_${_fake_mpi_language}_INCLUDE_DIRS}")
    endif()
endforeach()

unset(_fake_mpi_language)
unset(_fake_mpi_root)
