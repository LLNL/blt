# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level LICENSE file for details
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

# Users must provide MPI_<lang>_COMPILER variables for each enabled language unless BLT_ALLOW_MISSING_MPI_WRAPPER
# Check for each variable and print a warning/error message if appropriate
get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)

if (BLT_ENABLE_FIND_MPI)
    message(STATUS "FindMPI Enabled  (ENABLE_FIND_MPI == ON)")

    set(_blt_missing_mpi_wrappers)
    foreach(_lang "C" "CXX" "Fortran")
        if (${_lang} IN_LIST _enabled_languages AND NOT MPI_${_lang}_COMPILER)
            list(APPEND _blt_missing_mpi_wrappers "MPI_${_lang}_COMPILER")
        endif()
    endforeach()

    if (_blt_missing_mpi_wrappers)
        if (BLT_ALLOW_MISSING_MPI_WRAPPER)
            message(STATUS "BLT_ALLOW_MISSING_MPI_WRAPPER == ON. Proceeding without: ${_blt_missing_mpi_wrappers}")
        else()
            message(FATAL_ERROR
                    "MPI support is enabled, but missing MPI compiler wrapper(s): ${_blt_missing_mpi_wrappers}\n"
                    "Provide MPI_<lang>_COMPILER for each enabled language or set BLT_ALLOW_MISSING_MPI_WRAPPER=ON to bypass this check.")
        endif()
    endif()
    unset(_blt_missing_mpi_wrappers)
else()
    message(STATUS "FindMPI Disabled (ENABLE_FIND_MPI == OFF) ")
endif()

set(_blt_enable_mpi_c FALSE)
if("C" IN_LIST _enabled_languages AND MPI_C_COMPILER)
    set(_blt_enable_mpi_c TRUE)
endif()

set(_blt_enable_mpi_cxx FALSE)
if("CXX" IN_LIST _enabled_languages AND MPI_CXX_COMPILER)
    set(_blt_enable_mpi_cxx TRUE)
endif()

set(_blt_enable_mpi_fortran FALSE)
if ("Fortran" IN_LIST _enabled_languages AND MPI_Fortran_COMPILER)
    set(_blt_enable_mpi_fortran TRUE)
endif()

if (BLT_ENABLE_FIND_MPI)
    if (BLT_ALLOW_MISSING_MPI_WRAPPER)
        set(_blt_mpi_components)
        blt_list_append(TO _blt_mpi_components ELEMENTS C       IF _blt_enable_mpi_c)
        blt_list_append(TO _blt_mpi_components ELEMENTS CXX     IF _blt_enable_mpi_cxx)
        blt_list_append(TO _blt_mpi_components ELEMENTS Fortran IF _blt_enable_mpi_fortran)
        find_package(MPI REQUIRED COMPONENTS ${_blt_mpi_components})
    else()
        find_package(MPI REQUIRED)
    endif()

    #-------------------
    # Merge found MPI info and remove duplication
    #-------------------

    #-------------------
    # Compile flags
    #-------------------
    if(_blt_enable_mpi_c)
        set(_c_flag ${MPI_C_${_mpi_compile_flags_suffix}})
        if (_c_flag AND BLT_ENABLE_CUDA)
            list(APPEND _mpi_compile_flags
                        $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_c_flag}>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_c_flag}>)
        else()
            list(APPEND _mpi_compile_flags ${_c_flag})
        endif()
    endif()

    if(_blt_enable_mpi_cxx)
        set(_cxx_flag ${MPI_CXX_${_mpi_compile_flags_suffix}})
        if (_cxx_flag AND NOT "${_c_flag}" STREQUAL "${_cxx_flag}")
            if (BLT_ENABLE_CUDA)
                list(APPEND _mpi_compile_flags
                $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_cxx_flag}>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_cxx_flag}>)
            else()
                list(APPEND _mpi_compile_flags ${_cxx_flag})
            endif()
        endif()
    endif()

    if (_blt_enable_mpi_fortran)
        set(_f_flag ${MPI_Fortran_${_mpi_compile_flags_suffix}})
        if (_f_flag AND NOT "${_c_flag}" STREQUAL "${_f_flag}")
            list(APPEND _mpi_compile_flags ${_f_flag})
        endif()
    endif()
    unset(_c_flag)
    unset(_cxx_flag)
    unset(_f_flag)

    #-------------------
    # Include paths
    #-------------------
    set(_mpi_includes)
    if(_blt_enable_mpi_c)
        list(APPEND _mpi_includes ${MPI_C_${_mpi_includes_suffix}})
    endif()
    if(_blt_enable_mpi_cxx)
        list(APPEND _mpi_includes ${MPI_CXX_${_mpi_includes_suffix}})
    endif()
    if(_blt_enable_mpi_fortran)
        list(APPEND _mpi_includes ${MPI_Fortran_${_mpi_includes_suffix}})
    endif()
    blt_list_remove_duplicates(TO _mpi_includes)

    #-------------------
    # Link flags
    #-------------------
    if(_blt_enable_mpi_c)
        set(_mpi_link_flags ${MPI_C_LINK_FLAGS})
    endif()

    if(_blt_enable_mpi_cxx AND NOT "${_mpi_link_flags}" STREQUAL "${MPI_CXX_LINK_FLAGS}")
        list(APPEND _mpi_link_flags ${MPI_CXX_LINK_FLAGS})
    endif()

    if (_blt_enable_mpi_fortran AND NOT "${_mpi_link_flags}" STREQUAL "${MPI_Fortran_LINK_FLAGS}")
        list(APPEND _mpi_link_flags ${MPI_Fortran_LINK_FLAGS})
    endif()

    # Selectively remove set of known locations of spaces
    string(REPLACE " -Wl"      ";-Wl"       _mpi_link_flags "${_mpi_link_flags}")
    string(REPLACE " -L"       ";-L"        _mpi_link_flags "${_mpi_link_flags}")
    string(REPLACE " -pthread" ";-pthread"  _mpi_link_flags "${_mpi_link_flags}")

    # Fixes for linking with NVCC
    if (CUDA_LINK_WITH_NVCC)
        # Convert rpath flag if linking with CUDA
        string(REPLACE "-Wl,-rpath," "-Xlinker -rpath -Xlinker "
                       _mpi_link_flags "${_mpi_link_flags}")
        # -pthread just doesn't work with nvcc
        string(REPLACE "-pthread" " "
                       _mpi_link_flags "${_mpi_link_flags}")
    endif()

    #-------------------
    # Libraries
    #-------------------
    if(_blt_enable_mpi_c)
        list(APPEND _mpi_libraries ${MPI_C_LIBRARIES})
    endif()
    if(_blt_enable_mpi_cxx)
        list(APPEND _mpi_libraries ${MPI_CXX_LIBRARIES})
    endif()
    if(_blt_enable_mpi_fortran)
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

#
# We use `LINK_OPTIONS` for CMake 3.13 and beyond.
# To make sure that de-duping doesn't undermine our MPI flags
# use a string with SHELL: prefix
#
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
    string(REPLACE ";" " " _mpi_link_flags "${_mpi_link_flags}")
    set(_mpi_link_flags "SHELL:${_mpi_link_flags}")
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

if (_blt_enable_mpi_fortran)
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

# Don't create the mpi target if it already exists.
blt_import_library(NAME          mpi
                   INCLUDES      ${_mpi_includes}
                   TREAT_INCLUDES_AS_SYSTEM ON
                   LIBRARIES     ${_mpi_libraries}
                   COMPILE_FLAGS ${_mpi_compile_flags}
                   LINK_FLAGS    ${_mpi_link_flags}
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY})

add_library(blt::mpi ALIAS mpi)
