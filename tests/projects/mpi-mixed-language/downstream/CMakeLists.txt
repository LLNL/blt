foreach(_option MPI CUDA HIP OPENMP)
    unset(ENABLE_${_option} CACHE)
    unset(ENABLE_${_option})
endforeach()

cmake_minimum_required(VERSION 3.14)

project(mpi-mixed-language-user LANGUAGES C CXX Fortran)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../fake_mpi")

# Exercise the generator expressions without requiring GPU compilers in this
# project. Dedicated smoke tests compile the expressions with CUDA and HIP.
set(BLT_ENABLE_CUDA ON)
set(BLT_ENABLE_HIP ON)

find_package(mpi-mixed-language REQUIRED
             NO_DEFAULT_PATH
             PATHS ${base_install_dir}/lib/cmake/mpi-mixed-language)

set(_fake_mpi_root "${CMAKE_CURRENT_LIST_DIR}/../fake_mpi")
get_filename_component(_fake_mpi_root "${_fake_mpi_root}" ABSOLUTE)

function(_require_property_item target property expected)
    get_target_property(_items ${target} ${property})
    if(NOT _items)
        message(FATAL_ERROR "Expected ${target} to have ${property}")
    endif()

    if(NOT "${expected}" IN_LIST _items)
        message(FATAL_ERROR
            "Expected ${target} ${property} to contain '${expected}', "
            "but got '${_items}'")
    endif()
endfunction()

function(_reject_raw_property_item target property unexpected)
    get_target_property(_items ${target} ${property})
    if(NOT _items)
        return()
    endif()

    if("${unexpected}" IN_LIST _items)
        message(FATAL_ERROR
            "Expected ${target} ${property} not to contain unguarded item "
            "'${unexpected}', but got '${_items}'")
    endif()
endfunction()

set(_c_include "${_fake_mpi_root}/include/c")
set(_cxx_include "${_fake_mpi_root}/include/cxx")
set(_fortran_include "${_fake_mpi_root}/include/fortran")

_require_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<$<COMPILE_LANGUAGE:C>:${_c_include}>")
_require_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<$<COMPILE_LANGUAGE:CXX>:${_cxx_include}>")
_require_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<$<COMPILE_LANGUAGE:Fortran>:${_fortran_include}>")
_require_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<$<COMPILE_LANGUAGE:CUDA>:${_cxx_include}>")
_require_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<$<COMPILE_LANGUAGE:HIP>:${_cxx_include}>")

_reject_raw_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_c_include}")
_reject_raw_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_cxx_include}")
_reject_raw_property_item(mpi INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_fortran_include}")

_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:C>:-DBLT_FAKE_MPI_C_COMPILE_OPTION>")
_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:CXX>:-DBLT_FAKE_MPI_CXX_COMPILE_OPTION>")
_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:Fortran>:-DBLT_FAKE_MPI_FORTRAN_COMPILE_OPTION>")
_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-DBLT_FAKE_MPI_CXX_COMPILE_OPTION>")
_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:-DBLT_FAKE_MPI_CXX_COMPILE_OPTION>")
_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:CUDA>:$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler >-DBLT_FAKE_MPI_CXX_WRAPPED_OPTION>")
_require_property_item(mpi INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler >-DBLT_FAKE_MPI_CXX_WRAPPED_OPTION>")

_reject_raw_property_item(mpi INTERFACE_COMPILE_OPTIONS "-DBLT_FAKE_MPI_C_COMPILE_OPTION")
_reject_raw_property_item(mpi INTERFACE_COMPILE_OPTIONS "-DBLT_FAKE_MPI_CXX_COMPILE_OPTION")
_reject_raw_property_item(mpi INTERFACE_COMPILE_OPTIONS "-DBLT_FAKE_MPI_FORTRAN_COMPILE_OPTION")

_require_property_item(mpi INTERFACE_COMPILE_DEFINITIONS
    "$<$<COMPILE_LANGUAGE:C>:BLT_FAKE_MPI_C_COMPILE_DEFINITION>")
_require_property_item(mpi INTERFACE_COMPILE_DEFINITIONS
    "$<$<COMPILE_LANGUAGE:CXX>:BLT_FAKE_MPI_CXX_COMPILE_DEFINITION>")
_require_property_item(mpi INTERFACE_COMPILE_DEFINITIONS
    "$<$<COMPILE_LANGUAGE:Fortran>:BLT_FAKE_MPI_FORTRAN_COMPILE_DEFINITION>")
_require_property_item(mpi INTERFACE_COMPILE_DEFINITIONS
    "$<$<COMPILE_LANGUAGE:CUDA>:BLT_FAKE_MPI_CXX_COMPILE_DEFINITION>")
_require_property_item(mpi INTERFACE_COMPILE_DEFINITIONS
    "$<$<COMPILE_LANGUAGE:HIP>:BLT_FAKE_MPI_CXX_COMPILE_DEFINITION>")

add_executable(mpi-mixed-language-user
    mpi-mixed-language-user.c
    mpi-mixed-language-user.cpp
    mpi-mixed-language-user.F)
target_link_libraries(mpi-mixed-language-user PUBLIC mpi-mixed-language mpi)

set_source_files_properties(
    mpi-mixed-language-user.F
    PROPERTIES
    Fortran_FORMAT FREE)
