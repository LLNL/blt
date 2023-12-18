cmake_minimum_required(VERSION 3.14)

project(mpi-cpp-fortran-user LANGUAGES CXX Fortran)

# This call creates targets with relevant compile flags (i.e. for Fortran and
# cpp) for MPI.
find_package(mpi-cpp-fortran REQUIRED
             NO_DEFAULT_PATH
             PATHS ${base_install_dir}/lib/cmake/mpi-cpp-fortran)

add_executable(mpi-cpp-fortran-user mpi-cpp-fortran-user.F)
target_link_libraries(mpi-cpp-fortran-user PUBLIC mpi-cpp-fortran)

get_target_property(mpi_libraries mpi INTERFACE_LINK_LIBRARIES)

# Introducing FORTRAN in the downstream project will add an MPI FORTRAN library
# to the link libraries property of the target, which will contain `mpif`.
string(FIND "${mpi_libraries}" "mpif" mpi_fortran_library_idx)

if (${mpi_fortran_library_idx} EQUAL -1)
    message(FATAL_ERROR "ERROR: No MPI FORTRAN library was linked against.")
endif()

set_source_files_properties(
    mpi-cpp-fortran-user.F
    PROPERTIES
    Fortran_FORMAT FREE)
