# Clear variables set in the host-config so we can test that they are inherited
# from the upstream project.
foreach(_option MPI CUDA HIP OPENMP)
    unset(ENABLE_${_option} CACHE)
    unset(ENABLE_${_option})
endforeach()

unset(BLT_ALLOW_MISSING_MPI_WRAPPER CACHE)
unset(BLT_ALLOW_MISSING_MPI_WRAPPER)

cmake_minimum_required(VERSION 3.14)

project(mpi-cpp-fortran-missing-wrapper-user LANGUAGES CXX Fortran)

if (NOT MPI_Fortran_COMPILER)
    message(FATAL_ERROR "ERROR: Missing MPI Fortran compiler wrapper (mpif90/mpifort).")
endif()

find_package(mpi-cpp-fortran-missing-wrapper REQUIRED
             NO_DEFAULT_PATH
             PATHS ${base_install_dir}/lib/cmake/mpi-cpp-fortran-missing-wrapper)

if (NOT BLT_ALLOW_MISSING_MPI_WRAPPER)
    message(FATAL_ERROR "ERROR: BLT_ALLOW_MISSING_MPI_WRAPPER was not inherited from the upstream project.")
endif()

add_executable(mpi-cpp-fortran-missing-wrapper-user mpi-cpp-fortran-missing-wrapper-user.cpp)
target_link_libraries(mpi-cpp-fortran-missing-wrapper-user PUBLIC mpi-cpp-fortran-missing-wrapper)

if (NOT DEFINED MPI_Fortran_USE_MPIF)
    message(FATAL_ERROR "ERROR: Expected MPI_Fortran_USE_MPIF to be defined when MPI_Fortran_COMPILER is provided.")
endif()

if (MPI_Fortran_USE_MPIF)
    set(_mpi_fortran_example_src mpi_fortran_example_mpif.f90)
else()
    set(_mpi_fortran_example_src mpi_fortran_example_module.f90)
endif()

add_executable(mpi-cpp-fortran-missing-wrapper-user-fortran-mpi ${_mpi_fortran_example_src})
target_link_libraries(mpi-cpp-fortran-missing-wrapper-user-fortran-mpi PRIVATE mpi)

