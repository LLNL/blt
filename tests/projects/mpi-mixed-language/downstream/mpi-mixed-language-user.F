#ifndef BLT_FAKE_MPI_FORTRAN_COMPILE_OPTION
#error "Expected the fake Fortran MPI compile option"
#endif

#ifndef BLT_FAKE_MPI_FORTRAN_COMPILE_DEFINITION
#error "Expected the fake Fortran MPI compile definition"
#endif

#if defined(BLT_FAKE_MPI_C_COMPILE_OPTION) || \
    defined(BLT_FAKE_MPI_CXX_COMPILE_OPTION) || \
    defined(BLT_FAKE_MPI_C_COMPILE_DEFINITION) || \
    defined(BLT_FAKE_MPI_CXX_COMPILE_DEFINITION)
#error "Unexpected C or CXX MPI compile property in Fortran compilation"
#endif

program fortran_test
include 'mpif.h'

    if (blt_fake_mpi_fortran_header .ne. 1) stop 1

end program fortran_test
