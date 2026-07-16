#ifndef BLT_FAKE_MPI_FORTRAN_COMPILE_OPTION
#error "Expected the fake Fortran MPI compile option"
#endif

program fortran_test
include 'mpif.h'

    if (blt_fake_mpi_fortran_header .ne. 1) stop 1

end program fortran_test
