program mpi_fortran_example
  implicit none

  integer :: ierr

  include 'mpif.h'

  call MPI_Init(ierr)
  call MPI_Finalize(ierr)
end program mpi_fortran_example

