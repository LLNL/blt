program non_mpi_fortran_example
  implicit none

  integer :: value
  value = 42

  if (value .ne. 42) then
    stop 1
  end if
end program non_mpi_fortran_example

