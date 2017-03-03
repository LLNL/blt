!------------------------------------------------------------------------------
! Copyright (c) 2015, Lawrence Livermore National Security, LLC.
! Produced at the Lawrence Livermore National Laboratory.
!
! All rights reserved.
!
! This source code cannot be distributed without permission and
! further review from Lawrence Livermore National Laboratory.
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!
! blt_fruit_smoke.f
!
!------------------------------------------------------------------------------
module fruit_smoke
  use iso_c_binding
  use fruit
  implicit none

contains
!------------------------------------------------------------------------------

  subroutine simple_test
        call assert_equals (42, 42)
  end subroutine simple_test


!----------------------------------------------------------------------
end module fruit_smoke
!----------------------------------------------------------------------

program fortran_test
  use fruit
  use fruit_smoke
  implicit none
  logical ok

  call init_fruit
!----------
! Our tests
  call simple_test
!----------

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif
  
end program fortran_test

