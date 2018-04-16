!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!
! Copyright (c) 2017, Lawrence Livermore National Security, LLC.
!
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-725085
!
! All rights reserved.
!
! This file is part of BLT.
!
! For additional details, please also read BLT/LICENSE.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! * Redistributions of source code must retain the above copyright notice,
!   this list of conditions and the disclaimer below.
!
! * Redistributions in binary form must reproduce the above copyright notice,
!   this list of conditions and the disclaimer (as noted below) in the
!   documentation and/or other materials provided with the distribution.
!
! * Neither the name of the LLNS/LLNL nor the names of its contributors may
!   be used to endorse or promote products derived from this software without
!   specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
! ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
! LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
! DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
! OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
! HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
! STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
! IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
! 
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!


!------------------------------------------------------------------------------
!
! blt_fruit_smoke.f90
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

