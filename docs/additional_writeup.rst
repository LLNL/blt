.. ###############################################################################
.. # Copyright (c) 2017, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-725085
.. #
.. # All rights reserved.
.. #
.. # This file is part of BLT.
.. #
.. # For additional details, please also read BLT/LICENSE.
.. #
.. # Redistribution and use in source and binary forms, with or without
.. # modification, are permitted provided that the following conditions are met:
.. #
.. # * Redistributions of source code must retain the above copyright notice,
.. #   this list of conditions and the disclaimer below.
.. #
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. #
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. #
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
.. # POSSIBILITY OF SUCH DAMAGE.
.. #
.. ###############################################################################

.. # This file contains some additional writeup that we will not discuss in the tutorial.


FRUIT (Fortran Tests)
--------------------------

Fortran unit tests using the FRUIT framework are similar in structure to 
the Google Test tests for C and C++ described above.

The contents of a typical FRUIT test file look like this::

  module <test_case_name>
    use iso_c_binding
    use fruit
    use <your_code_module_name>
    implicit none

  contains

  subroutine test_name_1
  !  Test 1 code here...
  !  call assert_equals(...)
  end subroutine test_name_1

  subroutine test_name_2
  !  Test 2 code here...
  !  call assert_true(...)
  end subroutine test_name_2

  ! Etc.

The tests in a FRUIT test file are placed in a Fortran *module* named for
the *test case name*, such as the name of the C++ class whose Fortran interface
is being tested. Each unit test is in its own Fortran subroutine named
for the *test name*, which indicates the functionality being verified by the
unit test. Within each unit test, logical assertions are defined using
FRUIT methods. Failure of expected values will cause the test
to fail, but other tests will continue to run.

Note that each FRUIT test file defines an executable Fortran program. The
program is defined at the end of the test file and is organized as follows::

    program fortran_test
      use fruit
      use <your_component_unit_name>
      implicit none
      logical ok
      
      ! initialize fruit
      call init_fruit
      
      ! run tests
      call test_name_1
      call test_name_2
      
      ! compile summary and finalize fruit
      call fruit_summary
      call fruit_finalize
      
      call is_all_successful(ok)
      if (.not. ok) then
        call exit(1)
      endif
    end program fortran_test


Please refer to the `FRUIT documentation <https://sourceforge.net/projects/fortranxunit/>`_ for more information.



