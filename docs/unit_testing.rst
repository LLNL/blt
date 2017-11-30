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

.. _UnitTesting:

Unit Testing
============

BLT has a built-in copy of the 
`Google Test framework (gtest) <https://github.com/google/googletest>`_ for C and C++ unit tests and the 
`Fortran Unit Test Framework (FRUIT) <https://sourceforge.net/projects/fortranxunit/>`_ for Fortran unit tests. 


Each gtest or FRUIT file may contain multiple tests and is compiled into its own executable 
that can be run directly or as a ``make`` target. 

In this section, we give a brief overview of GTest and discuss how to add unit tests using the ``blt_add_test`` macro.

Google Test (C++/C Tests)
--------------------------

The contents of a typical Google Test file look like this:

.. code:: cpp

  #include "gtest/gtest.h"

  #include ...    // include headers needed to compile tests in file

  // ...

  TEST(<test_case_name>, <test_name_1>) 
  {
     // Test 1 code here...
     // ASSERT_EQ(...);
  }

  TEST(<test_case_name>, <test_name_2>) 
  {
     // Test 2 code here...
     // EXPECT_TRUE(...);
  }

  // Etc.

Each unit test is defined by the Google Test ``TEST()`` macro which accepts a 
*test case name* identifier, such as the name of the C++ class being tested, 
and a *test name*, which indicates the functionality being verified by the 
test.  Within a test, failure of logical assertions (macros prefixed by ``ASSERT_``)
will cause the test to fail immediately, while failures of expected values 
(macros prefixed by ``EXPECT_``) will cause the test to fail, but will 
continue running code within the test.

Note that the Google Test framework will generate a ``main()`` routine for 
each test file if it is not explicitly provided. However, sometimes it is 
necessary to provide a ``main()`` routine that contains operation to run 
before or after the unit tests in a file; e.g., initialization code or 
pre-/post-processing operations. A ``main()`` routine provided in a test 
file should be placed at the end of the file in which it resides.


Note that Google test is initialized before ``MPI_Init()`` is called. 

Other Google Test features, such as *fixtures* and *mock* objects (gmock) may be used as well. 

See the `Google Test Primer <https://github.com/google/googletest/blob/master/googletest/docs/Primer.md>`_ 
for a discussion of Google Test concepts, how to use them, and a listing of 
available assertion macros, etc.

Adding a BLT unit test 
----------------------

After writing a unit test, we add it to the project's build system 
by first generating an executable for the test using the ``blt_add_executable()`` macro.
We then register the test using the ``blt_add_test()`` macro.

.. admonition:: blt_add_test
   :class: hint

   This macro generates a named unit test from an existing executable
   and allows users to pass in command line arguments.


Returning to our running example (see  :ref:`AddTarget`), 
let's add a simple test for the ``calc_pi`` library, 
which has a single function with signature:

  .. code:: cpp

   double calc_pi(int num_intervals);

We add a simple unit test that invokes ``calc_pi()`` 
and compares the result to :math:`\pi`, within a given tolerance (``1e-6``). 
Here is the test code:

.. literalinclude:: tutorial/calc_pi/test_1.cpp
   :language: cpp
   :lines: 11-19

To add this test to the build system, we first generate a test executable:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt
   :language: cmake
   :lines: 45-50

Note that this test executable depends on two targets: ``calc_pi`` and ``gtest``.

We then register this executable as a test:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt
   :language: cmake
   :lines: 52-53

.. Hide these for now until we bring into an example
.. .. note::
..    The ``COMMAND`` parameter can be used to pass arguments into a test executable.
..    This feature is not exercised in this example.
..
.. .. note::
..    The ``NAME`` of the test can be different from the test executable,
..    which is passed in through the ``COMMAND`` parameter.
..    This feature is not exercised in this example.


Running tests and examples
--------------------------

To run the tests, type the following command in the build directory:

.. code:: bash

  $ make test 

This will run all tests through cmake's ``ctest`` tool 
and report a summary of passes and failures. 
Detailed output on individual tests is suppressed.

If a test fails, you can invoke its executable directly to see the detailed
output of which checks passed or failed. This is especially useful when 
you are modifying or adding code and need to understand how unit test details
are working, for example.

.. note:: 
    You can pass arguments to ctest via the ``TEST_ARGS`` parameter, e.g. ``make test TEST_ARGS="..."``
    Useful arguments include:
    
    -R
      Regular expression filtering of tests.  
      E.g. ``-R foo`` only runs tests whose names contain ``foo``
    -j
      Run tests in parallel, E.g. ``-j 16`` will run tests using 16 processors
    -VV
      (Very verbose) Dump test output to stdout


Configuring tests within BLT
----------------------------

Unit testing in BLT is controlled by the ``ENABLE_TESTS`` cmake option and is on by default. 

For additional configuration granularity, BLT provides configuration options 
for the individual built-in unit testing libraries.  The following additional options are available
when ``ENABLE_TESTS`` is on:

``ENABLE_GTEST``
  Option to enable gtest (default: ``ON``).
``ENABLE_GMOCK``
  Option to control gmock (default: ``OFF``).
  Since gmock requires gtest, gtest is also enabled whenever ``ENABLE_GMOCK`` is true, 
  regardless of the value of ``ENABLE_GTEST``. 
``ENABLE_FRUIT``
  Option to control FRUIT (Default ``ON``). It is only active when ``ENABLE_FORTRAN`` is enabled.



