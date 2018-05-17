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

.. _AddTarget:

Creating Libraries and Executables
==================================

In the previous section, we learned the basics about how to create a CMake project with BLT, 
how to configure the project and how to build and test BLT's built-in third party libraries.  

We now move on to creating libraries and executables
using two of BLT's core macros: ``blt_add_library`` and ``blt_add_executable``.

We begin with a simple executable that calculates pi by numerical integration (``example_1``).  
We will then extract that code into a library, which we link into a new executable (``example_2``).


Example 1: Basic executable
---------------------------

This example is as basic as it gets. After setting up a BLT CMake project, 
like ``blank_project`` in the previous section, we can start using BLT's macros.  
Creating an executable is as simple as calling the following macro:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt
   :language: cmake
   :lines: 28-29

This tells CMake to create an executable named ``example_1`` with one source file (``example_1.cpp``).

You can create this project yourself or you can run the already provided ``tutorial/calc_pi`` project.
For ease of use, we have combined many examples into this one CMake project.  After running
the following commands, you will create the executable ``<build dir>/bin/example_1``:

.. code:: bash

    cd <BLT repository/docs/tutorial/calc_pi
    mkdir build
    cd build
    cmake -DBLT_SOURCE_DIR=../../.. ..
    make

.. admonition:: blt_add_executable
   :class: hint
   
   This is one of the core macros that enables BLT to simplify our CMake-based project.
   It unifies many CMake calls into one easy to use macro.  
   ``blt_add_executable`` creates a CMake executable target with the 
   given sources, sets the output directory to ``<build dir>/bin`` (unless overridden with the
   macro parameter ``OUTPUT_DIR``) and handles internal and external dependencies in a greatly
   simplified manner.  There will be more on that in the following section.


Example 2: One library, one executable
--------------------------------------

This example is a bit more exciting.  This time we are creating a library that calculates
pi and then linking that library into an executable.

First, we create the library with the following BLT code:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt
   :language: cmake
   :lines: 36-38

Just like before, this creates a CMake library target that will get built to ``<build dir>/lib/libcalc_pi.a``.

Next, we create an executable named ``example_2`` and link in the previously created library target:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt
   :language: cmake
   :lines: 41-43

The ``DEPENDS_ON`` parameter properly links the previously defined library into this executable without any
more work or CMake function calls.


.. admonition:: blt_add_library
   :class: hint

   This is another core BLT macro.  It creates a CMake library target and associates the
   given sources and headers along with handling dependencies the same way as ``blt_add_executable``
   does.  It also provides a few commonly used build options, such as overriding the output name of the
   library and the output directory.  It defaults to building a static library unless you override it with
   ``SHARED`` or with the global CMake option ``BUILD_SHARED_LIBS``.

