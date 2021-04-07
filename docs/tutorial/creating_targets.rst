.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level LICENSE file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _CreatingTargets:

Creating Targets
================

In the previous section, we learned the basics about how to create a CMake
project with BLT, how to configure the project and how to build, and test
BLT's built-in third party libraries.  

We now move on to creating CMake targets using two of BLT's core macros: 
``blt_add_library()`` and ``blt_add_executable()``.

We begin with a simple executable that calculates :math:`\pi` by numerical integration,
``example_1``. We then extract that code into a library, which we link
into a new executable, ``example_2``.


Example 1: Basic executable
---------------------------

This example is as basic as it gets. After setting up a BLT CMake project, 
like the Bare Bones project in the previous section, we can start using BLT's macros.  

Creating an executable is as simple as calling the following macro:

.. literalinclude:: calc_pi/CMakeLists.txt
   :start-after: _blt_tutorial_example_executable_start
   :end-before:  _blt_tutorial_example_executable_end
   :language: cmake

This tells CMake to create an executable, named ``example_1``, with one source file,
``example_1.cpp``.

You can create this project yourself or you can run the already provided
``tutorial/calc_pi`` project. For ease of use, we have combined many examples
into this one CMake project.  You can create the executable ``<build dir>/bin/example_1``,
by running the following commands:

.. code-block:: bash

    cd <BLT repository/docs/tutorial/calc_pi
    mkdir build
    cd build
    cmake -DBLT_SOURCE_DIR=../../.. ..
    make

.. admonition:: blt_add_executable
   :class: hint
   
   This is one of the core macros that enables BLT to simplify our CMake-based
   project. It unifies many CMake calls into one easy to use macro.
   ``blt_add_executable()`` creates a CMake executable target with the
   given sources, sets the output directory to ``<build dir>/bin``
   (unless overridden with the macro parameter ``OUTPUT_DIR``) and handles
   internal and external dependencies in a greatly simplified manner. There
   will be more on that in the following section.


Example 2: One library, one executable
--------------------------------------

This example is a bit more exciting.  This time we are creating a library
that calculates the value of pi and then linking that library into an executable.

First, we create the library with the following BLT code:

.. literalinclude:: calc_pi/CMakeLists.txt
   :start-after: _blt_tutorial_calcpi_library_start
   :end-before:  _blt_tutorial_calcpi_library_end
   :language: cmake

Just like before, this creates a CMake library target that will get built to
``<build dir>/lib/libcalc_pi.a``.

Next, we create an executable named ``example_2`` and link in the previously
created library target:

.. literalinclude:: calc_pi/CMakeLists.txt
   :start-after: _blt_tutorial_calcpi_example2_start
   :end-before:  _blt_tutorial_calcpi_example2_end
   :language: cmake

The ``DEPENDS_ON`` parameter properly links the previously defined library
into this executable without any more work or extra CMake function calls.


.. admonition:: blt_add_library
   :class: hint

   This is another core BLT macro. It creates a CMake library target and associates
   the given sources and headers along with handling dependencies the same way as
   ``blt_add_executable()`` does. It also provides a few commonly used build options,
   such as overriding the output name of the library and the output directory. 
   It defaults to building a static library unless you override it with
   ``SHARED`` or with the global CMake option ``BUILD_SHARED_LIBS``.
