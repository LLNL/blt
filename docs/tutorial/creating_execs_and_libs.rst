.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _CreatingLibrariesAndExecutables:

Creating Libraries and Executables
==================================

In the previous section, we learned the basics about how to create a CMake
project with BLT, how to configure the project and how to build, and test
BLT's built-in third party libraries.  

We now move on to creating libraries and executables
using two of BLT's core macros: ``blt_add_library()`` and ``blt_add_executable()``.

We begin with a simple executable that calculates :math:`\pi` by numerical integration,
``example_1``. We then extract that code into a library, which we link
into a new executable, ``example_2``.


Example 1: Basic executable
---------------------------

This example is as basic as it gets. After setting up a BLT CMake project, 
like ``blank_project`` in the previous section, we can start using BLT's macros.  

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

Object Libraries
----------------

BLT has simplified the use of CMake object libraries through the 
``blt_add_library()`` macro. Object libraries are a collection of object files
that are not linked or archived into a library. They are used in other libraries
or executables through the ``DEPENDS_ON`` macro argument. This is generally
useful for combining smaller libraries into a larger library without 
the linker removing unused symbols in the larger library.

.. code-block:: cmake

    blt_add_library(NAME    myObjectLibrary
                    SOURCES source1.cpp
                    HEADERS header1.cpp
                    OBJECT  TRUE)

    blt_add_exectuble(NAME       helloWorld
                      SOURCES    main.cpp
                      DEPENDS_ON myObjectLibrary)

.. note::
  Due to record keeping on BLT's part to make object libraries as easy to use
  as possible, you need to define object libraries before you use them
  if you need their inheritable information to be correct.

..
.. ------------------------------------
.. CYRUS NOTE: 
.. Object Libraries CUDA Considerations
.. this seems like a detour given we haven't talked about CUDA yet
.. ------------------------------------
..


If you are using separable CUDA compilation (relocatable device code) in your
object library, users of that library will be required to use NVCC to link their
executables - in general, only NVCC can perform the "device link" step.  To remove
this restriction, you can enable the ``CUDA_RESOLVE_DEVICE_SYMBOLS`` property on
an object library:

.. code-block:: cmake

   set_target_properties(myObjectLibrary PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

To enable this device linking step for all libraries in your project (including object libraries), you
can set the ``CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS`` option to ``ON``.  This defaults the
``CUDA_RESOLVE_DEVICE_SYMBOLS`` target property to ``ON`` for all targets created by BLT.

You can read more about this property in the
`CMake documentation <https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RESOLVE_DEVICE_SYMBOLS.html>`_.

.. note::
  These options only apply when an object library in your project is linked later
  into a shared or static library, in which case a separate object file containing
  device symbols is created and added to the "final" library.  Object libraries
  provided directly to users of your project will still require a device link step.

The ``CUDA_RESOLVE_DEVICE_SYMBOLS`` property is also supported for static and shared libraries.
By default, it is enabled for shared libraries but disabled for static libraries.
