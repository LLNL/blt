.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Target Macros
=============

.. _blt_add_benchmark:

blt_add_benchmark
~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_benchmark( NAME          [name]
                       COMMAND       [command]
                       NUM_MPI_TASKS [n])

Adds a benchmark to the project.

NAME
  Name that CTest reports.

COMMAND
  Command line that will be used to run the test and can include arguments.  

NUM_MPI_TASKS
  Indicates this is an MPI test and how many MPI tasks to use.

This macro adds the benchmark to the ``run_benchmarks`` build target.
The underlying executable, previously added with :ref:`blt_add_executable`,
should include necessary benchmarking library in its DEPENDS_ON list.

BLT provides a built-in Google Benchmark that is enabled by default if you set
ENABLE_BENCHMARKS=ON and can be turned off with the option ENABLE_GBENCHMARK.

.. note::
  This is just a thin wrapper around :ref:`blt_add_test` that sets the CTest configuration
  to "Benchmark" to filter these benchmarks out of the ``test`` build target. It also assists
  with building up the correct command line.  For more info see :ref:`blt_add_test`.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_add_executable(NAME    component_benchmark
                       SOURCES my_benchmark.cpp
                       DEPENDS gbenchmark)
    blt_add_benchmark(
         NAME    component_benchmark
         COMMAND component_benchmark "--benchmark_min_time=0.0 --v=3 --benchmark_format=json")


.. _blt_add_executable:

blt_add_executable
~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_executable( NAME       <name>
                        SOURCES    [source1 [source2 ...]]
                        INCLUDES   [dir1 [dir2 ...]]
                        DEFINES    [define1 [define2 ...]]
                        DEPENDS_ON [dep1 [dep2 ...]]
                        OUTPUT_DIR [dir]
                        FOLDER     [name])

Adds an executable target, called <name>, to be built from the given sources.

The INCLUDES argument allows you to define what include directories are
needed to compile this executable.

The DEFINES argument allows you to add needed compiler definitions that are
needed to compile this executable.

If given a DEPENDS_ON argument, it will add the necessary includes and 
libraries if they are already registered with blt_register_library.  If
not it will add them as a cmake target dependency.

The OUTPUT_DIR is used to control the build output directory of this 
executable. This is used to overwrite the default bin directory.

If the first entry in SOURCES is a Fortran source file, the fortran linker 
is used. (via setting the CMake target property LINKER_LANGUAGE to Fortran )
FOLDER is an optional keyword to organize the target into a folder in an IDE.

This is available when ENABLE_FOLDERS is ON and when using a cmake generator
that supports this feature and will otherwise be ignored.


.. _blt_add_library:

blt_add_library
~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_library( NAME         <libname>
                     SOURCES      [source1 [source2 ...]]
                     HEADERS      [header1 [header2 ...]]
                     INCLUDES     [dir1 [dir2 ...]]
                     DEFINES      [define1 [define2 ...]]
                     DEPENDS_ON   [dep1 ...] 
                     OUTPUT_NAME  [name]
                     OUTPUT_DIR   [dir]
                     SHARED       [TRUE | FALSE]
                     OBJECT       [TRUE | FALSE]
                     CLEAR_PREFIX [TRUE | FALSE]
                     FOLDER       [name])

Adds a library target, called <libname>, to be built from the given sources.

This macro uses the BUILD_SHARED_LIBS, which is defaulted to OFF, to determine
whether the library will be build as shared or static. The optional boolean
SHARED argument can be used to override this choice.

The OBJECT argument creates a CMake object library. Basically it is a collection
of compiled source files that are not archived or linked. Unlike regular CMake
object libraries you do not have to use the $<TARGET_OBJECTS:<libname>> syntax,
you can just use <libname>.

Note: Object libraries do not follow CMake's transitivity rules until 3.12.
BLT will add the various information provided in this macro and its
dependencies in the order you provide them to help.

The INCLUDES argument allows you to define what include directories are
needed by any target that is dependent on this library.  These will
be inherited by CMake's target dependency rules.

The DEFINES argument allows you to add needed compiler definitions that are
needed by any target that is dependent on this library.  These will
be inherited by CMake's target dependency rules.

If given a DEPENDS_ON argument, it will add the necessary includes and 
libraries if they are already registered with blt_register_library.  If 
not it will add them as a CMake target dependency.

In addition, this macro will add the associated dependencies to the given
library target. Specifically, it will add the dependency for the CMake target
and for copying the headers for that target as well.

The OUTPUT_DIR is used to control the build output directory of this 
library. This is used to overwrite the default lib directory.
OUTPUT_NAME is the name of the output file; the default is NAME.

It's useful when multiple libraries with the same name need to be created
by different targets. NAME is the target name, OUTPUT_NAME is the library name.
CLEAR_PREFIX allows you to remove the automatically appended "lib" prefix
from your built library.  The created library will be foo.a instead of libfoo.a.
FOLDER is an optional keyword to organize the target into a folder in an IDE.

This is available when ENABLE_FOLDERS is ON and when the cmake generator
supports this feature and will otherwise be ignored. 

Note: Do not use with header-only (INTERFACE) libraries, as this will generate 
a CMake configuration error.


.. _blt_add_test:

blt_add_test
~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_test( NAME           [name]
                  COMMAND        [command]
                  NUM_MPI_TASKS  [n]
                  CONFIGURATIONS [config1 [config2...]])

Adds a test to the project.

NAME
  Name that CTest reports.

COMMAND
  Command line that will be used to run the test and can include arguments.

NUM_MPI_TASKS
  Indicates this is an MPI test and how many MPI tasks to use.

CONFIGURATIONS
  Set the CTest configuration for this test.  Do not specify if you want the
  test to run every ``test`` build target.

This macro adds the named test to CTest and the build target ``test``. This macro
does not build the executable and requires a call to :ref:`blt_add_executable` first.

This macro assists with building up the correct command line. It will prepend
the RUNTIME_OUTPUT_DIRECTORY target property to the executable. If NUM_MPI_TASKS
is given, the macro will appropiately use MPIEXEC, MPIEXEC_NUMPROC_FLAG, and 
BLT_MPI_COMMAND_APPEND to create the MPI run line.

MPIEXEC and MPIEXEC_NUMPROC_FLAG are filled in by CMake's FindMPI.cmake but can
be overwritten in your host-config specific to your platform. BLT_MPI_COMMAND_APPEND
is useful on machines that require extra arguments to MPIEXEC.

.. note::
  If you do not want the command line assistance, for example you already have a script
  you wish to run as a test, then just call CMake's ``add_test()``.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_add_executable(NAME    my_test
                       SOURCES my_test.cpp)
    blt_add_test(NAME    my_test
                 COMMAND my_test --with-some-argument)


.. _blt_register_library:

blt_register_library
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_register_library( NAME                     <libname>
                          DEPENDS_ON               [dep1 [dep2 ...]]
                          INCLUDES                 [include1 [include2 ...]]
                          TREAT_INCLUDES_AS_SYSTEM [ON|OFF]
                          FORTRAN_MODULES          [path1 [path2 ..]]
                          LIBRARIES                [lib1 [lib2 ...]]
                          COMPILE_FLAGS            [flag1 [flag2 ..]]
                          LINK_FLAGS               [flag1 [flag2 ..]]
                          DEFINES                  [def1 [def2 ...]] )

Registers a library to the project to ease use in other BLT macro calls.

Stores information about a library in a specific way that is easily recalled
in other macros.  For example, after registering gtest, you can add gtest to
the DEPENDS_ON in your blt_add_executable call and it will add the INCLUDES
and LIBRARIES to that executable.

TREAT_INCLUDES_AS_SYSTEM informs the compiler to treat this library's include paths
as system headers.  Only some compilers support this. This is useful if the headers
generate warnings you want to not have them reported in your build. This defaults
to OFF.

This does not actually build the library.  This is strictly to ease use after
discovering it on your system or building it yourself inside your project.

Note: The OBJECT parameter is for internal BLT support for object libraries
and is not for users.  Object libraries are created using blt_add_library().

Internally created variables (NAME = "foo"):
    | BLT_FOO_IS_REGISTERED_LIBRARY
    | BLT_FOO_IS_OBJECT_LIBRARY
    | BLT_FOO_DEPENDS_ON
    | BLT_FOO_INCLUDES
    | BLT_FOO_TREAT_INCLUDES_AS_SYSTEM
    | BLT_FOO_FORTRAN_MODULES
    | BLT_FOO_LIBRARIES
    | BLT_FOO_COMPILE_FLAGS
    | BLT_FOO_LINK_FLAGS
    | BLT_FOO_DEFINES
