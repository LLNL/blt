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

This macro adds a benchmark test to the ``Benchmark`` CTest configuration
which can be run by the ``run_benchmarks`` build target.  These tests are
not run when you use the regular ``test`` build target.

This macro is just a thin wrapper around :ref:`blt_add_test` and assists 
with building up the correct command line for running the benchmark.  For more
information see :ref:`blt_add_test`.

The underlying executable should be previously added to the build system
with :ref:`blt_add_executable`. It should include the necessary benchmarking 
library in its DEPENDS_ON list.

Any calls to this macro should be guarded with ENABLE_BENCHMARKS unless this option
is always on in your build project.

.. note::
  BLT provides a built-in Google Benchmark that is enabled by default if you set
  ENABLE_BENCHMARKS=ON and can be turned off with the option ENABLE_GBENCHMARK.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    if(ENABLE_BENCHMARKS)
        blt_add_executable(NAME    component_benchmark
                           SOURCES my_benchmark.cpp
                           DEPENDS gbenchmark)
        blt_add_benchmark(
             NAME    component_benchmark
             COMMAND component_benchmark "--benchmark_min_time=0.0 --v=3 --benchmark_format=json")
    endif()

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

Adds an executable target to the project.

NAME
  Name of the created CMake target

SOURCES
  List of all sources to be added

INCLUDES
  List of include directories both used by this target and inherited by dependent
  targets

DEFINES
  List of compiler defines both used by this target and inherited by dependent
  targets

DEPENDS_ON
  List of CMake targets and BLT registered libraries that this target
  depends on

OUTPUT_DIR
  Directory that this target will built to, defaults to bin

FOLDER
  Name of the IDE folder to ease organization

Adds an executable target, called <name>, to be built from the given sources.
It also adds the given INCLUDES and DEFINES from the parameters to this macro
and adds all inherited information from the list given by DEPENDS_ON.  This
macro creates a true CMake target that can be altered by other CMake commands
like normal, such as `set_target_property()`.

.. note::
  If the first entry in SOURCES is a Fortran source file, the fortran linker 
  is used. (via setting the CMake target property LINKER_LANGUAGE to Fortran )

.. note::
  The FOLDER option is only used when ENABLE_FOLDERS is ON and when the CMake generator
  supports this feature and will otherwise be ignored. 



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

Adds a library target to your project.

NAME
  Name of the created CMake target

SOURCES
  List of all sources to be added

HEADERS
  List of all headers to be added

INCLUDES
  List of include directories both used by this target and inherited by dependent
  targets

DEFINES
  List of compiler defines both used by this target and inherited by dependent
  targets

DEPENDS_ON
  List of CMake targets and BLT registered libraries that this library
  depends on

OUTPUT_NAME
  Override built file name of library (defaults to <NAME>)  

OUTPUT_DIR
  Directory that this target will built to

SHARED
  Builds library as shared and overrides global BUILD_SHARED_LIBS (defaults to OFF)

OBJECT
  Create an Object library

CLEAR_PREFIX
  Removes library prefix (defaults to 'lib' on linux)

FOLDER
  Name of the IDE folder to ease organization

This macro creates a true CMake target that can be altered by other CMake commands
like normal, such as `set_target_property()`.

This macro supports three types of libraries automatically: normal, header-only,
or object.

Normal libraries are libraries that have sources that are compiled and linked into a single
library and have headers that go along with them (unless it's a Fortran library).

Header-only libraries are useful when you do not want the library separately compiled or 
are using C++ templates that require the library's user to instatiate them. These libraries
have headers but no sources. To create a header-only library (CMake calls them INTERFACE libraries),
simply list all headers under the HEADER argument and do not specify SOURCES (because there aren't any).

Object libraries are basically a collection of compiled source files that are not
archived or linked. They are sometimes useful when you want to solve compilicated linking
problems (like circular dependencies) or when you want to combine smaller libraries into
one larger library but don't want the linker to remove unused symbols. Unlike regular CMake
object libraries you do not have to use the ``$<TARGET_OBJECTS:<libname>>`` syntax, you can just
use <libname> with BLT macros.  Unless you have a good reason don't use Object libraries.

.. note::
  BLT Object libraries do not follow CMake's normal transitivity rules. Due to CMake requiring
  you install the individual object files if you install the target that uses them. BLT manually
  adds the INTERFACE target properties to get around this.

This macro uses the BUILD_SHARED_LIBS, which is defaulted to OFF, to determine
whether the library will be built as shared or static. The optional boolean
SHARED argument can be used to override this choice.

If given a DEPENDS_ON argument, this macro will inherit the necessary information
from all targets given in the list.  This includes CMake targets as well as any
BLT registered libraries already defined via :ref:`blt_register_library`.  To ease
use, all information is used by this library and inherited by anything depending on this
library (CMake PUBLIC inheritance).

OUTPUT_NAME is useful when multiple libraries with the same name need to be created
by different targets. For example, you might want to build both a shared and static
library in the same build instead of building twice, once with BUILD_SHARED_LIBS set to ON
and then with OFF. NAME is the CMake target name, OUTPUT_NAME is the created library name.

.. note::
  The FOLDER option is only used when ENABLE_FOLDERS is ON and when the CMake generator
  supports this feature and will otherwise be ignored. 


.. _blt_add_test:

blt_add_test
~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_test( NAME            [name]
                  COMMAND         [command]
                  NUM_MPI_TASKS   [n]
                  NUM_OMP_THREADS [n]
                  CONFIGURATIONS  [config1 [config2...]])

Adds a test to the project.

NAME
  Name that CTest reports.

COMMAND
  Command line that will be used to run the test and can include arguments.

NUM_MPI_TASKS
  Indicates this is an MPI test and how many MPI tasks to use.

NUM_OMP_THREADS
  Indicates this test requires the defined environment variable OMP_NUM_THREADS set to the given variable.

CONFIGURATIONS
  Set the CTest configuration for this test.  When not specified, the test
  will be added to the default CTest configuration.

This macro adds the named test to CTest, which is run by the build target ``test``. This macro
does not build the executable and requires a prior call to :ref:`blt_add_executable`.

This macro assists with building up the correct command line. It will prepend
the RUNTIME_OUTPUT_DIRECTORY target property to the executable.

If NUM_MPI_TASKS is given or ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC is set, the macro 
will appropiately use MPIEXEC, MPIEXEC_NUMPROC_FLAG, and BLT_MPI_COMMAND_APPEND 
to create the MPI run line.

MPIEXEC and MPIEXEC_NUMPROC_FLAG are filled in by CMake's FindMPI.cmake but can
be overwritten in your host-config specific to your platform. BLT_MPI_COMMAND_APPEND
is useful on machines that require extra arguments to MPIEXEC.

If NUM_OMP_THREADS is given, this macro will set the environment variable OMP_NUM_THREADS
before running this test.  This is done by appending to the CMake tests property.

.. note::
  If you do not require this macros command line assistance, you can call CMake's
  ``add_test()`` directly. For example, you may have a script checked into your
  repository you wish to run as a test instead of an executable you built as a part
  of your build system.

Any calls to this macro should be guarded with ENABLE_TESTS unless this option
is always on in your build project.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    if (ENABLE_TESTS)
        blt_add_executable(NAME    my_test
                           SOURCES my_test.cpp)
        blt_add_test(NAME    my_test
                     COMMAND my_test --with-some-argument)
    endif()


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
