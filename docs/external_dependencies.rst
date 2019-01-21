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

External Dependencies
=====================

One key goal for BLT is to simplify the use of external dependencies when building your libraries and executables. 

To accomplish this BLT provides a ``DEPENDS_ON`` option for the
``blt_add_library`` and ``blt_add_executable`` macros that supports both CMake targets and external dependencies registered using the ``blt_register_library`` macro.

The ``blt_register_library`` macro allows you to reuse all information needed
for an external dependency under a single name.  This includes any include
directories, libraries, compile flags, link flags, defines, etc.  You can also
hide any warnings created by their headers by setting the
``TREAT_INCLUDES_AS_SYSTEM`` argument.

For example, to find and register the external dependency *axom* as a BLT registered library, you can simply use:

.. code-block:: cmake

    # FindAxom.cmake takes in AXOM_DIR, which is a installed Axom build and
    # sets variables AXOM_INCLUDES, AXOM_LIBRARIES
    include(FindAxom.cmake)
    blt_register_library(NAME      axom
                         TREAT_INCLUDES_AS_SYSTEM ON
                         DEFINES   HAVE_AXOM=1
                         INCLUDES  ${AXOM_INCLUDES}
                         LIBRARIES ${AXOM_LIBRARIES})

Then *axom* is available to be used in the DEPENDS_ON list in the following
``blt_add_executable`` or ``blt_add_library`` calls.

This is especially helpful for external libraries that are not built with CMake
and don't provide CMake friendly imported targets. Our ultimate goal is to use ``blt_register_library`` to import all external dependencies as first-class imported CMake targets to take full advanced of CMake's dependency lattice. 

MPI, CUDA, and OpenMP are all registered via ``blt_register_library``. You can see how these registered via ``blt_register_library`` in ``blt/thirdparty_builtin/CMakelists.txt``.

BLT also supports using ``blt_register_library`` to provide additional options for existing CMake targets. The implementation doesn't modify the properties of the existing targets, it just exposes these options via BLT's support for  ``DEPENDS_ON``.

.. admonition:: blt_register_library
   :class: hint

   A macro to register external libraries and dependencies with BLT.
   The named target can be added to the ``DEPENDS_ON`` argument of other BLT macros, like ``blt_add_library`` and ``blt_add_executable``.  


You have already seen one use of ``DEPENDS_ON`` for a BLT
registered dependency in test_1:  ``gtest``

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt 
   :language: cmake
   :lines: 46-51


``gtest`` is the name for the google test dependency in BLT registered via ``blt_register_library``.
Even though google test is built-in and uses CMake, ``blt_register_library`` allows us to easily set defines needed by all dependent targets.


MPI Example
~~~~~~~~~~~~~~~~~~~~~

Our next example, ``test_2``, builds and tests the ``calc_pi_mpi`` library,
which uses MPI to parallelize the calculation over the integration intervals.


To enable MPI, we set ``ENABLE_MPI``, ``MPI_C_COMPILER``, and ``MPI_CXX_COMPILER`` in our host config file. Here is a snippet with these settings for LLNL's Surface Cluster:

.. literalinclude:: ../host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 25-33


Here, you can see how ``calc_pi_mpi`` and ``test_2`` use ``DEPENDS_ON``:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt 
   :language: cmake
   :lines: 59-74


For MPI unit tests, you also need to specify the number of MPI Tasks
to launch. We use the ``NUM_MPI_TASKS`` argument to ``blt_add_test`` macro.

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt 
   :language: cmake
   :lines: 71-73



As mentioned in :ref:`UnitTesting`, google test provides a default ``main()`` driver
that will execute all unit tests defined in the source. To test MPI code we need to create a main that initializes and finalizes MPI in addition to google test. ``test_2.cpp`` provides an example driver for MPI with google test.


.. literalinclude:: tutorial/calc_pi/test_2.cpp
   :language: cpp
   :lines: 40-54


CUDA Example
~~~~~~~~~~~~~~~~~~~~~

Finally, ``test_3`` builds and tests the ``calc_pi_cuda`` library,
which uses CUDA to parallelize the calculation over the integration intervals.

To enable CUDA, we set ``ENABLE_CUDA``, ``CMAKE_CUDA_COMPILER``, and ``CUDA_TOOLKIT_ROOT_DIR`` in our host config file.  Also before enabling the CUDA language in CMake, you need to set ``CMAKE_CUDA_HOST_COMPILER`` in CMake 3.9+ or ``CUDA_HOST_COMPILER`` in previous versions.  If you do not call ``enable_language(CUDA)``, BLT will set the appropriate host compiler variable for you and enable the CUDA language.

Here is a snippet with these settings for LLNL's Surface Cluster:

.. literalinclude:: ../host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 39-44

Here, you can see how ``calc_pi_cuda`` and ``test_3`` use ``DEPENDS_ON``:

.. literalinclude:: tutorial/calc_pi/CMakeLists.txt 
   :language: cmake
   :lines: 82-100

The ``cuda`` dependency for ``calc_pi_cuda``  is a little special, 
along with adding the normal CUDA library and headers to your library or executable,
it also tells BLT that this target's C/CXX/CUDA source files need to be compiled via
``nvcc`` or ``cuda-clang``. If this is not a requirement, you can use the dependency
``cuda_runtime`` which also adds the CUDA runtime library and headers but will not
compile each source file with ``nvcc``.

Some other useful CUDA flags are:

.. code-block:: cmake

    # Enable separable compilation of all CUDA files for given target or all following targets
    set(CUDA_SEPARABLE_COMPILIATION ON CACHE BOOL “”)
    set(CUDA_ARCH “sm_60” CACHE STRING “”)
    set(CMAKE_CUDA_FLAGS “-restrict –arch ${CUDA_ARCH} –std=c++11” CACHE STRING “”)
    set(CMAKE_CUDA_LINK_FLAGS “-Xlinker –rpath –Xlinker /path/to/mpi” CACHE STRING “”)
    # Needed when you have CUDA decorations exposed in libraries
    set(CUDA_LINK_WITH_NVCC ON CACHE BOOL “”)

Here are the full example host-config files that use gcc 4.9.3 for LLNL's Surface, Ray and Quartz Clusters.

:download:`llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake <../host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake>`

:download:`llnl-ray-blue_os-gcc@4.9.3.cmake <../host-configs/llnl-ray-blue_os-clang-coral@2018.05.23.cmake>`

:download:`llnl-quartz-toss3-gcc@4.9.3.cmake <../host-configs/llnl-quartz-toss3-gcc@4.9.3.cmake>`

.. note::  Quartz does not have GPUs, so CUDA is not enabled in the Quartz host-config.

Here is a full example host-config file for an OSX laptop, using a set of dependencies built with spack.

:download:`llnl-naples-darwin-10.11-clang@7.3.0.cmake  <../host-configs/llnl-naples-darwin-10.11-clang@7.3.0.cmake>`


Building and testing on Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how you can use the host-config file to configure a build of the ``calc_pi``  project with MPI and CUDA enabled on Surface:

.. code-block:: bash
    
    # load new cmake b/c default on surface is too old
    use cmake-3.5.2
    # create build dir
    mkdir build
    cd build
    # configure using host-config
    cmake -C ../../host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake  -DBLT_SOURCE_DIR=../../../../blt ..

After building (``make``), you can run ``make test`` on a batch node (where the GPUs reside) to run the unit tests that are using MPI and CUDA:

.. code-block:: console

  bash-4.1$ salloc -A <valid bank>
  bash-4.1$ make   
  bash-4.1$ make test
  
  Running tests...
  Test project blt/docs/tutorial/calc_pi/build
      Start 1: test_1
  1/8 Test #1: test_1 ...........................   Passed    0.01 sec
      Start 2: test_2
  2/8 Test #2: test_2 ...........................   Passed    2.79 sec
      Start 3: test_3
  3/8 Test #3: test_3 ...........................   Passed    0.54 sec
      Start 4: blt_gtest_smoke
  4/8 Test #4: blt_gtest_smoke ..................   Passed    0.01 sec
      Start 5: blt_fruit_smoke
  5/8 Test #5: blt_fruit_smoke ..................   Passed    0.01 sec
      Start 6: blt_mpi_smoke
  6/8 Test #6: blt_mpi_smoke ....................   Passed    2.82 sec
      Start 7: blt_cuda_smoke
  7/8 Test #7: blt_cuda_smoke ...................   Passed    0.48 sec
      Start 8: blt_cuda_runtime_smoke
  8/8 Test #8: blt_cuda_runtime_smoke ...........   Passed    0.11 sec

  100% tests passed, 0 tests failed out of 8

  Total Test time (real) =   6.80 sec


Building and testing on Ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how you can use the host-config file to configure a build of the ``calc_pi``  project with MPI and CUDA 
enabled on the blue_os Ray cluster:

.. code-block:: bash
    
    # load new cmake b/c default on ray is too old
    ml cmake
    # create build dir
    mkdir build
    cd build
    # configure using host-config
    cmake -C ../../host-configs/llnl-ray-blue_os-gcc@4.9.3.cmake  -DBLT_SOURCE_DIR=../../../../blt ..

And here is how to build and test the code on Ray:

.. code-block:: console

  bash-4.2$ bsub -Is -n20 -G <valid group> bash
  bash-4.2$ make
  bash-4.2$ make test
  
  Running tests...
  Test project projects/blt/docs/tutorial/calc_pi/build
      Start 1: test_1
  1/7 Test #1: test_1 ...........................   Passed    0.01 sec
      Start 2: test_2
  2/7 Test #2: test_2 ...........................   Passed    1.24 sec
      Start 3: test_3
  3/7 Test #3: test_3 ...........................   Passed    0.17 sec
      Start 4: blt_gtest_smoke
  4/7 Test #4: blt_gtest_smoke ..................   Passed    0.01 sec
      Start 5: blt_mpi_smoke
  5/7 Test #5: blt_mpi_smoke ....................   Passed    0.82 sec
      Start 6: blt_cuda_smoke
  6/7 Test #6: blt_cuda_smoke ...................   Passed    0.15 sec
      Start 7: blt_cuda_runtime_smoke
  7/7 Test #7: blt_cuda_runtime_smoke ...........   Passed    0.04 sec
  
  100% tests passed, 0 tests failed out of 7
  
  Total Test time (real) =   2.47 sec  


