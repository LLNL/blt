.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level LICENSE file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _CommonHPCDependencies:

Common HPC Dependencies
=======================

BLT creates named targets for the common HPC dependencies that most HPC projects need, such as
MPI, CUDA, HIP, and OpenMP.  As previously mentioned in :ref:`AddingTests`, BLT also provides 
bundled versions of GoogleTest, GoogleMock, GoogleBenchmark, and FRUIT.  Not only are the source
for these included, we provide named CMake targets for them as well.

BLT's ``mpi``, ``cuda``, ``cuda_runtime``, ``hip``, ``hip_runtime``,and ``openmp`` targets are
all defined via ``blt_import_library()``. This creates a true CMake imported target that is inherited
properly through the CMake dependency DAG.

.. note::
    If your project exports targets and you would like BLT's provided third-party targets to also be
    exported (for example, if a project that imports your project does not use BLT), you can set the 
    ``BLT_EXPORT_THIRDPARTY`` option to ``ON``.  As with other
    EXPORTABLE targets created by ``blt_import_library()``, these targets should be prefixed with the name of
    the project.  Either the ``EXPORT_NAME`` target property or the ``NAMESPACE`` option to CMake's ``install``
    command can be used to modify the name of an installed target.  See  :ref:`ExportingTargets:`
    for more info.

You have already seen one use of ``DEPENDS_ON`` for a BLT dependency, ``gtest``, in ``test_1``:

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_test1_executable_start
   :end-before:  _blt_tutorial_calcpi_test1_executable_end
   :language: cmake


``gtest`` is the name for the GoogleTest target in BLT registered via 
``blt_register_library()``. Even though GoogleTest is built-in and uses CMake,
``blt_register_library()`` allows us to easily set defines needed by all dependent
targets.


MPI
~~~

Our next example, ``test_2``, builds and tests the ``calc_pi_mpi`` library,
which uses MPI to parallelize the calculation over the integration intervals.


To enable MPI, we set ``ENABLE_MPI``, ``MPI_C_COMPILER``, and ``MPI_CXX_COMPILER``
in our host config file. Here is a snippet with these settings for LLNL's Pascal Cluster:

.. literalinclude:: ../../host-configs/llnl/toss_3_x86_64_ib/gcc@4.9.3_nvcc.cmake
   :start-after: _blt_tutorial_mpi_config_start
   :end-before:  _blt_tutorial_mpi_config_end
   :language: cmake


Here, you can see how ``calc_pi_mpi`` and ``test_2`` use ``DEPENDS_ON``:

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_test2_executable_start
   :end-before:  _blt_tutorial_calcpi_test2_executable_end
   :language: cmake


For MPI unit tests, you also need to specify the number of MPI Tasks
to launch. We use the ``NUM_MPI_TASKS`` argument to ``blt_add_test()`` macro.

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_test2_test_start
   :end-before:  _blt_tutorial_calcpi_test2_test_end
   :language: cmake


As mentioned in :ref:`AddingTests`, GoogleTest provides a default ``main()``
driver that will execute all unit tests defined in the source. To test MPI code,
we need to create a main that initializes and finalizes MPI in addition to Google
Test. ``test_2.cpp`` provides an example driver for MPI with GoogleTest.

.. literalinclude:: calc_pi/test_2.cpp
   :start-after: _blt_tutorial_calcpi_test2_main_start
   :end-before:  _blt_tutorial_calcpi_test2_main_end
   :language: cpp

.. note::
  While we have tried to ensure that BLT chooses the correct setup information
  for MPI, there are several niche cases where the default behavior is
  insufficient. We have provided several available override variables:
  
  * ``BLT_MPI_COMPILE_FLAGS``
  * ``BLT_MPI_INCLUDES``
  * ``BLT_MPI_LIBRARIES``
  * ``BLT_MPI_LINK_FLAGS``
  
  BLT also has the variable ``ENABLE_FIND_MPI`` which turns off all CMake's ``FindMPI``
  logic and then uses the MPI wrapper directly when you provide them as the default
  compilers.


CUDA
~~~~

Finally, ``test_3`` builds and tests the ``calc_pi_cuda`` library,
which uses CUDA to parallelize the calculation over the integration intervals.

To enable CUDA, we set ``ENABLE_CUDA``, ``CMAKE_CUDA_COMPILER``, and
``CUDA_TOOLKIT_ROOT_DIR`` in our host config file.  Also before enabling the
CUDA language in CMake, you need to set ``CMAKE_CUDA_HOST_COMPILER`` in CMake 3.9+
or ``CUDA_HOST_COMPILER`` in previous versions.  If you do not call 
``enable_language(CUDA)``, BLT will set the appropriate host compiler variable
for you and enable the CUDA language.

Here is a snippet with these settings for LLNL's Pascal Cluster:

.. literalinclude:: ../../host-configs/llnl/toss_3_x86_64_ib/gcc@4.9.3_nvcc.cmake
   :start-after: _blt_tutorial_cuda_config_start
   :end-before:  _blt_tutorial_cuda_config_end
   :language: cmake

Here, you can see how ``calc_pi_cuda`` and ``test_3`` use ``DEPENDS_ON``:

.. literalinclude:: calc_pi/CMakeLists.txt 
   :start-after: _blt_tutorial_calcpi_cuda_start
   :end-before:  _blt_tutorial_calcpi_cuda_end
   :language: cmake

The ``cuda`` dependency for ``calc_pi_cuda``  is a little special, 
along with adding the normal CUDA library and headers to your library or executable,
it also tells BLT that this target's C/C++/CUDA source files need to be compiled via
``nvcc`` or ``cuda-clang``. If this is not a requirement, you can use the dependency
``cuda_runtime`` which also adds the CUDA runtime library and headers but will not
compile each source file with ``nvcc``.

Some other useful CUDA flags are:

.. literalinclude:: ../../host-configs/llnl/blueos_3_ppc64le_ib_p9/clang@upstream_nvcc_xlf.cmake
   :start-after: _blt_tutorial_useful_cuda_flags_start
   :end-before:  _blt_tutorial_useful_cuda_flags_end
   :language: cmake


OpenMP
~~~~~~

To enable OpenMP, set ``ENABLE_OPENMP`` in your host-config file or before loading
``SetupBLT.cmake``.  Once OpenMP is enabled, simply add ``openmp`` to your library
executable's ``DEPENDS_ON`` list.

Here is an example of how to add an OpenMP enabled executable:

.. literalinclude:: ../../tests/smoke/CMakeLists.txt
   :start-after: _blt_tutorial_openmp_executable_start
   :end-before:  _blt_tutorial_openmp_executable_end
   :language: cmake

.. note::
  While we have tried to ensure that BLT chooses the correct compile and link flags for
  OpenMP, there are several niche cases where the default options are insufficient.
  For example, linking with NVCC requires to link in the OpenMP libraries directly instead
  of relying on the compile and link flags returned by CMake's FindOpenMP package.  An
  example of this is in ``host-configs/llnl/blueos_3_ppc64le_ib_p9/clang@upstream_link_with_nvcc.cmake``. 
  We provide two variables to override BLT's OpenMP flag logic: 
  
  * ``BLT_OPENMP_COMPILE_FLAGS``
  * ``BLT_OPENMP_LINK_FLAGS``

Here is an example of how to add an OpenMP enabled test that sets the amount of threads used:

.. literalinclude:: ../../tests/smoke/CMakeLists.txt
   :start-after: _blt_tutorial_openmp_test_start
   :end-before:  _blt_tutorial_openmp_test_end
   :language: cmake
