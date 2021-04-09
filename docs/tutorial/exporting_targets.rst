.. # Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level LICENSE file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _ExportingTargets:

Exporting Targets
=================

BLT provides several built-in targets for commonly used libraries:

 * ``mpi`` (when ``ENABLE_MPI`` is ``ON``)
 * ``openmp`` (when ``ENABLE_OPENMP`` is ``ON``)
 * ``cuda`` and ``cuda_runtime`` (when ``ENABLE_CUDA`` is ``ON``)
 * ``hip`` and ``hip_runtime`` (when ``ENABLE_HIP`` is ``ON``)

These targets can be made exportable in order to make them available to users of
your project via CMake's ``install()`` command.  Setting BLT's ``BLT_EXPORT_THIRDPARTY``
option to ``ON`` will mark all active targets in the above list as ``EXPORTABLE``
(see the :ref:`blt_import_library` API documentation for more info).

.. note::  As with other ``EXPORTABLE`` targets created by ``blt_import_library()``,
    these targets should be prefixed with the name of the project.  Either the ``EXPORT_NAME``
    target property or the ``NAMESPACE`` option to CMake's ``install``
    command can be used to modify the name of an installed target.

.. note:: If a target in your project is added to an export set, any of its dependencies
    marked ``EXPORTABLE`` must be added to the same export set.  Failure to add them will
    result in a CMake error in the exporting project.

Typical usage of the ``BLT_EXPORT_THIRDPARTY`` option is as follows:

.. code-block:: cmake

    # BLT configuration - enable MPI
    set(ENABLE_MPI ON CACHE BOOL "")
    # and mark the subsequently created MPI target as exportable
    set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")
    # Both of the above must happen before SetupBLT.cmake is included
    include(/path/to/SetupBLT.cmake)

    # Later, a project might mark a target as dependent on MPI
    blt_add_executable( NAME    example_1
                        SOURCES example_1.cpp
                        DEPENDS_ON mpi )

    # Add the example_1 target to the example-targets export set
    install(TARGETS example_1 EXPORT example-targets)

    # Add the MPI target to the same export set - this is required
    # because the mpi target was marked exportable
    install(TARGETS mpi EXPORT example-targets)

To avoid collisions with projects that import "example-targets", there are
two options for adjusting the exported name of the ``mpi`` target.

The first is to rename only the ``mpi`` target's exported name:

.. code-block:: cmake

    set_target_properties(mpi PROPERTIES EXPORT_NAME example::mpi)
    install(EXPORT example-targets)

With this approach the ``example_1`` target's exported name is unchanged - a 
project that imports the ``example-targets`` export set will have ``example_1``
and ``example::mpi`` targets made available.  The imported ``example_1`` will
depend on ``example::mpi``.

Another approach is to install all targets in the export set behind a namespace:

.. code-block:: cmake

    install(EXPORT example-targets NAMESPACE example::)

With this approach all targets in the export set are prefixed, so an importing
project will have ``example::example_1`` and ``example::mpi`` targets made available.
The imported ``example::example_1`` will depend on ``example::mpi``.
