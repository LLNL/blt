# <img src="/share/blt/logo/blt_logo.png?raw=true" width="128" valign="middle" alt="BLT"/> v0.2

[![Build Status](https://travis-ci.org/LLNL/blt.svg)](https://travis-ci.org/LLNL/blt)
[![Build Status](https://ci.appveyor.com/api/projects/status/fuaftu9mvp0y488j/branch/master?svg=true)](https://ci.appveyor.com/project/cyrush/blt/branch/master)
[![Documentation Status](https://readthedocs.org/projects/llnl-blt/badge/?version=latest)](https://llnl-blt.readthedocs.io/en/latest/?badge=latest)

BLT is a streamlined `CMake <https://cmake.org/>`_-based foundation for building, linking and testing HPC software.

BLT makes is easy to get up and running on a wide range of compilers (gcc, clang, intel, XL, Visual Studio), 
operating systems (Linux, Mac, Windows) and HPC technologies (MPI, OpenMP, CUDA, HIP).

Getting started
---------------

BLT is easy to pull into in your existing or new CMake-based project using a single CMake `include()` command:

  ```cmake
  include(path/to/blt/SetupBLT.cmake)
  ```

For more information, please check our user documentation and tutorial at https://llnl-blt.readthedocs.io/en/latest/

Questions
---------

Any questions can be sent to blt-dev@llnl.gov.

Authors
-------

Developers include:

 * Chris White (white238@llnl.gov)
 * Kenneth Weiss (kweiss@llnl.gov)
 * Cyrus Harrison (harrison37@llnl.gov)
 * George Zagaris (zagaris2@llnl.gov)
 * Lee Taylor (taylor16@llnl.gov)
 * Aaron Black (black27@llnl.gov)
 * David A. Beckingsale (beckingsale1@llnl.gov)
 * Richard Hornung (hornung1@llnl.gov)
 * Randolph Settgast (settgast1@llnl.gov)

Please see the `BLT Contributors Page <https://github.com/LLNL/BLT/graphs/contributors>`_ for the full list of project contributors.

Projects using this library
---------------------------

 * `Ascent <https://github.com/Alpine-DAV/ascent>`_: A flyweight in-situ visualization and analysis runtime for multi-physics HPC simulations
 * `Axom <https://github.com/LLNL/axom>`_: Software infrastructure for the development of multi-physics applications and computational tools
 * `CHAI <https://github.com/LLNL/CHAI>`_: Copy-hiding array abstraction to automatically migrate data between memory spaces
 * `Conduit <https://github.com/LLNL/conduit>`_: Simplified data exchange for HPC simulations
 * `RAJA <https://github.com/LLNL/raja>`_: Performance portability layer for HPC
 * `Umpire <https://github.com/LLNL/Umpire>`_: Application-focused API for memory management on NUMA and GPU architectures
 * `VTK-h <https://github.com/Alpine-DAV/vtk-h>`_: Scientific visualization algorithms for emerging processor architectures

If you would like to add a library to this list, please let us know 
via `email <mailto:blt-dev@llnl.gov>`_
or by submitting an `issue <https://github.com/LLNL/blt/issues>`_ 
or `pull-request <https://github.com/LLNL/blt/pulls>`_.

Release
-------

Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

All rights reserved.

Unlimited Open Source - BSD Distribution
`LLNL-CODE-725085`  `OCEC-17-023`

Additional license and copyright information can be found in the following files:
 * [LICENSE](./LICENSE)
 * [COPYRIGHT](./COPYRIGHT) 

