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

BLT
===

**Build, Link, and Triumph**


BLT is composition of CMake macros and several widely used open source tools assembled to simplify HPC software development. BLT is developed at Lawrence Livermore National Laboratory (LLNL) under a BSD-style open source license. 

BLT at a Glance
~~~~~~~~~~~~~~~~~~

* Simplifies Setup

  * CMake macros for:

    * Creating libraries and executables
    * Managing compiler flags
    * Managing external dependencies

  * Multi-platform support (HPC Platforms, OSX, Windows)

* Batteries Included

  * Built-in support for HPC Basics: MPI, OpenMP, and CUDA
  * Built-in Support for unit testing in C/C++ and Fortran
  
* Streamlines Development Processes

  * Support for Documentation Generation
  * Support for Code Health Tools:

    * Runtime and Static Analysis, Benchmarking


BLT User Tutorial
~~~~~~~~~~~~~~~~~~~
This tutorial is aimed at getting BLT users up and running as quickly as possible.

It provides instructions for:

    * Adding BLT to a CMake project
    * Setting up host-configs to handle multiple platform configurations
    * Building, linking, and installing libraries and executables
    * Setting up unit tests with GTest
    * Using external project dependencies
    * Creating documentation with Sphinx and Doxygen

**Tutorial Contents**

.. toctree::
    :maxdepth: 3

    setup_blt
    add_target
    unit_testing
    external_dependencies
    creating_documentation
    other_features

