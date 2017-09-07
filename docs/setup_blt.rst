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

Setup BLT CMake Project
=======================

BLT is easy to include into your project whether it is an existing CMake project or
you are starting from scratch.  This example assumes you have a git repository and 
access to github.

You have two choices to include BLT in your repository:

1. Add BLT as a git submodule
2. Copy BLT into your repository

BLT as a git submodule
----------------------

This code example will add BLT as a submodule then commit and push the changes
to your repository.

.. code:: bash

    cd <your repository>
    git submodule add git@github.com:LLNL/blt.git blt
    git commit -m "Adding BLT"
    git push

At this point enabling BLT in your CMake project is trivial.  Just include the
following CMake lines in your base CMakeLists.txt after your project() call.
This example will give a meaningful error if the user doesn't recursively clone
or needs to init submodules.

.. code:: cmake

    if (NOT EXISTS ${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)
        message(FATAL_ERROR "\
    The BLT submodule is not present. \
    If in git repository run the following two commands:\n \
    git submodule init\n \
    git submodule update")

    endif()

    include(blt/SetupBLT.cmake)

Copy BLT into your repository
-----------------------------

This code example will clone BLT into your repository then remove the unneeded 
git files from the clone.  Finally it will commit and push the changes to your
repository.

.. code:: bash

    cd <your repository>
    git clone git@github.com:LLNL/blt.git
    rm -rf blt/.git
    git commit -m "Adding BLT"
    git push

At this point enabling BLT in your CMake project is trivial.  Just include the
following CMake line in your base CMakeLists.txt after your project() call.

.. code:: cmake

    include(blt/SetupBLT.cmake)


Host-configs
--------------


To capture (and revision control) build options, third party library paths, etc we recommend using CMake's initial-cache file mechanism. This feature allows you to pass a file to CMake that provides variables to bootstrap the configure process. 

You can pass initial-cache files to cmake via the ``-C`` command line option.

.. code:: bash
    
    cmake -C config_file.cmake


We call these initial-cache files ``host-config`` files, since we typically create a file for each platform or specific hosts if necessary. 


These files use standard CMake commands. CMake *set* commands need to specify ``CACHE`` as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")

For this section of the tutorial, we create a host-config file that specifies CMake details for a set of compilers on LLNL's surface cluster. 

First we set the paths to the compilers we want to use:

.. literalinclude:: tutorial/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 10-24
   :linenos:
