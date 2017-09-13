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

Setup BLT in your CMake Project
=================================

BLT is easy to include into your CMake project whether it is an existing project or
you are starting from scratch. You simply pull it into your project using a CMake ``include()`` command.

.. code:: cmake

    include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

You can include the BLT source in your repository or you can pass the location of BLT at CMake configure time. 

There are two standard choices for including the BLT source in your repository:

1. Add BLT as a git submodule
2. Copy BLT into a subdirectory in your repository


BLT as a git submodule
----------------------

This example adds BLT as a submodule then commits and pushes the changes to your repository.

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

This example will clones BLT into your repository then removes the unneeded 
git files from the clone.  Finally it commits and pushes the changes to your
repository.

.. code:: bash

    cd <your repository>
    git clone git@github.com:LLNL/blt.git
    rm -rf blt/.git
    git commit -m "Adding BLT"
    git push


Include BLT in your project
---------------------------

In most projects, including BLT is as simple as including the following CMake
line in your base CMakeLists.txt after your project() call.

.. code:: cmake

    include(blt/SetupBLT.cmake)

This enables all of BLT's features in your project. 


Include BLT via BLT_SOURCE_DIR
--------------------------------

Some projects want BLT to live outside of their project.  This is usually to share
one instance of BLT between many dependent projects.

You can also include BLT from a directory outside of your source tree using `BLT_SOURCE_DIR`.

To support this, in your CMakeLists.txt file add:

.. literalinclude:: tutorial/blank_project/CMakeLists.txt
   :language: cmake
   :lines: 55

We do it in our blank_project example because we also want this project to be automatically tested
by just a single check-out of our repository.


Running CMake
-------------

To configure a project with CMake, first create a build directory and cd into it.  Then run cmake with pointing
it at your project.  Here is an example on how to do this with our blank project example.

.. code:: bash

    cd <your project>
    mkdir build
    cd build
    cmake ../

If you are using BLT outside of your project pass the location of BLT as follows:

.. code:: bash

    cd <your project>
    mkdir build
    cd build
    cmake -DBLT_SOURCE_DIR="path/to/blt" ../

Blank Project Example
---------------------

The blank project example is provided to show you some of the built-in features BLT provides and show
the bare minimum required for testing purposes.

BLT also enforces some build best practices such as not allowing in-source builds.  This means
that if you try to run cmake directly in your project. For example if you run the following commands:

.. code:: bash

    cd <BLT repository>/docs/tutorial/blank_project
    cmake . -DBLT_SOURCE_DIR=`pwd`/../.. 

you will get the following error:

.. code:: bash

    CMake Error at blt/SetupBLT.cmake:59 (message):
      In-source builds are not supported.  Please remove CMakeCache.txt from the
      'src' dir and configure an out-of-source build in another directory.
    Call Stack (most recent call first):
      CMakeLists.txt:55 (include)


    -- Configuring incomplete, errors occurred!

On the otherhand, to correctly run cmake create a build directory and run cmake from there as follows:

.. code:: bash

    cd <BLT repository>/docs/blank_project
    mkdir build
    cd build
    cmake .. -DBLT_SOURCE_DIR=`pwd`/../../..

This will generate among many things a configured Makefile in your build directory to build the blank
project.  This includes gtest, gmock, and several base built-in BLT tests depending on how many features
you have turned on in your build.  Blank project only has gtest.  To build use the following command:

.. code:: bash

    make

Just like any other make project you can utilize job tasks to speed up the build with the following:

.. code:: bash

    make -j8

Next run all tests (only blt_gtest_smoke) is in this project with the following:

.. code:: bash

    make test

If everything went correctly you should have the following output:

.. code:: bash

    Running tests...
    Test project blt/docs/tutorial/blank_project/build
        Start 1: blt_gtest_smoke
    1/1 Test #1: blt_gtest_smoke ..................   Passed    0.01 sec

    100% tests passed, 0 tests failed out of 1

    Total Test time (real) =   0.10 sec


Host-configs
------------

To capture (and revision control) build options, third party library paths, etc we recommend using CMake's initial-cache file mechanism. This feature allows you to pass a file to CMake that provides variables to bootstrap the configure process. 

You can pass initial-cache files to cmake via the ``-C`` command line option.

.. code:: bash
    
    cmake -C config_file.cmake


We call these initial-cache files ``host-config`` files, since we typically create a file for each platform or specific hosts if necessary. 


These files use standard CMake commands. CMake *set* commands need to specify ``CACHE`` as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")

Here is a snippet from a host-config file that specifies compiler details for using gcc 4.9.3 on LLNL's surface cluster. 

.. literalinclude:: tutorial/host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 9-22
   :linenos:

.. danger:: If using the host-config, ``ENABLE_CUDA`` should only be enabled on a machine with CUDA (otherwise, we get config errors).


