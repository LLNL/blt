Installation
-----------

To install, 

    cmake -DSPACK_ROOT=... -DCMAKE_INSTALL_PREFIX=...

CMake 
-----------

Dibs allows users to integrate Spack into their CMake builds. For a sample CMakeLists.txt taking advantage of Dibs, see [the example](dibs_test/CMakeLists.txt)

The Dibs interface allows a user to do things in their CMakeLists like

    require_spack_package(raja caliper openmpi+cuda)
    find_package(raja)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${RAJA_COMPILE_FLAGS})
    target_link_libraries(my_executable caliper caliper-common)

And have the packages they requested installed by Spack if they aren't already, used if they are, and available in the build for things like find_package

Invocation
-----------

There are two main ways to use Dibs

  * Use Dibs to choose your compilers
  * Use CMake directly, and ensure that your compilers are compatible with the packages you request

To invoke dibs directly, try

    dibs spack_compiler_spec typical_cmake_args

For example

    dibs gcc@4.9.3 -DCMAKE_INSTALL_PREFIX=$HOME -DCMAKE_CXX_FLAGS=...

Release
-------

Copyright (c) 2017, Lawrence Livermore National Security, LLC.

Produced at the Lawrence Livermore National Laboratory.

All rights reserved.

Unlimited Open Source - BSD Distribution

For release details and restrictions, please read the LICENSE.txt file.
It is also linked here:
- [LICENSE](../../LICENSE)

`LLNL-CODE-725085`  `OCEC-17-023`
