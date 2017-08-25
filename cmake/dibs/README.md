Installation
-----------

To install, 

    cmake -DSPACK\_ROOT=... -DCMAKE\_INSTALL\_PREFIX=...

CMake 
-----------

Dibs allows users to integrate Spack into their CMake builds. For a sample CMakeLists.txt taking advantage of Dibs, see [the example](dibs_test/CMakeLists.txt)

The Dibs interface allows a user to do things in their CMakeLists like

    require\_spack\_package(raja caliper openmpi+cuda)
    find\_package(raja)
    set(CMAKE\_CXX\_FLAGS ${CMAKE\_CXX\_FLAGS} ${RAJA\_COMPILE\_FLAGS})
    target\_link\_libraries(my\_executable caliper caliper-common)

And have the packages they requested installed by Spack if they aren't already, used if they are, and available in the build for things like find\_package

Invocation
-----------

There are two main ways to use Dibs

  * Use Dibs to choose your compilers
  * Use CMake directly, and ensure that your compilers are compatible with the packages you request

To invoke dibs directly, try

    dibs spack\_compiler\_spec typical\_cmake\_args

For example

    dibs gcc@4.9.3 -DCMAKE\_INSTALL\_PREFIX=$HOME -DCMAKE\_CXX\_FLAGS=...

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
