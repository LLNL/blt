# <img src="/share/blt/logo/blt_logo.png?raw=true" width="128" valign="middle" alt="BLT"/> v0.2.5

[![Build Status](https://travis-ci.org/LLNL/blt.svg)](https://travis-ci.org/LLNL/blt)
[![Build Status](https://ci.appveyor.com/api/projects/status/fuaftu9mvp0y488j/branch/master?svg=true)](https://ci.appveyor.com/project/cyrush/blt/branch/master)
[![Documentation Status](https://readthedocs.org/projects/llnl-blt/badge/?version=develop)](https://llnl-blt.readthedocs.io/en/develop/?badge=develop)

BLT is a streamlined [CMake](https://cmake.org)-based foundation for 
<b>B</b>uilding, <b>L</b>inking and <b>T</b>esting large-scale high performance computing (HPC) applications.

BLT makes it easy to get up and running on a wide range of HPC compilers, 
operating systems and technologies:
 * Compiler families:
      [gcc](https://gcc.gnu.org), 
      [clang](https://clang.llvm.org), 
      [Intel](https://software.intel.com/en-us/compilers), 
      [XL](https://www.ibm.com/us-en/marketplace/ibm-c-and-c-plus-plus-compiler-family), 
      [Visual Studio](https://visualstudio.microsoft.com/vs/features/cplusplus)
 * Operating systems: 
      Linux, 
      Mac OS, 
      Windows
 * HPC programming models:
      [MPI](https://www.mpi-forum.org/), 
      [OpenMP](https://www.openmp.org/), 
      [CUDA](https://developer.nvidia.com/cuda-zone), 
      [HIP](https://gpuopen.com/compute-product/hip-convert-cuda-to-portable-c-code)
 * Unit testing and benchmarking (built-in):
      [Google Test (gtest and gmock)](https://github.com/google/googletest),
      [FRUIT](https://sourceforge.net/projects/fortranxunit),
      [gbenchmark](https://github.com/google/benchmark)
 * Documentation:
      [Doxygen](http://www.doxygen.nl/), 
      [Sphinx](http://www.sphinx-doc.org)
 * Code style and health:
      [Uncrustify](http://uncrustify.sourceforge.net), 
      [AStyle](http://astyle.sourceforge.net), 
      [Cppcheck](http://cppcheck.sourceforge.net),
      [clang-query](http://clang.llvm.org/docs/LibASTMatchers.html)
 

Getting started
---------------

BLT is easy to pull into an existing or new CMake-based project using a single CMake `include()` command:

  ```cmake
  include(path/to/blt/SetupBLT.cmake)
  ```

For more information, please check our [user documentation and tutorial](https://llnl-blt.readthedocs.io).

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

Please see the [BLT Contributors Page](https://github.com/LLNL/BLT/graphs/contributors) for the full list of project contributors.

Open-Source Projects using BLT
------------------------------

 * [Ascent](https://github.com/Alpine-DAV/ascent): A flyweight in-situ visualization and analysis runtime for multi-physics HPC simulations
 * [Axom](https://github.com/LLNL/axom): Software infrastructure for the development of multi-physics applications and computational tools
 * [CHAI](https://github.com/LLNL/CHAI): Copy-hiding array abstraction to automatically migrate data between memory spaces
 * [Conduit](https://github.com/LLNL/conduit): Simplified data exchange for HPC simulations
 * [RAJA](https://github.com/LLNL/raja): Performance portability layer for HPC
 * [Umpire](https://github.com/LLNL/Umpire): Application-focused API for memory management on NUMA and GPU architectures
 * [VTK-h](https://github.com/Alpine-DAV/vtk-h): Scientific visualization algorithms for emerging processor architectures

If you would like to add a library to this list, please let us know via [email](mailto:blt-dev@llnl.gov)
or by submitting an [issue](https://github.com/LLNL/blt/issues) or [pull-request](https://github.com/LLNL/blt/pulls).

License
-------

BLT is licensed under the BSD 3-Clause license,
(BSD-3-Clause or https://opensource.org/licenses/BSD-3-Clause).

Copyrights and patents in the BLT project are retained by contributors.
No copyright assignment is required to contribute to BLT.

See [LICENSE](./LICENSE) for details.
 
Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-725085`  `OCEC-17-023`

SPDX usage
------------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

External Packages
-------------------
BLT bundles its external dependencies in thirdparty_builtin/.  These
packages are covered by various permissive licenses.  A summary listing
follows.  See the license included with each package for full details.

PackageName: fruit  
PackageHomePage: https://sourceforge.net/projects/fortranxunit/  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: gbenchmark  
PackageHomePage: https://github.com/google/benchmark  
PackageLicenseDeclared: Apache-2.0  

PackageName: gmock  
PackageHomePage: https://github.com/google/googlemock  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: gtest  
PackageHomePage: https://github.com/google/googletest  
PackageLicenseDeclared: BSD-3-Clause  
