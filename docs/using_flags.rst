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


Portable compiler flags
=========================

To ease with the development of code that is portable across different architectures
and compilers, BLT provides the ``blt_append_custom_compiler_flag()`` macro,
which allows users to easily place a compiler dependent flag into a CMake variable.

.. admonition:: blt_append_custom_compiler_flag
   :class: hint

   To use this macro, supply a cmake variable in which to append a flag (``FLAGS_VAR``), 
   and the appropriate flag for each of our supported compilers. 

   This macro currently supports the following compilers:

   * GNU
   * CLANG
   * XL (IBM compiler)
   * INTEL (Intel compiler)
   * MSVC (Microsoft Visual Studio)

Here is an example for setting the appropriate flag to treat warnings as errors:

.. code:: cmake

    blt_append_custom_compiler_flag(
      FLAGS_VAR BLT_WARNINGS_AS_ERRORS_FLAG
      DEFAULT  "-Werror"
      MSVC     "/WX"
      XL       "qhalt=w"
      )

Since values for ``GNU``, ``CLANG`` and ``INTEL`` are not supplied, 
they will get the default value (``-Werror``)
which is supplied by the macro's ``DEFAULT`` argument.

BLT also provides a simple macro to add compiler flags to a target.  
You can append the above compiler flag to an already defined executable, 
such as ``example_1`` with the following line:

.. code:: cmake

    blt_add_target_compile_flags(TO example_1
                                 FLAGS BLT_WARNINGS_AS_ERRORS_FLAG )

Here is another example to disable warnings about unknown OpenMP pragmas in the code:

.. code:: cmake

    # Flag for disabling warnings about omp pragmas in the code
    blt_append_custom_compiler_flag(
        FLAGS_VAR DISABLE_OMP_PRAGMA_WARNINGS_FLAG
        DEFAULT "-Wno-unknown-pragmas"
        XL      "-qignprag=omp"
        INTEL   "-diag-disable 3180"
        MSVC    "/wd4068"
        )

Note that GNU does not have a way to only disable warnings about openmp pragmas, 
so one must disable warnings about all unknown pragmas on this compiler.

