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

Host-configs
============


To capture (and revision control) build options, third party library paths, etc we recommend using CMake's initial-cache file mechanism. This feature allows you to pass a file to CMake that provides variables to bootstrap the configure process. 

You can pass initial-cache files to cmake via the ``-C`` command line option.

.. code:: bash
    
    cmake -C config_file.cmake


We call these initial-cache files *host-config* files, since we typically create a file for each platform or specific hosts if necessary. 


These files use standard CMake commands. CMake *set* commands need to specify ``CACHE`` as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")

For this section of the tutorial, we create a host-config file that specifies CMake details for compilers, MPI, and CUDA support.


First we set the paths the compilers we want to use:

.. literalinclude:: tutorial/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 10-24
   :linenos:

Next, we provide paths for MPI:

.. literalinclude:: tutorial/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines:: 26-36
   :linenos:


Finally, we provide paths for CUDA:

.. literalinclude:: tutorial/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 37-43
   :linenos:


:download:`Here is the complete surface host-config file: <tutorial/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3>`.

Now, we use the host-config file to configure a build:

.. code:: bash
    
    mkdir build-debug
    cd build-debug
    cmake -C llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake ../example/
    


 



