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

External Dependencies
=====================

More coming soon!

.. danger:: talk about expanding host config for mpi and cuda here.
.. danger:: talk about test_2 here, since it involves MPI.


Next, we provide paths for MPI:

.. literalinclude:: tutorial/host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 26-36
   :linenos:


Finally, we provide paths for CUDA:

.. literalinclude:: tutorial/host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake
   :language: cmake
   :lines: 37-43
   :linenos:


Here are example host-config files that use gcc 4.9.3 for LLNL's Surface and Quartz Clusters.

:download:`llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake <tutorial/host-configs/llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake>`


:download:`llnl-quartz-toss3-gcc@4.9.3.cmake <tutorial/host-configs/llnl-quartz-toss3-gcc@4.9.3.cmake>`

Now, we use the host-config file to configure a build of the ``calc_pi`` example:

.. code:: bash
    
    mkdir build-calc-pi-surface
    cd build-calc-pi-surface
    cmake -C llnl-surface-chaos_5_x86_64_ib-gcc@4.9.3.cmake  -DBLT_SOURCE_DIR={path_to_blt} ../calc_pi/
