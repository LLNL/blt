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

.. _Recommends:

CMake Recommendations 
====================== 


This section includes several recommendations for how to wield CMake. Some of them are embodied in BLT, others are broader suggestions for using CMake.



* Disable in-source builds
* Avoid using glob to identify source files

This causes confusion with compile time and configure time states.

* Use arguments instead of options in CMake Macros and Functions
* Prefer explicit paths to tpls
* Emit configure error if explicitly identified tpl is not found or is incorrect version (don't warn, disable, and move on)

* Add headers as source files to targets as well. (BLT supports this)

This ensures they are properly included by the build system IDE generators, like Xcode or Eclipse.  


* Always support ‘make install’.

This allows cmake to do the 'right thing' based on CMAKE_INSTALL_PREFIX, and also helps support CPack with creating release packages. 
This is especially important for libraries. In addition to targets, header files require an explicit install command:

Example:

.. code:: cmake

  ##################################
  # Install Targets for example lib
  ##################################
  install(FILES ${example_headers} DESTINATION include)
  install(TARGETS example
    EXPORT example
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
