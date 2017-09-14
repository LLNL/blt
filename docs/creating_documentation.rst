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

Creating Documentation
======================

BLT provides macros to build documentation using `Sphinx <http://www.sphinx-doc.org/>`_
and `Doxygen <http://www.doxygen.org/>`_.


Sphinx is the documentation system used for the Python programming language (and many other projects).

Doxygen is  a widely used system that generates documentation from annotated source code.
Doxygen is heavily used for documenting  C++ software.

Sphinx and Doxygen are not built into BLT, so the ``sphinx-build`` and ``doxygen`` executables
must be available via a users ``PATH`` at configuration time, or explicitly specified using the
CMake variables ``SPHINX_EXECUTABLE`` and ``DOXYGEN_EXECUTABLE`` .


Here is an example of setting ``sphinx-build`` and ``doxygen`` paths in a host-config file:

.. code:: cmake
   
   set(SPHINX_EXECUTABLE "/usr/bin/sphinx-build" CACHE PATH "")

   set(DOXYGEN_EXECUTABLE "/usr/bin/doxygen" CACHE PATH "")


The ``calc_pi`` example provides examples of both Sphinx and Doxygen documentation. 


Calc Pi Sphinx Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Sphinx is a python package that depends on several other packages.  
It can be installed via spack, pip, anaconda, etc. 

``sphinx-build`` process a ``config.py`` file which includes a tree of reStructuredText files.
The Sphinx ``sphinx-quickstart`` utility helps you generate a new sphinx project, including
selecting common settings for the ``config.py``.


BLT provides a ``blt_add_sphinx_target()`` macro which, which will look for a ``conf.py`` file
in the current directory and add a command to build the Sphinx docs using this file to the ``docs``
CMake target.

.. admonition:: blt_add_sphinx_target
   :class: hint

   A macro to create a named sphinx target for user documentation.  
   Assumes there is a ``conf.py`` sphinx configuration file in the current directory.  
   This macro is active when BLT is configured with a valid ``SPHINX_EXECUTABLE`` path.


Here is an example of using ``blt_add_sphinx_target()`` in a CMakeLists.txt file:

.. literalinclude:: tutorial/calc_pi/docs/sphinx/CMakeLists.txt
   :language: rst


Here is the example reStructuredText file that contains documentation for the *calc_pi* example.

.. literalinclude:: tutorial/calc_pi/docs/sphinx/index.rst
   :language: rst
   

Calc Pi Doxygen Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Doxygen is a compiled executable that can be installed via spack, built-by-hand, etc. 

``doxygen`` processes a ``Doxyfile`` which specifies options, including where to look for
annotated source files.

BLT provides a ``blt_add_doxygen_target()`` macro which, which will look for a ``Doxyfile.in``
file in the current directory, configure this file to create a ``Doxyfile`` in the build directory,
and add a command to build the Doxygen docs using this file to the ``docs`` CMake target.

.. admonition:: blt_add_doxygen_target
   :class: hint

   A macro to create a named doxygen target for API documentation.  
   Assumes there is a ``Doxyfile.in`` doxygen configuration file in the current directory.  
   This macro is active when BLT is configured with a valid ``DOXYGEN_EXECUTABLE`` path.


Here is an example of using ``blt_add_doxygen_target()`` in a CMakeLists.txt file:

.. literalinclude:: tutorial/calc_pi/docs/doxygen/CMakeLists.txt
   :language: rst


Here is the example ``Doxyfile.in`` file that is configured by CMake and passed to ``doxygen``.

.. literalinclude:: tutorial/calc_pi/docs/doxygen/Doxyfile.in
   :language: rst



Building the Calc Pi Example Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Here is an example of building both the calc_pi Sphinx and Doxygen docs using the ``docs`` CMake target:

.. code:: console

   cd build-calc-pi
   make docs
   ...
   [ 50%] Building HTML documentation with Sphinx
   [ 50%] Built target calc_pi_sphinx
   [ 50%] Built target sphinx_docs
   [100%] Generating API documentation with Doxygen
   Searching for include files...
   Searching for example files...
   Searching for images...
   Searching for dot files...
   ...
   lookup cache used 3/65536 hits=3 misses=3
   finished...
   [100%] Built target calc_pi_doxygen
   [100%] Built target doxygen_docs
   [100%] Built target docs




After this, you can view the Sphinx docs at:

* ``build-calc-pi/docs/sphinx/html/index.html``

and the Doxygen docs at:

* ``build-calc-pi/docs/doxygen/html/index.html``

