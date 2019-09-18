.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Target Property Macros
======================


blt_add_target_compile_flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_target_compile_flags( TO    <target>
                                  FLAGS [FOO [BAR ...]])

Adds compiler flags to a target (library, executable or interface) by 
appending to the target's existing flags.

The TO argument (required) specifies a cmake target.

The FLAGS argument contains a list of compiler flags to add to the target. 

This macro will strip away leading and trailing whitespace from each flag.


blt_add_target_definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_target_definitions( TO <target>
                                TARGET_DEFINITIONS [FOO [BAR ...]])

Adds pre-processor definitions to the given target. This macro provides very
similar functionality to cmake's native "add_definitions" command, but,
it provides more fine-grained scoping for the compile definitions on a
per target basis. Given a list of definitions, e.g., FOO and BAR, this macro
adds compiler definitions to the compiler command for the given target, i.e.,
it will pass -DFOO and -DBAR.

The supplied target must be added via add_executable() or add_library() or
with the corresponding blt_add_executable() and blt_add_library() macros.

Note, the target definitions can either include or omit the "-D" characters. 
E.g. the following are all valid ways to add two compile definitions 
(A=1 and B) to target 'foo'.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_add_target_definitions(TO foo TARGET_DEFINITIONS A=1 B)
    blt_add_target_definitions(TO foo TARGET_DEFINITIONS -DA=1 -DB)
    blt_add_target_definitions(TO foo TARGET_DEFINITIONS "A=1;-DB")
    blt_add_target_definitions(TO foo TARGET_DEFINITIONS " " -DA=1;B)


blt_add_target_link_flags
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_target_link_flags( TO <target>
                               FLAGS [FOO [BAR ...]])

Adds linker flags to a target by appending to the target's existing flags.

The FLAGS argument expects a ; delimited list of linker flags to add to the target.

If `CUDA_LINK_WITH_NVCC` is set to ON, this macro will automatically convert
"-Wl,-rpath," to "-Xlinker -rpath -Xlinker ".

Note: In CMake versions prior to 3.13, this list is converted to a string internally
and any ; characters will be removed.


blt_print_target_properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_print_target_properties(TARGET <target>)

Prints out all properties of the given target.

The required target parameteter must either be a valid cmake target 
or was registered via blt_register_library.

Output is of the form for each property:
 | [<target> property] <property>: <value>


blt_set_target_folder
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_set_target_folder( TARGET <target>
                           FOLDER <folder>)

Sets the folder property of cmake target <target> to <folder>.

This feature is only available when blt's ENABLE_FOLDERS option is ON and 
in cmake generators that support folders (but is safe to call regardless
of the generator or value of ENABLE_FOLDERS).

Note: Do not use this macro on header-only (INTERFACE) library targets, since 
this will generate a cmake configuration error.

