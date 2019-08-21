.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Utility Macros
==============


blt_append_custom_compiler_flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_append_custom_compiler_flag( 
                       FLAGS_VAR  flagsVar       (required)
                       DEFAULT    defaultFlag    (optional)
                       GNU        gnuFlag        (optional)
                       CLANG      clangFlag      (optional)
                       HCC        hccFlag        (optional)
                       INTEL      intelFlag      (optional)
                       XL         xlFlag         (optional)
                       MSVC       msvcFlag       (optional)
                       MSVC_INTEL msvcIntelFlag  (optional)
                       PGI        pgiFlag        (optional))

Appends compiler-specific flags to a given variable of flags

If a custom flag is given for the current compiler, we use that.
Otherwise, we will use the DEFAULT flag (if present).

If ENABLE_FORTRAN is On, any flagsVar with "fortran" (any capitalization)
in its name will pick the compiler family (GNU,CLANG, INTEL, etc) based on
the fortran compiler family type. This allows mixing C and Fortran compiler
families, e.g. using Intel fortran compilers with clang C compilers. 

When using the Intel toolchain within visual studio, we use the 
MSVC_INTEL flag, when provided, with a fallback to the MSVC flag.


blt_find_libraries
~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_find_libraries( FOUND_LIBS <FOUND_LIBS variable name>
                        NAMES      [libname1 [libname2 ...]]
                        REQUIRED   [TRUE (default) | FALSE ]
                        PATHS      [path1 [path2 ...]])

This command is used to find a list of libraries.

If the libraries are found the results are appended to the given FOUND_LIBS variable name.
NAMES lists the names of the libraries that will be searched for in the given PATHS.

If REQUIRED is set to TRUE, BLT will produce an error message if any of the
given libraries are not found.  The default value is TRUE.

PATH lists the paths in which to search for NAMES. No system paths will be searched.


blt_list_append
~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_list_append(TO       <list>
                    ELEMENTS [<element>...]
                    IF       <bool>)

Appends elements to a list if the specified bool evaluates to true.

This macro is essentially a wrapper around CMake's ``list(APPEND ...)``
command which allows inlining a conditional check within the same call
for clarity and convenience.

This macro requires specifying:

    * The target list to append to by passing TO <list>
    * A condition to check by passing IF <bool>
    * The list of elements to append by passing ELEMENTS [<element>...]

Note, the argument passed to the IF option has to be a single boolean value
and cannot be a boolean expression since CMake cannot evaluate those inline.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    set(mylist A B)
    blt_list_append( TO mylist ELEMENTS C IF ${ENABLE_C} )


blt_list_remove_duplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_list_remove_duplicates(TO <list>)

Removes duplicate elements from the given TO list.

This macro is essentially a wrapper around CMake's ``list(REMOVE_DUPLICATES ...)``
command but doesn't throw an error if the list is empty or not defined.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    set(mylist A B A)
    blt_list_remove_duplicates( TO mylist )

