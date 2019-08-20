.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Utility Macros
==============


General
-------


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


Git
---


blt_git
~~~~~~~

.. code-block:: cmake

    blt_git(SOURCE_DIR      <dir>
            GIT_COMMAND     <command>
            OUTPUT_VARIABLE <out>
            RETURN_CODE     <rc>
            [QUIET] )

Runs the supplied git command on the given Git repository.

This macro runs the user-supplied Git command, given by GIT_COMMAND, on the
given Git repository corresponding to SOURCE_DIR. The supplied GIT_COMMAND
is just a string consisting of the Git command and its arguments. The
resulting output is returned to the supplied CMake variable provided by
the OUTPUT_VARIABLE argument.

A return code for the Git command is returned to the caller via the CMake
variable provided with the RETURN_CODE argument. A non-zero return code
indicates that an error has occured.

Note, this macro assumes FindGit() was invoked and was successful. It relies
on the following variables set by FindGit():

    * Git_FOUND flag that indicates if git is found
    * GIT_EXECUTABLE points to the Git binary

If Git_FOUND is "false" this macro will throw a FATAL_ERROR message.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_git( SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
             GIT_COMMAND describe --tags master
             OUTPUT_VARIABLE axom_tag
             RETURN_CODE rc )
    if (NOT ${rc} EQUAL 0)
        message( FATAL_ERROR "blt_git failed!" )
    endif()


blt_is_git_repo
~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_is_git_repo(OUTPUT_STATE <state>
                    [SOURCE_DIR  <dir>] )

Checks if we are working with a valid Git repository.

This macro checks if the corresponding source directory is a valid Git repo.
Nominally, the corresponding source directory that is used is set to
${CMAKE_CURRENT_SOURCE_DIR}. A different source directory may be optionally
specified using the SOURCE_DIR argument.

The resulting state is stored in the CMake variable specified by the caller
using the OUTPUT_STATE parameter.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_is_git_repo( OUTTPUT_STATE is_git_repo )
    if ( ${is_git_repo} )
        message(STATUS "Pointing to a valid Git repo!")
    else()
        message(STATUS "Not a Git repo!")
    endif()


blt_git_tag
~~~~~~~~~~~

.. code-block:: cmake

    blt_git_tag( OUTPUT_TAG  <tag>
                 RETURN_CODE <rc>
                 [SOURCE_DIR <dir>]
                 [ON_BRANCH  <branch>] )

Returns the latest tag on a corresponding Git repository.

This macro gets the latest tag from a Git repository that can be specified
via the SOURCE_DIR argument. If SOURCE_DIR is not supplied, the macro will
use ${CMAKE_CURRENT_SOURCE_DIR}. By default the macro will return the latest
tag on the branch that is currently checked out. A particular branch may be
specified using the ON_BRANCH option.

The tag is stored in the CMake variable specified by the caller using the
the OUTPUT_TAG parameter.

A return code for the Git command is returned to the caller via the CMake
variable provided with the RETURN_CODE argument. A non-zero return code
indicates that an error has occured.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_git_tag( OUTPUT_TAG tag RETURN_CODE rc ON_BRANCH master )
    if ( NOT ${rc} EQUAL 0 )
        message( FATAL_ERROR "blt_git_tag failed!" )
    endif()
    message( STATUS "tag=${tag}" )


blt_git_branch
~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_git_branch( BRANCH_NAME <branch>
                    RETURN_CODE <rc>
                    [SOURCE_DIR <dir>] )

Returns the name of the active branch in the checkout space.

This macro gets the name of the current active branch in the checkout space
that can be specified using the SOURCE_DIR argument. If SOURCE_DIR is not
supplied by the caller, this macro will point to the checkout space
corresponding to ${CMAKE_CURRENT_SOURCE_DIR}.

A return code for the Git command is returned to the caller via the CMake
variable provided with the RETURN_CODE argument. A non-zero return code
indicates that an error has occured.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_git_branch( BRANCH_NAME active_branch RETURN_CODE rc )
    if ( NOT ${rc} EQUAL 0 )
        message( FATAL_ERROR "blt_git_tag failed!" )
    endif()
    message( STATUS "active_branch=${active_branch}" )


blt_git_hashcode
~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_git_hashcode( HASHCODE    <hc>
                      RETURN_CODE <rc>
                      [SOURCE_DIR <dir>]
                      [ON_BRANCH  <branch>])

Returns the SHA-1 hashcode at the tip of a branch.

This macro returns the SHA-1 hashcode at the tip of a branch that may be
specified with the ON_BRANCH argument. If the ON_BRANCH argument is not
supplied, the macro will return the SHA-1 hash at the tip of the current
branch. In addition, the caller may specify the target Git repository using
the SOURCE_DIR argument. Otherwise, if SOURCE_DIR is not specified, the
macro will use ${CMAKE_CURRENT_SOURCE_DIR}.

A return code for the Git command is returned to the caller via the CMake
variable provided with the RETURN_CODE argument. A non-zero return code
indicates that an error has occured.

.. code-block:: cmake
    :caption: **Example**
    :linenos:

    blt_git_hashcode( HASHCODE sha1 RETURN_CODE rc )
    if ( NOT ${rc} EQUAL 0 )
        message( FATAL_ERROR "blt_git_hashcode failed!" )
    endif()
    message( STATUS "sha1=${sha1}" )


Target Properties
-----------------


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
[<target> property] <property>: <value>


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

