.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Code Check Macros
==================

blt_add_code_checks
~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_code_checks( PREFIX               <Base name used for created targets>
                         SOURCES              [source1 [source2 ...]]
                         ASTYLE_CFG_FILE      <Path to AStyle config file>
                         CLANGFORMAT_CFG_FILE <Path to ClangFormat config file>
                         UNCRUSTIFY_CFG_FILE  <Path to Uncrustify config file>
                         CPPCHECK_FLAGS       <List of flags added to Cppcheck>)

This macro adds all enabled code check targets for the given SOURCES.

PREFIX
  Prefix used for the created code check build targets. For example:
  <PREFIX>_uncrustify_check

SOURCES
  Source list that the code checks will be ran on

ASTYLE_CFG_FILE
  Path to AStyle config file

CLANGFORMAT_CFG_FILE
  Path to ClangFormat config file

UNCRUSTIFY_CFG_FILE
  Path to Uncrustify config file

CPPCHECK_FLAGS
  List of flags added to Cppcheck

Sources are filtered based on file extensions for use in these code checks.  If you need
additional file extensions defined add them to BLT_C_FILE_EXTS and BLT_Fortran_FILE_EXTS.
Currently this macro only has code checks for C/C++ and simply filters out the Fortran files.

This macro supports code formatting with either AStyle, ClangFormat, or Uncrustify
(but not all at the same time) only if the following requirements are met:

- AStyle

  * ASTYLE_CFG_FILE is given
  * ASTYLE_EXECUTABLE is defined and found prior to calling this macro

- ClangFormat

  * CLANGFORMAT_CFG_FILE is given
  * CLANGFORMAT_EXECUTABLE is defined and found prior to calling this macro

- Uncrustify

  * UNCRUSTIFY_CFG_FILE is given
  * UNCRUSTIFY_EXECUTABLE is defined and found prior to calling this macro


Enabled code formatting checks produce a `check` build target that will test to see if you
are out of compliance with your code formatting and a `style` build target that will actually
modify your source files.  It also creates smaller child build targets that follow the pattern
`<PREFIX>_<astyle|clangformat|uncrustify>_<check|style>`.

This macro supports the following static analysis tools with their requirements:

- CppCheck

  * CPPCHECK_EXECUTABLE is defined and found prior to calling this macro
  * <optional> CPPCHECK_FLAGS added to the cppcheck command line before the sources

- Clang-Query

  * CLANGQUERY_EXECUTABLE is defined and found prior to calling this macro

These are added as children to the `check` build target and produce child build targets
that follow the pattern `<PREFIX>_<cppcheck|clangquery>_check`.

blt_add_clang_query_target
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_clang_query_target( NAME              <Created Target Name>
                                WORKING_DIRECTORY <Working Directory>
                                COMMENT           <Additional Comment for Target Invocation>
                                CHECKERS          <specifies a subset of checkers>
                                DIE_ON_MATCH      <TRUE | FALSE (default)>
                                SRC_FILES         [source1 [source2 ...]])

Creates a new build target for running clang-query.

NAME
  Name of created build target

WORKING_DIRECTORY
  Directory in which the clang-query command is run. Defaults to where macro is called.

COMMENT
  Comment prepended to the build target output

CHECKERS
  list of checkers to be run by created build target

DIE_ON_MATCH
  Causes build failure on first clang-query match. Defaults to FALSE.S

SRC_FILES
  Source list that clang-query will be ran on

Clang-query is a tool used for examining and matching the Clang AST. It is useful for enforcing
coding standards and rules on your source code.  A good primer on how to use clang-query can be
found `here <https://devblogs.microsoft.com/cppblog/exploring-clang-tooling-part-2-examining-the-clang-ast-with-clang-query/>`_.

Turning on DIE_ON_MATCH is useful if you're using this in CI to enforce rules about your code.

CHECKERS are the static analysis passes to specifically run on the target. The following checker options
can be given:

    * (no value)          : run all available static analysis checks found
    * (checker1:checker2) : run checker1 and checker2
    * (interpreter)       : run the clang-query interpeter to interactively develop queries


blt_add_cppcheck_target
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_cppcheck_target( NAME                <Created Target Name>
                             WORKING_DIRECTORY   <Working Directory>
                             PREPEND_FLAGS       <Additional flags for cppcheck>
                             APPEND_FLAGS        <Additional flags for cppcheck>
                             COMMENT             <Additional Comment for Target Invocation>
                             SRC_FILES           [source1 [source2 ...]] )

Creates a new build target for running cppcheck

NAME
  Name of created build target

WORKING_DIRECTORY
  Directory in which the clang-query command is run. Defaults to where macro is called.

PREPEND_FLAGS
  Additional flags added to the front of the cppcheck flags

APPEND_FLAGS
 Additional flags added to the end of the cppcheck flags

COMMENT
  Comment prepended to the build target output

SRC_FILES
  Source list that cppcheck will be ran on

Cppcheck is a static analysis tool for C/C++ code. More information about
Cppcheck can be found `here <http://cppcheck.sourceforge.net/>`_.


blt_add_astyle_target
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_astyle_target( NAME              <Created Target Name>
                           MODIFY_FILES      [TRUE | FALSE (default)]
                           CFG_FILE          <AStyle Configuration File> 
                           PREPEND_FLAGS     <Additional Flags to AStyle>
                           APPEND_FLAGS      <Additional Flags to AStyle>
                           COMMENT           <Additional Comment for Target Invocation>
                           WORKING_DIRECTORY <Working Directory>
                           SRC_FILES         [FILE1 [FILE2 ...]] )

Creates a new build target for running AStyle

NAME
  Name of created build target

MODIFY_FILES
  Modify the files in place. Defaults to FALSE.

CFG_FILE
  Path to AStyle config file

PREPEND_FLAGS
  Additional flags added to the front of the AStyle flags

APPEND_FLAGS
 Additional flags added to the end of the AStyle flags

COMMENT
  Comment prepended to the build target output

WORKING_DIRECTORY
  Directory in which the AStyle command is run. Defaults to where macro is called.

SRC_FILES
  Source list that AStyle will be ran on

AStyle is a Source Code Beautifier for C/C++ code. More information about
AStyle can be found `here <http://astyle.sourceforge.net/>`_.

When MODIFY_FILES is set to TRUE, modifies the files in place and adds the created build
target to the parent `style` build target.  Otherwise the files are not modified and the
created target is added to the parent `check` build target. This target will notify you
which files do not conform to your style guide.
.. Note::
  Setting MODIFY_FILES to FALSE is only supported in AStyle v2.05 or greater.


blt_add_clangformat_target
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_clangformat_target( NAME              <Created Target Name>
                                MODIFY_FILES      [TRUE | FALSE (default)]
                                CFG_FILE          <ClangFormat Configuration File> 
                                PREPEND_FLAGS     <Additional Flags to ClangFormat>
                                APPEND_FLAGS      <Additional Flags to ClangFormat>
                                COMMENT           <Additional Comment for Target Invocation>
                                WORKING_DIRECTORY <Working Directory>
                                SRC_FILES         [FILE1 [FILE2 ...]] )

Creates a new build target for running ClangFormat

NAME
  Name of created build target

MODIFY_FILES
  Modify the files in place. Defaults to FALSE.

CFG_FILE
  Path to ClangFormat config file

PREPEND_FLAGS
  Additional flags added to the front of the ClangFormat flags

APPEND_FLAGS
 Additional flags added to the end of the ClangFormat flags

COMMENT
  Comment prepended to the build target output

WORKING_DIRECTORY
  Directory in which the ClangFormat command is run. Defaults to where macro is called.

SRC_FILES
  Source list that ClangFormat will be ran on

ClangFormat is a Source Code Beautifier for C/C++ code. More information about
ClangFormat can be found `here <https://clang.llvm.org/docs/ClangFormat.html>`_.

When MODIFY_FILES is set to TRUE, modifies the files in place and adds the created build
target to the parent `style` build target.  Otherwise the files are not modified and the
created target is added to the parent `check` build target. This target will notify you
which files do not conform to your style guide.


blt_add_uncrustify_target
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_uncrustify_target( NAME              <Created Target Name>
                               MODIFY_FILES      [TRUE | FALSE (default)]
                               CFG_FILE          <Uncrustify Configuration File> 
                               PREPEND_FLAGS     <Additional Flags to Uncrustify>
                               APPEND_FLAGS      <Additional Flags to Uncrustify>
                               COMMENT           <Additional Comment for Target Invocation>
                               WORKING_DIRECTORY <Working Directory>
                               SRC_FILES         [source1 [source2 ...]] )

Creates a new build target for running Uncrustify

NAME
  Name of created build target

MODIFY_FILES
  Modify the files in place. Defaults to FALSE.

CFG_FILE
  Path to Uncrustify config file

PREPEND_FLAGS
  Additional flags added to the front of the Uncrustify flags

APPEND_FLAGS
 Additional flags added to the end of the Uncrustify flags

COMMENT
  Comment prepended to the build target output

WORKING_DIRECTORY
  Directory in which the Uncrustify command is run. Defaults to where macro is called.

SRC_FILES
  Source list that Uncrustify will be ran on

Uncrustify is a Source Code Beautifier for C/C++ code. More information about
Uncrustify can be found `here <http://uncrustify.sourceforge.net/>`_.

When MODIFY_FILES is set to TRUE, modifies the files in place and adds the created build
target to the parent `style` build target.  Otherwise the files are not modified and the
created target is added to the parent `check` build target. This target will notify you
which files do not conform to your style guide.
.. Note::
  Setting MODIFY_FILES to FALSE is only supported in Uncrustify v0.61 or greater.
