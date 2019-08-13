.. # Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level COPYRIGHT file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

Code Health Macros
==================

blt_add_code_checks
~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_code_checks( PREFIX              <Base name used for created targets>
                         SOURCES             [source1 [source2 ...]]
                         UNCRUSTIFY_CFG_FILE <path to uncrustify config file>
                         ASTYLE_CFG_FILE     <path to astyle config file>)

This macro adds all enabled code check targets for the given SOURCES.

Sources are filtered based on file extensions for use in these code checks.  If you need
additional file extensions defined add them to BLT_C_FILE_EXTS and BLT_Fortran_FILE_EXTS.

PREFIX is used in the creation of all the underlying targets. For example:
<PREFIX>_uncrustify_check.

This macro supports formatting with either Uncrustify or AStyle (but not both at the same time)
using the following parameters:

* UNCRUSTIFY_CFG_FILE is the configuration file for Uncrustify. If UNCRUSTIFY_EXECUTABLE
  is defined, found, and UNCRUSTIFY_CFG_FILE is provided it will create both check and
  style function for the given C/C++ files.

* ASTYLE_CFG_FILE is the configuration file for AStyle. If ASTYLE_EXECUTABLE
  is defined, found, and ASTYLE_CFG_FILE is provided it will create both check and
  style function for the given C/C++ files.


blt_add_clang_query_target
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_clang_query_target( NAME              <Created Target Name>
                                WORKING_DIRECTORY <Working Directory>
                                COMMENT           <Additional Comment for Target Invocation>
                                CHECKERS          <If specified, requires a specific set of checkers>
                                DIE_ON_MATCH      <If true, matches stop the build>
                                SRC_FILES         [FILE1 [FILE2 ...]] )

Creates a new target with the given NAME for running clang_query over the given SRC_FILES.

COMMENT is prepended to the commented outputted by CMake.

WORKING_DIRECTORY is the directory that clang_query will be ran.  It defaults to the directory
where this macro is called.

DIE_ON_MATCH will make a match cause the build to fail, useful if you're using this in CI to enforce
rules about your code.

CHECKERS are the static analysis passes to specifically run on the target. The following checker options
can be given:

    * (no value)          : run all available static analysis checks found
    * (checker1:checker2) : run checker1 and checker2
    * (interpreter)       : run the clang-query interpeter to interactively develop queries

SRC_FILES is a list of source files that clang_query will be run on.


blt_add_cppcheck_target
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

    blt_add_cppcheck_target( NAME                <Created Target Name>
                             WORKING_DIRECTORY   <Working Directory>
                             PREPEND_FLAGS       <additional flags for cppcheck>
                             APPEND_FLAGS        <additional flags for cppcheck>
                             COMMENT             <Additional Comment for Target Invocation>
                             SRC_FILES           [FILE1 [FILE2 ...]] )

Creates a new target with the given NAME for running cppcheck over the given SRC_FILES

PREPEND_FLAGS are additional flags added to the front of the cppcheck flags.

APPEND_FLAGS are additional flags added to the end of the cppcheck flags.

COMMENT is prepended to the commented outputted by CMake.

WORKING_DIRECTORY is the directory that cppcheck will be ran.  It defaults to the directory
where this macro is called.

SRC_FILES is a list of source files that cppcheck will be run on.


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
                               SRC_FILES         [FILE1 [FILE2 ...]] )

Creates a new target with the given NAME for running uncrustify over the given SRC_FILES.

MODIFY_FILES, if set to TRUE, modifies the files in place and adds the created target to
the style target.  Otherwise the files are not modified and the created target is added
to the check target.
Note: Setting MODIFY_FILES to FALSE is only supported in Uncrustify v0.61 or greater.

CFG_FILE defines the uncrustify settings file.

PREPEND_FLAGS are additional flags added to the front of the uncrustify flags.

APPEND_FLAGS are additional flags added to the end of the uncrustify flags.

COMMENT is prepended to CMake's output for this target.

WORKING_DIRECTORY is the directory in which uncrustify will be run.  It defaults 
to the directory where this macro is called.

SRC_FILES is a list of source files to style/check with uncrustify.


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

Creates a new target with the given NAME for running astyle over the given SRC_FILES.

MODIFY_FILES, if set to TRUE, modifies the files in place and adds the created target to
the style target. Otherwise the files are not modified and the created target is added
to the check target. Note: Setting MODIFY_FILES to FALSE is only supported in AStyle v2.05 or greater.

CFG_FILE defines the astyle settings file.

PREPEND_FLAGS are additional flags added to the front of the astyle flags.

APPEND_FLAGS are additional flags added to the end of the astyle flags.

COMMENT is prepended to CMake's output for this target.

WORKING_DIRECTORY is the directory in which astyle will be run. It defaults to 
the directory where this macro is called.

SRC_FILES is a list of source files to style/check with astyle.
