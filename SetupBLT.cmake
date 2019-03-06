# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

if (NOT BLT_LOADED)
    set(BLT_VERSION "0.2.0" CACHE STRING "")
    mark_as_advanced(BLT_VERSION)
    message(STATUS "BLT Version: ${BLT_VERSION}")

    set(BLT_LOADED True)
    mark_as_advanced(BLT_LOADED)

    set( BLT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE )

    # if an explicit build dir was not specified, set a default.
    if( NOT BLT_BUILD_DIR )
        set( BLT_BUILD_DIR ${PROJECT_BINARY_DIR}/blt CACHE PATH "" FORCE )
    endif()

    ################################
    # Prevent in-source builds
    ################################
    # Fail if someone tries to config an in-source build.
    if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
        message(FATAL_ERROR "In-source builds are not supported. Please remove the "
                            "CMakeFiles directory and CMakeCache.txt from the 'src' "
                            "dir and configure an out-of-source build in another "
                            "directory.")
    endif()

    #################################
    # Show CMake info right out of the gate
    ################################
    message(STATUS "CMake Version: ${CMAKE_VERSION}")

    if(${CMAKE_VERSION} VERSION_LESS 3.8.0)
        message("*************************************")
        message("* Unsupported version of CMake detected.")
        message("* BLT requires CMake 3.8 or above.")
        message("* Some BLT features may not work.")
        message("*************************************")
    endif()

    message(STATUS "CMake Executable: ${CMAKE_COMMAND}")


    ################################
    # Setup build options and their default values
    ################################
    include(${BLT_ROOT_DIR}/cmake/BLTOptions.cmake)

    ################################
    # CMP0057 New is required by blt_setup_target()
    # in order to support the IN_LIST if() operator
    ################################
    if(POLICY CMP0057)
        cmake_policy(SET CMP0057 NEW)
    endif()

    ################################
    # Invoke CMake Fortran setup
    # if ENABLE_FORTRAN == ON
    ################################
    if(ENABLE_FORTRAN)
        enable_language(Fortran)
    endif()

    ################################
    # Enable ctest support
    ################################
    if(ENABLE_TESTS)
        enable_testing()
    endif()

    ################################
    # Enable cmake generator folder feature
    ################################
    if(ENABLE_FOLDERS)
        set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    endif()


    ################################
    # Enable cmake compilation database feature
    ################################
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    ################################
    # Macros
    ################################
    include(${BLT_ROOT_DIR}/cmake/BLTMacros.cmake)

    ################################
    # Standard TPL support
    ################################
    include(${BLT_ROOT_DIR}/cmake/thirdparty/SetupThirdParty.cmake)

    ################################
    # Git related Macros
    ################################
    if (Git_FOUND)
        include(${BLT_ROOT_DIR}/cmake/BLTGitMacros.cmake)
    endif()

    ################################
    # Setup docs targets
    ################################
    include(${BLT_ROOT_DIR}/cmake/SetupDocs.cmake)

    ################################
    # Setup source checks
    ################################
    include(${BLT_ROOT_DIR}/cmake/SetupCodeChecks.cmake)

    ################################
    # Standard Build Layout
    ################################

    # Set the path where all the libraries will be stored
    set(LIBRARY_OUTPUT_PATH
        ${PROJECT_BINARY_DIR}/lib
        CACHE PATH
        "Directory where compiled libraries will go in the build tree")

    # Set the path where all the installed executables will go
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin
        CACHE PATH
        "Directory where executables will go in the build tree")

    # Set the path were all test executables will go
    set(TEST_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/tests
        CACHE PATH
        "Directory where test executables will go in the build tree")

    # Set the path were all example test executables will go
    set(EXAMPLE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/examples
        CACHE PATH
        "Directory where example executables will go in the build tree")

    # Set the Fortran module directory
    set(CMAKE_Fortran_MODULE_DIRECTORY
        ${PROJECT_BINARY_DIR}/lib/fortran
        CACHE PATH
        "Directory where all Fortran modules will go in the build tree")

    # Mark as advanced
    mark_as_advanced(
       LIBRARY_OUTPUT_PATH
       CMAKE_RUNTIME_OUTPUT_DIRECTORY
       CMAKE_Fortran_MODULE_DIRECTORY)

    ################################
    # Global variables needed by BLT
    #
    ################################
    set(BLT_C_FILE_EXTS ".cpp" ".hpp" ".cxx" ".hxx" ".c" ".h" ".cc" ".hh" ".inl" ".cu"
               CACHE LIST "List of known file extensions used for C/CXX sources")
    set(BLT_Fortran_FILE_EXTS ".F" ".f" ".f90" ".F90"
               CACHE LIST "List of known file extensions used for Fortran sources")


    ################################
    # Setup compiler options
    # (must be included after HEADER_INCLUDES_DIRECTORY and MPI variables are set)
    ################################
    include(${BLT_ROOT_DIR}/cmake/SetupCompilerOptions.cmake)

    ################################
    # Setup code metrics -
    # profiling, code coverage, etc.
    # (must be included after SetupCompilerOptions)
    ################################
    include(${BLT_ROOT_DIR}/cmake/SetupCodeMetrics.cmake)

    ################################
    # builtin third party libs used by BLT
    ################################
    add_subdirectory(${BLT_ROOT_DIR}/thirdparty_builtin ${BLT_BUILD_DIR}/thirdparty_builtin)

    ################################
    # BLT smoke tests
    ################################
    if(ENABLE_TESTS)
        add_subdirectory(${BLT_ROOT_DIR}/tests/smoke ${BLT_BUILD_DIR}/tests/smoke)
    endif()

endif() # only load BLT once!
