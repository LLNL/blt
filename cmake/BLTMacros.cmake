###############################################################################
# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-725085
#
# All rights reserved.
#
# This file is part of BLT.
#
# For additional details, please also read BLT/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


include(${BLT_ROOT_DIR}/cmake/BLTPrivateMacros.cmake)

##------------------------------------------------------------------------------
## blt_list_append( TO <list> ELEMENTS [ <element>...] IF <bool> )
##
## Appends elements to a list if the specified bool evaluates to true.
##
## This macro is essentially a wrapper around CMake's `list(APPEND ...)`
## command which allows inlining a conditional check within the same call
## for clarity and convenience.
##
## This macro requires specifying:
##   (1) The target list to append to by passing TO <list>
##   (2) A condition to check by passing IF <bool>
##   (3) The list of elements to append by passing ELEMENTS [<element>...]
##
## Note, the argument passed to the IF option has to be a single boolean value
## and cannot be a boolean expression since CMake cannot evaluate those inline.
##
## Usage Example:
##
##  set(mylist A B)
##  blt_list_append( TO mylist ELEMENTS C IF ${ENABLE_C} )
##
##------------------------------------------------------------------------------
macro(blt_list_append)

    set(options)
    set(singleValueArgs TO IF)
    set(multiValueArgs ELEMENTS )

    # parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

     # sanity checks
    if( NOT DEFINED arg_TO )
        message(FATAL_ERROR "blt_list_append() requires a TO <list> argument")
    endif()

    if ( NOT DEFINED arg_ELEMENTS )
         message(FATAL_ERROR "blt_list_append() requires ELEMENTS to be specified" )
    endif()

    # append if
    if ( ${arg_IF} )
        list( APPEND ${arg_TO} ${arg_ELEMENTS} )
    endif()

endmacro(blt_list_append)

##------------------------------------------------------------------------------
## blt_add_target_definitions(TO <target> TARGET_DEFINITIONS [FOO [BAR ...]])
##
## Adds pre-processor definitions to the given target.
##
## Adds pre-processor definitions to a particular. This macro provides very
## similar functionality to cmake's native "add_definitions" command, but,
## it provides more fine-grained scoping for the compile definitions on a
## per target basis. Given a list of definitions, e.g., FOO and BAR, this macro
## adds compiler definitions to the compiler command for the given target, i.e.,
## it will pass -DFOO and -DBAR.
##
## The supplied target must be added via add_executable() or add_library() or
## with the corresponding blt_add_executable() and blt_add_library() macros.
##
## Note, the target definitions can either include or omit the "-D" characters. 
## E.g. the following are all valid ways to add two compile definitions 
## (A=1 and B) to target 'foo'
##
##   blt_add_target_definitions(TO foo TARGET_DEFINITIONS A=1 B)
##   blt_add_target_definitions(TO foo TARGET_DEFINITIONS -DA=1 -DB)
##   blt_add_target_definitions(TO foo TARGET_DEFINITIONS "A=1;-DB")
##   blt_add_target_definitions(TO foo TARGET_DEFINITIONS " " -DA=1;B")
##------------------------------------------------------------------------------
macro(blt_add_target_definitions)

    set(options)
    set(singleValueArgs TO)
    set(multiValueArgs TARGET_DEFINITIONS)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    ## check that the passed in parameter TO is actually a target
    if(NOT TARGET ${arg_TO})
        message(FATAL_ERROR "Target ${arg_TO} passed to blt_add_target_definitions is not a valid cmake target")    
    endif()

    ## only add the flag if it is not empty
    string(STRIP "${arg_TARGET_DEFINITIONS}" _strippedDefs)
    if(NOT "${_strippedDefs}" STREQUAL "")
        get_property(_targetType TARGET ${arg_TO} PROPERTY TYPE)
        if(${_targetType} STREQUAL "INTERFACE_LIBRARY")
            target_compile_definitions(${arg_TO} INTERFACE ${_strippedDefs})
        else()
            target_compile_definitions(${arg_TO} PUBLIC ${_strippedDefs})
        endif()        
    endif()

    unset(_targetType)
    unset(_strippedDefs)

endmacro(blt_add_target_definitions)


##------------------------------------------------------------------------------
## blt_add_target_compile_flags (TO <target> FLAGS [FOO [BAR ...]])
##
## Adds compiler flags to a target (library, executable or interface) by 
## appending to the target's existing flags.
##
## The TO argument (required) specifies a cmake target.
##
## The FLAGS argument contains a list of compiler flags to add to the target. 
## This macro will strip away leading and trailing whitespace from each flag.
##------------------------------------------------------------------------------
macro(blt_add_target_compile_flags)

    set(options)
    set(singleValuedArgs TO)
    set(multiValuedArgs FLAGS)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    ## check that the passed in parameter TO is actually a target
    if(NOT TARGET ${arg_TO})
        message(FATAL_ERROR "Target ${arg_TO} passed to blt_add_target_compile_flags is not a valid cmake target")    
    endif()

    ## only add the flag if it is not empty
    string(STRIP "${arg_FLAGS}" _strippedFlags)
    if(NOT "${_strippedFlags}" STREQUAL "")
        get_property(_targetType TARGET ${arg_TO} PROPERTY TYPE)
        if(${_targetType} STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${arg_TO} INTERFACE ${_strippedFlags})
        else()
            target_compile_options(${arg_TO} PUBLIC ${_strippedFlags})
        endif()        
    endif()

    unset(_targetType)
    unset(_strippedFlags)

endmacro(blt_add_target_compile_flags)


##------------------------------------------------------------------------------
## blt_set_target_folder (TARGET <target> FOLDER <folder>)
##
## Sets the folder property of cmake target <target> to <folder>.
##
## This feature is only available when blt's ENABLE_FOLDERS option is ON and 
## in cmake generators that support folders (but is safe to call regardless
## of the generator or value of ENABLE_FOLDERS).
##
## Note: Do not use this macro on header-only (INTERFACE) library targets, since 
## this will generate a cmake configuration error.
##------------------------------------------------------------------------------
macro(blt_set_target_folder)

    set(options)
    set(singleValuedArgs TARGET FOLDER)
    set(multiValuedArgs)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    ## check for required arguments
    if(NOT DEFINED arg_TARGET)
        message(FATAL_ERROR "TARGET is a required parameter for blt_set_target_folder macro")
    endif()

    if(NOT TARGET ${arg_TARGET})
        message(FATAL_ERROR "Target ${arg_TARGET} passed to blt_set_target_folder is not a valid cmake target")
    endif()

    if(NOT DEFINED arg_FOLDER)
        message(FATAL_ERROR "FOLDER is a required parameter for blt_set_target_folder macro")
    endif()

    ## set the folder property for this target
    if(ENABLE_FOLDERS AND NOT "${arg_FOLDER}" STREQUAL "")
        set_property(TARGET ${arg_TARGET} PROPERTY FOLDER "${arg_FOLDER}")
    endif()

endmacro(blt_set_target_folder)


##------------------------------------------------------------------------------
## blt_add_target_link_flags (TO <target> FLAGS [FOO [BAR ...]])
##
## Adds linker flags to a target by appending to the target's existing flags.
##------------------------------------------------------------------------------
macro(blt_add_target_link_flags)

    set(options)
    set(singleValuedArgs TO)
    set(multiValuedArgs FLAGS)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    if(NOT "${arg_FLAGS}" STREQUAL "")
        # get prev flags
        get_target_property(_LINK_FLAGS ${arg_TO} LINK_FLAGS)
        if(NOT _LINK_FLAGS)
            set(_LINK_FLAGS "")
        endif()
        # append new flag
        set(_LINK_FLAGS "${arg_FLAGS} ${_LINK_FLAGS}")
        set_target_properties(${arg_TO}
                              PROPERTIES LINK_FLAGS "${_LINK_FLAGS}" )
    endif()

endmacro(blt_add_target_link_flags)

##------------------------------------------------------------------------------
## blt_register_library( NAME <libname>
##                       DEPENDS_ON [dep1 [dep2 ...]]
##                       INCLUDES [include1 [include2 ...]]
##                       TREAT_INCLUDES_AS_SYSTEM [ON|OFF]
##                       FORTRAN_MODULES [ path1 [ path2 ..]]
##                       LIBRARIES [lib1 [lib2 ...]]
##                       COMPILE_FLAGS [ flag1 [ flag2 ..]]
##                       LINK_FLAGS [ flag1 [ flag2 ..]]
##                       DEFINES [def1 [def2 ...]] )
##
## Registers a library to the project to ease use in other blt macro calls.
##
## Stores information about a library in a specific way that is easily recalled
## in other macros.  For example, after registering gtest, you can add gtest to
## the DEPENDS_ON in your blt_add_executable call and it will add the INCLUDES
## and LIBRARIES to that executable.
##
## TREAT_INCLUDES_AS_SYSTEM informs the compiler to treat this library's include paths
## as system headers.  Only some compilers support this. This is useful if the headers
## generate warnings you want to not have them reported in your build. This defaults
## to OFF.
##
## This does not actually build the library.  This is strictly to ease use after
## discovering it on your system or building it yourself inside your project.
##
## Output variables (name = "foo"):
##  BLT_FOO_IS_REGISTERED_LIBRARY
##  BLT_FOO_DEPENDS_ON
##  BLT_FOO_INCLUDES
##  BLT_FOO_TREAT_INCLUDES_AS_SYSTEM
##  BLT_FOO_FORTRAN_MODULES
##  BLT_FOO_LIBRARIES
##  BLT_FOO_COMPILE_FLAGS
##  BLT_FOO_LINK_FLAGS
##  BLT_FOO_DEFINES
##------------------------------------------------------------------------------
macro(blt_register_library)

    set(singleValueArgs NAME TREAT_INCLUDES_AS_SYSTEM)
    set(multiValueArgs INCLUDES 
                       DEPENDS_ON
                       FORTRAN_MODULES
                       LIBRARIES
                       COMPILE_FLAGS
                       LINK_FLAGS
                       DEFINES )

    ## parse the arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    string(TOUPPER ${arg_NAME} uppercase_name)

    set(BLT_${uppercase_name}_IS_REGISTERED_LIBRARY TRUE CACHE BOOL "" FORCE)

    if( arg_DEPENDS_ON )
        set(BLT_${uppercase_name}_DEPENDS_ON ${arg_DEPENDS_ON} CACHE LIST "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_DEPENDS_ON)
    endif()

    if( arg_INCLUDES )
        set(BLT_${uppercase_name}_INCLUDES ${arg_INCLUDES} CACHE LIST "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_INCLUDES)
    endif()

    if( ${arg_TREAT_INCLUDES_AS_SYSTEM} )
        set(BLT_${uppercase_name}_TREAT_INCLUDES_AS_SYSTEM ON CACHE BOOL "" FORCE)
    else()
        set(BLT_${uppercase_name}_TREAT_INCLUDES_AS_SYSTEM OFF CACHE BOOL "" FORCE)
    endif()
    mark_as_advanced(BLT_${uppercase_name}_TREAT_INCLUDES_AS_SYSTEM)

    if( ENABLE_FORTRAN AND arg_FORTRAN_MODULES )
        set(BLT_${uppercase_name}_FORTRAN_MODULES ${arg_INCLUDES} CACHE LIST "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_FORTRAN_MODULES)
    endif()

    if( arg_LIBRARIES )
        set(BLT_${uppercase_name}_LIBRARIES ${arg_LIBRARIES} CACHE LIST "" FORCE)
    else()
        # This prevents cmake from falling back on adding -l<library name>
        # to the command line for BLT registered libraries which are not
        # actual CMake targets
        set(BLT_${uppercase_name}_LIBRARIES "BLT_NO_LIBRARIES" CACHE STRING "" FORCE)
    endif()
    mark_as_advanced(BLT_${uppercase_name}_LIBRARIES)

    if( arg_COMPILE_FLAGS )
        set(BLT_${uppercase_name}_COMPILE_FLAGS "${arg_COMPILE_FLAGS}" CACHE STRING "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_COMPILE_FLAGS)
    endif()

    if( arg_LINK_FLAGS )
        set(BLT_${uppercase_name}_LINK_FLAGS "${arg_LINK_FLAGS}" CACHE STRING "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_LINK_FLAGS)
    endif()

    if( arg_DEFINES )
        set(BLT_${uppercase_name}_DEFINES ${arg_DEFINES} CACHE LIST "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_DEFINES)
    endif()

endmacro(blt_register_library)


##------------------------------------------------------------------------------
## blt_add_library( NAME <libname>
##                  SOURCES [source1 [source2 ...]]
##                  HEADERS [header1 [header2 ...]]
##                  INCLUDES [dir1 [dir2 ...]]
##                  DEFINES [define1 [define2 ...]]
##                  DEPENDS_ON [dep1 ...] 
##                  OUTPUT_NAME [name]
##                  OUTPUT_DIR [dir]
##                  SHARED [TRUE | FALSE]
##                  CLEAR_PREFIX [TRUE | FALSE]
##                  FOLDER [name]
##                 )
##
## Adds a library target, called <libname>, to be built from the given sources.
## This macro uses the BUILD_SHARED_LIBS, which is defaulted to OFF, to determine
## whether the library will be build as shared or static. The optional boolean
## SHARED argument can be used to override this choice.
##
## The INCLUDES argument allows you to define what include directories are
## needed by any target that is dependent on this library.  These will
## be inherited by CMake's target dependency rules.
##
## The DEFINES argument allows you to add needed compiler definitions that are
## needed by any target that is dependent on this library.  These will
## be inherited by CMake's target dependency rules.
##
## If given a DEPENDS_ON argument, it will add the necessary includes and 
## libraries if they are already registered with blt_register_library.  If 
## not it will add them as a CMake target dependency.
##
## In addition, this macro will add the associated dependencies to the given
## library target. Specifically, it will add the dependency for the CMake target
## and for copying the headers for that target as well.
##
## The OUTPUT_DIR is used to control the build output directory of this 
## library. This is used to overwrite the default lib directory.
##
## OUTPUT_NAME is the name of the output file; the default is NAME.
## It's useful when multiple libraries with the same name need to be created
## by different targets. NAME is the target name, OUTPUT_NAME is the library name.
##
## CLEAR_PREFIX allows you to remove the automatically appended "lib" prefix
## from your built library.  The created library will be foo.a instead of libfoo.a.
##
## FOLDER is an optional keyword to organize the target into a folder in an IDE.
## This is available when ENABLE_FOLDERS is ON and when the cmake generator
## supports this feature and will otherwise be ignored. 
## Note: Do not use with header-only (INTERFACE)libraries, as this will generate 
## a cmake configuration error.
##------------------------------------------------------------------------------
macro(blt_add_library)

    set(options)
    set(singleValueArgs NAME OUTPUT_NAME OUTPUT_DIR HEADERS_OUTPUT_SUBDIR SHARED CLEAR_PREFIX FOLDER)
    set(multiValueArgs SOURCES HEADERS INCLUDES DEFINES DEPENDS_ON)

    # parse the arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # sanity checks
    if( "${arg_NAME}" STREQUAL "" )
        message(FATAL_ERROR "blt_add_library() must be called with argument NAME <name>")
    endif()

    if (NOT arg_SOURCES AND NOT arg_HEADERS )
        message(FATAL_ERROR "blt_add_library(NAME ${arg_NAME} ...) called with no given sources or headers")
    endif()

    # Determine whether to build as a shared library. Default to global variable unless
    # SHARED parameter is specified
    set(_build_shared_library ${BUILD_SHARED_LIBS})
    if( DEFINED arg_SHARED )
        set(_build_shared_library ${arg_SHARED})
    endif()

    if ( arg_SOURCES )
        #
        #  CUDA support
        #
        list(FIND arg_DEPENDS_ON "cuda" check_for_cuda)
        if ( ${check_for_cuda} GREATER -1 AND NOT ENABLE_CLANG_CUDA)

            blt_setup_cuda_source_properties(BUILD_TARGET ${arg_NAME}
                                             TARGET_SOURCES ${arg_SOURCES})

        endif()
        if ( ${_build_shared_library} )
            add_library( ${arg_NAME} SHARED ${arg_SOURCES} ${arg_HEADERS} )
        else()
            add_library( ${arg_NAME} STATIC ${arg_SOURCES} ${arg_HEADERS} )
        endif()
        if ( ${check_for_cuda} GREATER -1 AND NOT ENABLE_CLANG_CUDA)
           set_target_properties( ${arg_NAME} PROPERTIES LANGUAGE CUDA)
           if ( ${_build_shared_library})
               set_target_properties( ${arg_NAME} PROPERTIES CMAKE_CUDA_CREATE_STATIC_LIBRARY ON)
           else()
               set_target_properties( ${arg_NAME} PROPERTIES CMAKE_CUDA_CREATE_STATIC_LIBRARY OFF)
           endif()
           if (CUDA_SEPARABLE_COMPILATION)
              set_target_properties( ${arg_NAME} PROPERTIES
                                                 CUDA_SEPARABLE_COMPILATION ON)
           endif()
        endif()
    else()
        #
        #  Header-only library support
        #
        foreach (_file ${arg_HEADERS})
            get_filename_component(_name ${_file} NAME)
            get_filename_component(_absolute ${_file} ABSOLUTE)

            # Determine build location of headers
            set(_build_headers ${_build_headers} ${_absolute})

            # Determine install location of headers
            set(_install_headers ${_install_headers} include/${arg_HEADERS_OUTPUT_SUBDIR}/${_name})
        endforeach()

        #Note: This only works if both libraries are handled in the same directory,
        #  otherwise just don't include non-header files in your source list.
        set_source_files_properties(${_build_headers} PROPERTIES HEADER_FILE_ONLY ON)

        add_library( ${arg_NAME} INTERFACE )
        target_sources( ${arg_NAME} INTERFACE
                        $<BUILD_INTERFACE:${_build_headers}>)
    endif()

    # Clear value of _have_fortran from previous calls
    set(_have_fortran False)

    # Must tell fortran where to look for modules
    # CMAKE_Fortran_MODULE_DIRECTORY is the location of generated modules
    foreach (_file ${arg_SOURCES})
        get_source_file_property(_lang ${_file} LANGUAGE)
        if(_lang STREQUAL Fortran)
            set(_have_fortran TRUE)
        endif()
    endforeach()
    if(_have_fortran)
        target_include_directories(${arg_NAME} PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
    endif()

    blt_setup_target( NAME ${arg_NAME}
                      DEPENDS_ON ${arg_DEPENDS_ON} )

    if ( arg_INCLUDES )
        target_include_directories(${arg_NAME} PUBLIC ${arg_INCLUDES})
    endif()

    if ( arg_DEFINES )
        target_compile_definitions(${arg_NAME} PUBLIC ${arg_DEFINES})
    endif()

    if ( arg_OUTPUT_DIR )
        set_target_properties(${arg_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${arg_OUTPUT_DIR} )
    endif()

    if ( arg_OUTPUT_NAME )
        set_target_properties(${arg_NAME} PROPERTIES
            OUTPUT_NAME ${arg_OUTPUT_NAME} )
    endif()

    if ( arg_CLEAR_PREFIX )
        set_target_properties(${arg_NAME} PROPERTIES
            PREFIX "" )
    endif()

    # Handle optional FOLDER keyword for this target
    if(ENABLE_FOLDERS AND DEFINED arg_FOLDER)
        blt_set_target_folder(TARGET ${arg_NAME} FOLDER "${arg_FOLDER}")
    endif()

    blt_update_project_sources( TARGET_SOURCES ${arg_SOURCES} ${arg_HEADERS})

endmacro(blt_add_library)

##------------------------------------------------------------------------------
## blt_add_executable( NAME <name>
##                     SOURCES [source1 [source2 ...]]
##                     INCLUDES [dir1 [dir2 ...]]
##                     DEFINES [define1 [define2 ...]]
##                     DEPENDS_ON [dep1 [dep2 ...]]
##                     OUTPUT_DIR [dir]
##                     FOLDER [name])
##
## Adds an executable target, called <name>.
##
## The INCLUDES argument allows you to define what include directories are
## needed to compile this executable.
##
## The DEFINES argument allows you to add needed compiler definitions that are
## needed to compile this executable.
##
## If given a DEPENDS_ON argument, it will add the necessary includes and 
## libraries if they are already registered with blt_register_library.  If
## not it will add them as a cmake target dependency.
##
## The OUTPUT_DIR is used to control the build output directory of this 
## executable. This is used to overwrite the default bin directory.
##
## If the first entry in SOURCES is a Fortran source file, the fortran linker 
## is used. (via setting the CMake target property LINKER_LANGUAGE to Fortran )
##
## FOLDER is an optional keyword to organize the target into a folder in an IDE.
## This is available when ENABLE_FOLDERS is ON and when using a cmake generator
## that supports this feature and will otherwise be ignored.
##------------------------------------------------------------------------------
macro(blt_add_executable)

    set(options )
    set(singleValueArgs NAME OUTPUT_DIR FOLDER)
    set(multiValueArgs SOURCES INCLUDES DEFINES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # sanity checks
    if( "${arg_NAME}" STREQUAL "" )
        message(FATAL_ERROR "blt_add_executable() must be called with argument NAME <name>")
    endif()

    if (NOT arg_SOURCES )
        message(FATAL_ERROR "blt_add_executable(NAME ${arg_NAME} ...) given with no sources")
    endif()

    #
    #  cuda support
    #
    list(FIND arg_DEPENDS_ON "cuda" check_for_cuda)
    if ( ${check_for_cuda} GREATER -1 AND NOT ENABLE_CLANG_CUDA)
        blt_setup_cuda_source_properties(BUILD_TARGET ${arg_NAME}
                                         TARGET_SOURCES ${arg_SOURCES})
    endif()
    add_executable( ${arg_NAME} ${arg_SOURCES} )
    if ( ${check_for_cuda} GREATER -1 AND NOT ENABLE_CLANG_CUDA)
        if (CUDA_SEPARABLE_COMPILATION)
           set_target_properties( ${arg_NAME} PROPERTIES
                                              CUDA_SEPARABLE_COMPILATION ON)
        endif()
        if (CUDA_LINK_WITH_NVCC) 
            set_target_properties( ${arg_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
        endif()
    endif()
    list(FIND arg_DEPENDS_ON "cuda_runtime" check_for_cuda_rt)
    if ( ${check_for_cuda_rt} GREATER -1 AND NOT ENABLE_CLANG_CUDA)
        if (CUDA_LINK_WITH_NVCC) 
            set_target_properties( ${arg_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
        endif()
    endif()
    
    # CMake wants to load with C++ if any of the libraries are C++.
    # Force to load with Fortran if the first file is Fortran.
    list(GET arg_SOURCES 0 _first)
    get_source_file_property(_lang ${_first} LANGUAGE)
    if(_lang STREQUAL Fortran)
        if (NOT CUDA_LINK_WITH_NVCC)
            set_target_properties( ${arg_NAME} PROPERTIES LINKER_LANGUAGE Fortran )
        endif()
        target_include_directories(${arg_NAME} PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
    endif()
       
    blt_setup_target(NAME ${arg_NAME}
                     DEPENDS_ON ${arg_DEPENDS_ON} )

    if ( arg_INCLUDES )
        target_include_directories(${arg_NAME} PUBLIC ${arg_INCLUDES})
    endif()

    if ( arg_DEFINES )
        target_compile_definitions(${arg_NAME} PUBLIC ${arg_DEFINES})
    endif()

    # when using shared libs on windows, all runtime targets
    # (dlls and exes) must live in the same dir
    # so we do not set runtime_output_dir in this case
    if ( arg_OUTPUT_DIR AND NOT (WIN32 AND BUILD_SHARED_LIBS) )
           set_target_properties(${arg_NAME} PROPERTIES
           RUNTIME_OUTPUT_DIRECTORY ${arg_OUTPUT_DIR} )
    endif()

    # Handle optional FOLDER keyword for this target
    if(ENABLE_FOLDERS AND DEFINED arg_FOLDER)
        blt_set_target_folder(TARGET ${arg_NAME} FOLDER "${arg_FOLDER}")
    endif()

    blt_update_project_sources( TARGET_SOURCES ${arg_SOURCES} )

endmacro(blt_add_executable)


##------------------------------------------------------------------------------
## blt_add_test( NAME [name] COMMAND [command] NUM_MPI_TASKS [n] )
##
## Adds a CMake test to the project.
##
## NAME is used for the name that CTest reports with.
##
## COMMAND is the command line that will be used to run the test. This will
## have the RUNTIME_OUTPUT_DIRECTORY prepended to it to fully qualify the path.
##
## NUM_MPI_TASKS indicates this is an MPI test and how many tasks to use. The
## command line will use MPIEXEC, MPIEXEC_NUMPROC_FLAG, and BLT_MPI_COMMAND_APPEND
## to create the MPI run line.
##
## MPIEXEC and MPIEXEC_NUMPROC_FLAG are filled in by CMake's FindMPI.cmake but can
## be overwritten in your host-config specific to your platform. BLT_MPI_COMMAND_APPEND
## is useful on machines that require extra arguments to MPIEXEC.
##------------------------------------------------------------------------------
macro(blt_add_test)

    set(options )
    set(singleValueArgs NAME NUM_MPI_TASKS)
    set(multiValueArgs COMMAND)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    if ( NOT DEFINED arg_NAME )
        message(FATAL_ERROR "NAME is a required parameter to blt_add_test")
    endif()

    if ( NOT DEFINED arg_COMMAND )
        message(FATAL_ERROR "COMMAND is a required parameter to blt_add_test")
    endif()

    # Extract test directory and executable from arg_NAME and arg_COMMAND
    if ( NOT TARGET ${arg_NAME} )
        # Handle cases where multiple tests are run against one executable
        # the NAME will not be the target
        list(GET arg_COMMAND 0 test_executable)
        get_target_property(test_directory ${test_executable} RUNTIME_OUTPUT_DIRECTORY )
    else()
        set(test_executable ${arg_NAME})
        get_target_property(test_directory ${arg_NAME} RUNTIME_OUTPUT_DIRECTORY )
    endif()
    
    # Append the test_directory to the test argument, accounting for multi-config generators
    if(NOT CMAKE_CONFIGURATION_TYPES)
        set(test_command ${test_directory}/${arg_COMMAND} )
    else()
        list(INSERT arg_COMMAND 0 "$<TARGET_FILE:${test_executable}>")
        list(REMOVE_AT arg_COMMAND 1)
        set(test_command ${arg_COMMAND})
    endif()

    # If configuration option ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC is set, 
    # ensure NUM_MPI_TASKS is at least one. This invokes the test 
    # through MPIEXEC.
    if ( ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC AND NOT arg_NUM_MPI_TASKS )
        set( arg_NUM_MPI_TASKS 1 )
    endif()

    # Handle MPI
    if ( ${arg_NUM_MPI_TASKS} )
        set(test_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_MPI_TASKS} ${BLT_MPI_COMMAND_APPEND} ${test_command} )
    endif()

    add_test(NAME ${arg_NAME}
             COMMAND ${test_command} 
             )

endmacro(blt_add_test)


##------------------------------------------------------------------------------
## blt_add_benchmark( NAME [name] COMMAND [command]  )
##
## Adds a (google) benchmark test to the project.
##
## NAME is used for the name that CTest reports and should include the string 'benchmark'.
##
## COMMAND is the command line that will be used to run the test and can include arguments.  
## This will have the RUNTIME_OUTPUT_DIRECTORY prepended to it to fully qualify the path.
##
## The underlying executable (added with blt_add_executable) should include gbenchmark
## as one of its dependencies.
##
##  Example
##    blt_add_executable(NAME component_benchmark ... DEPENDS gbenchmark)
##    blt_add_benchmark( 
##          NAME component_benchmark
##          COMMAND component_benchmark "--benchmark_min_time=0.0 --v=3 --benchmark_format=json"
##          )
##------------------------------------------------------------------------------
macro(blt_add_benchmark)

   if(ENABLE_BENCHMARKS)

      set(options)
      set(singleValueArgs NAME)      
      set(multiValueArgs COMMAND)

      ## parse the arguments to the macro
      cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

      if ( NOT DEFINED arg_NAME )
          message(FATAL_ERROR "NAME is a required parameter to blt_add_benchmark")
      endif()

      if ( NOT DEFINED arg_COMMAND )
          message(FATAL_ERROR "COMMAND is a required parameter to blt_add_benchmark")
      endif()

      # Generate command
      if ( NOT TARGET ${arg_NAME} )
          # Handle case of running multiple tests against one executable, 
          # the NAME will not be the target
          list(GET arg_COMMAND 0 executable)
          get_target_property(runtime_output_directory ${executable} RUNTIME_OUTPUT_DIRECTORY )
      else()
          get_target_property(runtime_output_directory ${arg_NAME} RUNTIME_OUTPUT_DIRECTORY )
      endif()
      set(test_command ${runtime_output_directory}/${arg_COMMAND} )

      # Note: No MPI handling for now.  If desired, see how this is handled in blt_add_test macro

      # The 'CONFIGURATIONS Benchmark' line excludes benchmarks 
      # from the general list of tests
      add_test( NAME ${arg_NAME}
                COMMAND ${test_command}
                CONFIGURATIONS Benchmark   
                )

      if(ENABLE_TESTS)
        add_dependencies(run_benchmarks ${arg_NAME})
      endif()
   endif()

endmacro(blt_add_benchmark)

##------------------------------------------------------------------------------
## blt_append_custom_compiler_flag( 
##                    FLAGS_VAR  flagsVar       (required)
##                    DEFAULT    defaultFlag    (optional)
##                    GNU        gnuFlag        (optional)
##                    CLANG      clangFlag      (optional)
##                    HCC        hccFlag        (optional)
##                    INTEL      intelFlag      (optional)
##                    XL         xlFlag         (optional)
##                    MSVC       msvcFlag       (optional)
##                    MSVC_INTEL msvcIntelFlag  (optional)
##                    PGI        pgiFlag        (optional)
## )
##
## Appends compiler-specific flags to a given variable of flags
##
## If a custom flag is given for the current compiler, we use that.
## Otherwise, we will use the DEFAULT flag (if present)
## If ENABLE_FORTRAN is On, any flagsVar with "fortran" (any capitalization)
## in its name will pick the compiler family (GNU,CLANG, INTEL, etc) based on
## the fortran compiler family type. This allows mixing C and Fortran compiler
## families, e.g. using Intel fortran compilers with clang C compilers. 
## When using the Intel toolchain within visual studio, we use the 
## MSVC_INTEL flag, when provided, with a fallback to the MSVC flag.
##------------------------------------------------------------------------------
macro(blt_append_custom_compiler_flag)

   set(options)
   set(singleValueArgs FLAGS_VAR DEFAULT GNU CLANG HCC PGI INTEL XL MSVC MSVC_INTEL)
   set(multiValueArgs)

   # Parse the arguments
   cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

   # Sanity check for required keywords
   if(NOT DEFINED arg_FLAGS_VAR)
      message( FATAL_ERROR "append_custom_compiler_flag macro requires FLAGS_VAR keyword and argument." )
   endif()
   
   # Set the desired flags based on the compiler family   
   # MSVC COMPILER FAMILY applies to C/C++ and Fortran
   string( TOLOWER ${arg_FLAGS_VAR} lower_flag_var )
   if( DEFINED arg_MSVC_INTEL AND COMPILER_FAMILY_IS_MSVC_INTEL )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_MSVC_INTEL} " )
   elseif( DEFINED arg_MSVC AND (COMPILER_FAMILY_IS_MSVC OR COMPILER_FAMILY_IS_MSVC_INTEL) )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_MSVC} " )
   
   #else, if we are setting a fortran flag, check against the fortran compiler family
   elseif ( ENABLE_FORTRAN AND ${lower_flag_var} MATCHES "fortran" )
       if( DEFINED arg_CLANG AND Fortran_COMPILER_FAMILY_IS_CLANG )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_CLANG} " )
       elseif( DEFINED arg_XL AND Fortran_COMPILER_FAMILY_IS_XL )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_XL} " )
       elseif( DEFINED arg_INTEL AND Fortran_COMPILER_FAMILY_IS_INTEL )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_INTEL} " )
       elseif( DEFINED arg_PGI AND Fortran_COMPILER_FAMILY_IS_PGI )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_PGI} " )
       elseif( DEFINED arg_GNU AND Fortran_COMPILER_FAMILY_IS_GNU )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_GNU} " )
       elseif( DEFINED arg_DEFAULT )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_DEFAULT} ")
       endif()
  
   #else, we are setting a non MSVC C/C++ flag, check against the C family. 
   else()
       if( DEFINED arg_CLANG AND C_COMPILER_FAMILY_IS_CLANG )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_CLANG} " )
       elseif( DEFINED arg_XL AND C_COMPILER_FAMILY_IS_XL )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_XL} " )
       elseif( DEFINED arg_INTEL AND C_COMPILER_FAMILY_IS_INTEL )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_INTEL} " )
       elseif( DEFINED arg_PGI AND C_COMPILER_FAMILY_IS_PGI )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_PGI} " )
       elseif( DEFINED arg_GNU AND C_COMPILER_FAMILY_IS_GNU )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_GNU} " )
       elseif( DEFINED arg_DEFAULT )
          set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_DEFAULT} ")
       endif()
   endif()
   unset(lower_flag_var)

endmacro(blt_append_custom_compiler_flag)


##------------------------------------------------------------------------------
## blt_find_libraries( FOUND_LIBS <FOUND_LIBS variable name>
##                     NAMES [libname1 [libname2 ...]]
##                     REQUIRED [TRUE (default) | FALSE ]
##                     PATHS [path1 [path2 ...]] )
##
## This command is used to find a list of libraries.
## 
## If the libraries are found the results are appended to the given FOUND_LIBS variable name.
##
## NAMES lists the names of the libraries that will be searched for in the given PATHS.
##
## If REQUIRED is set to TRUE, BLT will produce an error message if any of the
## given libraries are not found.  The default value is TRUE.
##
## PATH lists the paths in which to search for NAMES. No system paths will be searched.
##
##------------------------------------------------------------------------------
macro(blt_find_libraries)

    set(options )
    set(singleValueArgs FOUND_LIBS REQUIRED )
    set(multiValueArgs NAMES PATHS )

    ## parse the arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    if ( NOT DEFINED arg_FOUND_LIBS )
        message(FATAL_ERROR "The blt_find_libraries required parameter FOUND_LIBS specifies the list that found libraries will be appended to.")
    endif()

    if ( NOT DEFINED arg_NAMES )
        message(FATAL_ERROR "The blt_find_libraries required parameter NAMES specifies the library names you are searching for.")
    endif()

    if ( NOT DEFINED arg_PATHS )
        message(FATAL_ERROR "The blt_find_libraries required parameter PATHS specifies the paths to search for NAMES.")
    endif()

    if ( NOT DEFINED arg_REQUIRED)
        set(arg_REQUIRED TRUE)
    endif()

    foreach( lib ${arg_NAMES} )
        unset( temp CACHE )
        find_library( temp NAMES ${lib}
                      PATHS ${arg_PATHS}
                      NO_DEFAULT_PATH
                      NO_CMAKE_ENVIRONMENT_PATH
                      NO_CMAKE_PATH
                      NO_SYSTEM_ENVIRONMENT_PATH
                      NO_CMAKE_SYSTEM_PATH)
        if( temp )
            list( APPEND ${arg_FOUND_LIBS} ${temp} )
        elseif (${arg_REQUIRED})
            message(FATAL_ERROR "blt_find_libraries required NAMES entry ${lib} not found. These are not the libs you are looking for.")
        endif()
    endforeach()

endmacro(blt_find_libraries)


##------------------------------------------------------------------------------
## blt_combine_static_libraries( NAME <libname>
##                               SOURCE_LIBS [lib1 ...] 
##                               LIB_TYPE [STATIC,SHARED]
##                               LINK_PREPEND []
##                               LINK_POSTPEND []
##                             )
##
## Adds a library target, called <libname>, to be built from the set of 
## static libraries given in SOURCE_LIBS.
## 
## The LINK_PREPEND argument will be prepended to the library on the link line,
## while the LINK_POSTPEND will be appended to the library on the link line.
## These values are defaulted to the appropriate values for CMAKE_HOST_APPLE and 
## CMAKE_HOST_UNIX.
##
## Note: This macro does not currently work for Windows
##
##------------------------------------------------------------------------------
macro(blt_combine_static_libraries)

    set(options )
    set(singleValueArgs NAME LINK_PREPEND LINK_POSTPEND LIB_TYPE )
    set(multiValueArgs SOURCE_LIBS )

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # sanity checks
    if( "${arg_NAME}" STREQUAL "" )
        message(FATAL_ERROR "blt_combine_static_libraries() must be called with argument NAME <name>")
    endif()

    if(TARGET ${arg_NAME})
        message(FATAL_ERROR "A target with the name ${arg_NAME} already exists!")
    endif()

    if( NOT arg_SOURCE_LIBS )
        message(FATAL_ERROR "blt_combine_static_libraries(NAME ${arg_NAME} ...) called with no given source libraries")
    endif()
    
    # Default linker flags if not given
    if( NOT arg_LINK_PREPEND )
        if( CMAKE_HOST_APPLE )
            set( _link_prepend "-Wl,-force_load" )
        elseif( CMAKE_HOST_UNIX )
            set( _link_prepend "-Wl,--whole-archive" )
        elseif( CMAKE_HOST_WIN32 )
            # This used to work. Needs to be fixed
            # set( _link_prepend "-WHOLEARCHIVE:" )
            message(FATAL_ERROR "blt_combine_static_libraries does not support ${CMAKE_HOST_SYSTEM}")
        else()
            message(FATAL_ERROR "blt_combine_static_libraries does not support ${CMAKE_HOST_SYSTEM}")
        endif()
    else()
        set( _link_prepend ${arg_LINK_PREPEND})
    endif()

    if( NOT arg_LINK_POSTPEND )
        if( CMAKE_HOST_APPLE )
            set( _link_postpend "" )
        elseif( CMAKE_HOST_UNIX )
            set( _link_postpend "-Wl,--no-whole-archive" )
        elseif( CMAKE_HOST_WIN32 )
            set( _link_postpend "" )
        else()
            message(FATAL_ERROR "blt_combine_static_libraries does not support ${CMAKE_HOST_SYSTEM}")
        endif()
    else()
        set( _link_postpend ${arg_LINK_POSTPEND})
    endif()
    
    # Create link line that has all the libraries to combine on it
    set( libLinkLine "" )
    foreach( lib ${arg_SOURCE_LIBS} )
        if( CMAKE_HOST_UNIX )
            list( APPEND libLinkLine ${_link_prepend} ${lib} ${_link_postpend} )
        elseif( CMAKE_HOST_WIN32 )
            list( APPEND libLinkLine "${_link_prepend}${lib}" )
        endif()
    endforeach()

    # Decide if the created library is static or shared
    if( ${arg_LIB_TYPE} STREQUAL "STATIC" )
        set( _lib_type STATIC )
    elseif( ${arg_LIB_TYPE} STREQUAL "SHARED" ) 
        set( _lib_type SHARED )
    else()
        message(FATAL_ERROR "blt_combine_static_libraries(NAME ${arg_NAME} ...) LIB_TYPE must be SHARED OR STATIC")
    endif()
    
    # Create new library with empty source file
    add_library( ${arg_NAME} ${_lib_type}
                 ${BLT_ROOT_DIR}/tests/internal/src/combine_static_library_test/dummy.cpp)

    # Add the combined link line flag
    target_link_libraries( ${arg_NAME} PRIVATE ${libLinkLine})

    # Add the includes that should be inherited from themselves and their dependencies
    set( interface_include_directories "" )
    set( interface_system_include_directories "" )
    foreach( source_lib ${arg_SOURCE_LIBS} )
    
        get_target_property( source_lib_system_include_directories 
                             ${source_lib} 
                             INTERFACE_SYSTEM_INCLUDE_DIRECTORIES )

        if( source_lib_system_include_directories )
            list( APPEND interface_system_include_directories
                         ${source_lib_system_include_directories} )
        endif()

        get_target_property( source_lib_include_directories 
                             ${source_lib} 
                             INTERFACE_INCLUDE_DIRECTORIES )
        
        if( source_lib_include_directories )
            list( APPEND interface_include_directories ${source_lib_include_directories} )
        endif()

        # Get all includes from the dependencies of the libraries to be combined
        get_target_property( interface_link_libs ${source_lib} INTERFACE_LINK_LIBRARIES )        
        foreach( interface_link_lib ${interface_link_libs} )
            # Filter out non-CMake targets
            if( TARGET ${interface_link_lib} )
                get_target_property( interface_link_lib_include_dir 
                                     ${interface_link_lib} 
                                     INTERFACE_INCLUDE_DIRECTORIES )
                                     
                if( interface_link_lib_include_dir )
                    list( APPEND interface_include_directories ${interface_link_lib_include_dir} )
                endif()

                get_target_property( interface_link_lib_system_include_dir 
                                     ${interface_link_lib} 
                                     INTERFACE_SYSTEM_INCLUDE_DIRECTORIES )
                                     
                if( interface_link_lib_system_include_dir )
                    list( APPEND interface_system_include_directories
                                 ${interface_link_lib_system_include_dir} )
                endif()

                get_target_property( target_type ${interface_link_lib}  TYPE )
                if( target_type STREQUAL "SHARED_LIBRARY" )
                     target_link_libraries( ${arg_NAME} PUBLIC ${interface_link_lib} )
                endif ()

            elseif( ${interface_link_lib} MATCHES ".so" OR 
                    ${interface_link_lib} MATCHES ".dll" OR 
                    ${interface_link_lib} MATCHES ".dylib" )
                # Add any shared libraries that were added by file name
                target_link_libraries( ${arg_NAME} PUBLIC ${interface_link_lib} )
            endif()            
        endforeach()
    endforeach()
    
    # Remove duplicates from the includes
    list( REMOVE_DUPLICATES interface_include_directories )
    list( REMOVE_DUPLICATES interface_system_include_directories )

    # Remove any system includes from the regular includes
    foreach( include_dir ${interface_system_include_directories} )
        if( ${include_dir} IN_LIST interface_include_directories )
            list( REMOVE_ITEM interface_include_directories ${include_dir} )
        endif()
    endforeach()

    target_include_directories( ${arg_NAME} INTERFACE
                                ${interface_include_directories} )

    target_include_directories( ${arg_NAME} SYSTEM INTERFACE
                                ${interface_system_include_directories} )

    unset( libLinkLine )
    unset( interface_include_directories )
    unset( interface_system_include_directories )
    unset( interface_link_lib_include_dir )
    unset( source_lib_include_directories )
    unset( _lib_type )
    unset( _link_prepend )
    unset( _link_postpend )
endmacro(blt_combine_static_libraries)



##------------------------------------------------------------------------------
## blt_print_target_properties (TARGET <target> )
##
## Prints out all properties of the given target.
##
## The required target parameteter must either be a valid cmake target 
## or was registered via blt_register_library.
##
## Output is of the form:
##     [<target> property] <property>: <value>
## for each property
##------------------------------------------------------------------------------
macro(blt_print_target_properties)

    set(options)
    set(singleValuedArgs TARGET)
    set(multiValuedArgs)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN})

    ## check for required arguments
    if(NOT DEFINED arg_TARGET)
        message(FATAL_ERROR "TARGET is a required parameter for the blt_print_target_properties macro")
    endif()

    ## check if this is a valid cmake target of blt_registered target
    set(_is_cmake_target FALSE)
    if(TARGET ${arg_TARGET})
        set(_is_cmake_target TRUE)
        message (STATUS "[${arg_TARGET} property] '${arg_TARGET}' is a cmake target")
    endif()

    set(_is_blt_registered_target FALSE)
    string(TOUPPER ${arg_TARGET} _target_upper)
    if(BLT_${_target_upper}_IS_REGISTERED_LIBRARY)
        set(_is_blt_registered_target TRUE)
        message (STATUS "[${arg_TARGET} property] '${arg_TARGET}' is a blt_registered target")
    endif()

    if(NOT _is_cmake_target AND NOT _is_blt_registered_target)
        message (STATUS "[blt_print_target_properties] Invalid argument '${arg_TARGET}'. "
                         "This macro applies only to valid cmake targets or blt_registered targets.")
    endif()


    if(_is_cmake_target)
        ## Solution adapted from https://stackoverflow.com/q/32183975
        ## Create list of cmake properties
        set(_property_list)
        execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE _property_list)
        string(REGEX REPLACE ";" "\\\\;" _property_list "${_property_list}")
        string(REGEX REPLACE "\n" ";" _property_list "${_property_list}")
        blt_filter_list(TO _property_list REGEX "^LOCATION$|^LOCATION_|_LOCATION$" OPERATION "exclude")
        list(REMOVE_DUPLICATES _property_list)   

        ## For interface targets, filter against whitelist of valid properties
        get_property(_targetType TARGET ${arg_TARGET} PROPERTY TYPE)
        if(${_targetType} STREQUAL "INTERFACE_LIBRARY")
            blt_filter_list(TO _property_list
                            REGEX "^(INTERFACE_|IMPORTED_LIBNAME_|COMPATIBLE_INTERFACE_|MAP_IMPORTED_CONFIG_)|^(NAME|TYPE|EXPORT_NAME)$"
                            OPERATION "include")
        endif()

        ## Print all such properties that have been SET
        foreach (prop ${_property_list})
            string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
            get_property(_propval TARGET ${arg_TARGET} PROPERTY ${prop} SET)
            if (_propval)
                get_target_property(_propval ${arg_TARGET} ${prop})
                message (STATUS "[${arg_TARGET} property] ${prop}: ${_propval}")
            endif()
        endforeach()
        unset(_property_list)
        unset(_propval)
    endif()

    ## Additionally, output variables generated via blt_register_target of the form "BLT_<target>_*"
    if(_is_blt_registered_target)
        set(_target_prefix "BLT_${_target_upper}_")

        ## Filter to get variables of the form BLT_<target>_ and print
        get_cmake_property(_variable_names VARIABLES)
        foreach (prop ${_variable_names})
            if(prop MATCHES "${_target_prefix}?")
                message (STATUS "[${arg_TARGET} property] ${prop}: ${${prop}}")
            endif()
        endforeach()
        unset(_target_prefix)
        unset(_variable_names)
    endif()

    unset(_target_upper)
    unset(_is_blt_registered_target)
    unset(_is_cmake_target)
endmacro(blt_print_target_properties)

