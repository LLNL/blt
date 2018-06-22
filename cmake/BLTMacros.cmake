###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
## Note, the list of target definitions *SHOULD NOT* include the "-D" flag. This
## flag is added internally by cmake.
##------------------------------------------------------------------------------
macro(blt_add_target_definitions)

    set(options)
    set(singleValueArgs TO)
    set(multiValueArgs TARGET_DEFINITIONS)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    get_target_property(defs ${arg_TO} COMPILE_DEFINITIONS)
    if (defs MATCHES "NOTFOUND")
        set(defs "")
    endif()

    foreach (def ${defs} ${arg_TARGET_DEFINITIONS})
        list(APPEND deflist ${def})
    endforeach()

    set_target_properties(${arg_TO} PROPERTIES COMPILE_DEFINITIONS "${deflist}")

endmacro(blt_add_target_definitions)


##------------------------------------------------------------------------------
## blt_add_target_compile_flags (TO <target> FLAGS [FOO [BAR ...]])
##
## Adds compiler flags to a target by appending to the target's existing flags.
##------------------------------------------------------------------------------
macro(blt_add_target_compile_flags)

    set(options)
    set(singleValuedArgs TO)
    set(multiValuedArgs FLAGS)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    if(NOT "${arg_FLAGS}" STREQUAL "")
        # get prev flags
        get_target_property(_COMP_FLAGS ${arg_TO} COMPILE_FLAGS)
        if(NOT _COMP_FLAGS)
            set(_COMP_FLAGS "")
        endif()
        # append new flags
        set(_COMP_FLAGS "${arg_FLAGS} ${_COMP_FLAGS}")
        set_target_properties(${arg_TO}
                              PROPERTIES COMPILE_FLAGS "${_COMP_FLAGS}" )
    endif()

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
##                  HEADERS_OUTPUT_SUBDIR [dir]
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
## If given a HEADERS argument and ENABLE_COPY_HEADERS is ON, it first copies
## the headers into the out-of-source build directory under the
## include/<HEADERS_OUTPUT_SUBDIR>. Because of this HEADERS_OUTPUT_SUBDIR must
## be a relative path.
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
    set(singleValueArgs NAME OUTPUT_NAME OUTPUT_DIR HEADERS_OUTPUT_SUBDIR SHARED CLEAR_PREFIX FOLDER OBJECT )
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
    
    if( ${arg_OBJECT} )
        if( ${_build_shared_library} )
            message(FATAL_ERROR "blt_add_library() specified Object Library AND Shared Library")
        endif()
    endif()

    if ( arg_SOURCES )
        #
        #  CUDA support
        #
        list(FIND arg_DEPENDS_ON "cuda" check_for_cuda)
        if ( ${check_for_cuda} GREATER -1 AND NOT ENABLE_CLANG_CUDA)

            blt_setup_cuda_source_properties(BUILD_TARGET ${arg_NAME}
                                             TARGET_SOURCES ${arg_SOURCES})

        if ( ${arg_OBJECT} )
            message( "adding ${arg_NAME} as an OBJECT LIBRARY " )
            add_library( ${arg_NAME} OBJECT ${arg_SOURCES} ${arg_HEADERS} )
        elseif ( ${_build_shared_library} )
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
        if ( ENABLE_COPY_HEADERS )
            target_sources( ${arg_NAME} INTERFACE
                        $<BUILD_INTERFACE:${_build_headers}>
                        $<INSTALL_INTERFACE:${_install_headers}>)

            target_include_directories(${arg_NAME} INTERFACE
                        $<BUILD_INTERFACE:${HEADER_INCLUDES_DIRECTORY}>
                        $<INSTALL_INTERFACE:include/${arg_HEADERS_OUTPUT_SUBDIR}>)
        else()
            target_sources( ${arg_NAME} INTERFACE
                        $<BUILD_INTERFACE:${_build_headers}>)
        endif()
    endif()

    # Handle copying headers
    if ( arg_HEADERS AND ENABLE_COPY_HEADERS )
        # Determine build location of headers
        set(headers_build_dir ${HEADER_INCLUDES_DIRECTORY})
        if (arg_HEADERS_OUTPUT_SUBDIR)
            if (IS_ABSOLUTE ${arg_HEADERS_OUTPUT_SUBDIR})
                message(FATAL_ERROR "blt_add_library must be called with a relative path for HEADERS_OUTPUT_SUBDIR")
            endif()
            set(headers_build_dir ${headers_build_dir}/${arg_HEADERS_OUTPUT_SUBDIR})
        endif()

        blt_copy_headers_target( NAME        ${arg_NAME}
                                 HEADERS     ${arg_HEADERS}
                                 DESTINATION ${headers_build_dir})
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
                      DEPENDS_ON ${arg_DEPENDS_ON}
                      OBJECT ${arg_OBJECT} )

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
##                     OUTPUT_DIR [dir])
##                     FOLDER [name]
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

    # CMake wants to load with C++ if any of the libraries are C++.
    # Force to load with Fortran if the first file is Fortran.
    list(GET arg_SOURCES 0 _first)
    get_source_file_property(_lang ${_first} LANGUAGE)
    if(_lang STREQUAL Fortran)
        set_target_properties( ${arg_NAME} PROPERTIES LINKER_LANGUAGE Fortran )
        target_include_directories(${arg_NAME} PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
    endif()
       
    blt_setup_target(NAME ${arg_NAME}
                     DEPENDS_ON ${arg_DEPENDS_ON} )

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
## Adds a cmake test to the project.
##
## NAME is used for the name that CTest reports with.
##
## COMMAND is the command line that will be used to run the test. This will
## have the RUNTIME_OUTPUT_DIRECTORY prepended to it to fully qualify the path.
##
## NUM_MPI_TASKS indicates this is an MPI test and how many tasks to use. The
## command line will use MPIEXEC and MPIXEC_NUMPROC_FLAG to create the mpi run
## line.
###
## These should be defined in your host-config specific to your platform.
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

    # Handle mpi
    if ( ${arg_NUM_MPI_TASKS} )
        set(test_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_MPI_TASKS} ${test_command} )
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
    
    if( NOT arg_LINK_PREPEND )
        if( CMAKE_HOST_APPLE )
            set( _link_prepend "-Wl,-force_load" )
        elseif( CMAKE_HOST_UNIX )
            set( _link_prepend "-Wl,--whole-archive" )
        elseif( CMAKE_HOST_WIN32 )
            set( _link_prepend "-WHOLEARCHIVE:" )
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
    
    set( libLinkLine "" )
    foreach( lib ${arg_SOURCE_LIBS} )
        if( CMAKE_HOST_UNIX )
            list( APPEND libLinkLine ${_link_prepend} ${lib} ${_link_postpend} )
        elseif( CMAKE_HOST_WIN32 )
            list( APPEND libLinkLine "${_link_prepend}${lib}" )
        endif()
    endforeach()

    message( "libLinkLine = ${libLinkLine}" )
    
    if( ${arg_LIB_TYPE} STREQUAL "STATIC" )
        set( _lib_type STATIC )
    elseif( ${arg_LIB_TYPE} STREQUAL "SHARED" ) 
        set( _lib_type SHARED )
    else()
        message(FATAL_ERROR "blt_combine_static_libraries(NAME ${arg_NAME} ...) LIB_TYPE must be SHARED OR STATIC")
    endif()
    
    add_library ( ${arg_NAME} ${_lib_type} ${BLT_ROOT_DIR}/tests/internal/src/combine_static_library_test/dummy.cpp)
    target_link_libraries( ${arg_NAME} PRIVATE ${libLinkLine})
    blt_register_library( NAME ${arg_NAME}
                          DEPENDS_ON ${arg_SOURCE_LIBS}
                          LIBRARIES ${arg_NAME}
                         )

    unset( libLinkLine )
endmacro(blt_combine_static_libraries)
