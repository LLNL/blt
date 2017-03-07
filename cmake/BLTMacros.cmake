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
## blt_add_component( COMPONENT_NAME <name> DEFAULT_STATE [ON/OFF] )
##
## Adds a project component to the build.
##
## Adds a component to the build given the component's name and default state
## (ON/OFF). This macro also adds an "option" so that the user can control,
## which components to build.
##------------------------------------------------------------------------------
macro(blt_add_component)

    set(options)
    set(singleValueArgs COMPONENT_NAME DEFAULT_STATE )
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # Setup a cmake vars to capture sources added via our macros
    set("${arg_COMPONENT_NAME}_ALL_SOURCES" CACHE PATH "" FORCE)

    # Adds an option so that the user can control whether to build this
    # component.
    # convert the component name to capitals for the ENABLE option.
    string(TOUPPER ${arg_COMPONENT_NAME} COMPONENT_NAME_CAPITALIZED)
    string(TOLOWER ${arg_COMPONENT_NAME} COMPONENT_NAME_LOWERED)

    option(ENABLE_${COMPONENT_NAME_CAPITALIZED}
           "Enables ${arg_component_name}"
           ${arg_DEFAULT_STATE})

    if ( ENABLE_${COMPONENT_NAME_CAPITALIZED} )
        add_subdirectory( ${arg_COMPONENT_NAME} )
    endif()

    unset(COMPONENT_NAME_CAPITALIZED)
    unset(COMPONENT_NAME_LOWERED)

endmacro(blt_add_component)


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
## blt_register_library( NAME <libname>
##                       INCLUDES [include1 [include2 ...]] 
##                       FORTRAN_MODULES [ path1 [ path2 ..]]
##                       LIBRARIES [lib1 [lib2 ...]]
##                       DEFINES [def1 [def2 ...]] )
##
## Registers a library to the project to ease use in other blt macro calls.
##
## Stores information about a library in a specific way that is easily recalled
## in other macros.  For example, after registering gtest, you can add gtest to
## the DEPENDS_ON in your blt_add_executable call and it will add the INCLUDES
## and LIBRARIES to that executable.
##
## This does not actually build the library.  This is strictly to ease use after
## discovering it on your system or building it yourself inside your project.
##
## Output variables (name = "foo"):
##  BLT_FOO_INCLUDES
##  BLT_FOO_FORTRAN_MODULES
##  BLT_FOO_LIBRARIES
##  BLT_FOO_DEFINES
##------------------------------------------------------------------------------
macro(blt_register_library)

    set(singleValueArgs NAME )
    set(multiValueArgs INCLUDES FORTRAN_MODULES LIBRARIES DEFINES )

    ## parse the arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    string(TOUPPER ${arg_NAME} uppercase_name)

    if( arg_INCLUDES )
        set(BLT_${uppercase_name}_INCLUDES ${arg_INCLUDES} CACHE PATH "" FORCE)
    endif()

    if( ENABLE_FORTRAN AND arg_FORTRAN_MODULES )
        set(BLT_${uppercase_name}_FORTRAN_MODULES ${arg_INCLUDES} CACHE PATH "" FORCE)
    endif()

    if( arg_LIBRARIES )
        set(BLT_${uppercase_name}_LIBRARIES ${arg_LIBRARIES} CACHE PATH "" FORCE)
    else()
        set(BLT_${uppercase_name}_LIBRARIES "BLT_NO_LIBRARIES" CACHE PATH "" FORCE)
    endif()

    if( arg_DEFINES )
        set(BLT_${uppercase_name}_DEFINES ${arg_DEFINES} CACHE PATH "" FORCE)
    endif()

endmacro(blt_register_library)


##------------------------------------------------------------------------------
## blt_add_library( NAME <libname>
##                  SOURCES [source1 [source2 ...]]
##                  HEADERS [header1 [header2 ...]]
##                  DEPENDS_ON [dep1 ...] 
##                  OUTPUT_NAME [name]
##                  OUTPUT_DIR [dir]
##                  HEADERS_OUTPUT_SUBDIR [dir]
##                  PYTHON_MODULE
##                  LUA_MODULE
##                  SHARED
##                 )
##
## Adds a library to the project composed by the given source files.
##
## Adds a library target, called <libname>, to be built from the given sources.
## This macro internally checks if the global option "ENABLE_SHARED_LIBS" is
## ON, in which case, it will create a shared library. By default, a static
## library is generated unless the SHARED option is added.
##
## If given a HEADERS argument, it first copies the headers into the out-of-source
## build directory under the include/<HEADERS_OUTPUT_SUBDIR> and installs
## them to the CMAKE_INSTALL_PREFIX/include/<HEADERS_OUTPUT_SUBDIR>.
## Because of this HEADERS_OUTPUT_SUBDIR must be a relative path.
## 
## If given a DEPENDS_ON argument, it will add the necessary includes and 
## libraries if they are already registered with blt_register_library.  If 
## not it will add them as a cmake target dependency.
##
## In addition, this macro will add the associated dependencies to the given
## library target. Specifically, it will add the dependency for the CMake target
## and for copying the headers for that target as well.
##
## The OUTPUT_DIR is used to control the build output directory of this 
## library. This is used to overwrite the default lib directory.
##
## OUTPUT_NAME is the name of the output file.  It defaults to NAME.
## It's useful when multiple libraries with the same name need to be created
## by different targets. NAME is the target name, OUTPUT_NAME is the library name.
##
## If DEPENDS_ON includes "openmp", the openmp compiler flags will be added 
## to the target and -DUSE_OPENMP will be added to the target's compiler 
## definitions.
##
## The PYTHON_MODULE option customizes arguments for a Python module.
## The target created will be NAME-python-module and the library will be NAME.so.
## In addition, python is added to DEPENDS_ON and OUTPUT_DIR is defaulted to
## BLT_Python_MODULE_DIRECTORY.
## Likewise, LUA_MODULE helps create a module for Lua.

##------------------------------------------------------------------------------
macro(blt_add_library)

    set(arg_CLEAR_PREFIX FALSE)
    set(options SHARED PYTHON_MODULE LUA_MODULE)
    set(singleValueArgs NAME OUTPUT_NAME OUTPUT_DIR HEADERS_OUTPUT_SUBDIR)
    set(multiValueArgs SOURCES HEADERS DEPENDS_ON)

    # parse the arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Check for the variable-based options and sanity checks
    if(arg_PYTHON_MODULE)
        set(arg_SHARED TRUE)
        if( NOT arg_OUTPUT_DIR )
            set(arg_OUTPUT_DIR ${BLT_Python_MODULE_DIRECTORY})
        endif()
        set(arg_DEPENDS_ON "${arg_DEPENDS_ON};python")
        set(arg_OUTPUT_NAME ${arg_NAME})
        set(arg_NAME "${arg_NAME}-python-module")
        set(arg_CLEAR_PREFIX TRUE)
    endif()

    if(arg_LUA_MODULE)
        set(arg_SHARED TRUE)
        if( NOT arg_OUTPUT_DIR )
            set(arg_OUTPUT_DIR ${BLT_Lua_MODULE_DIRECTORY})
        endif()
        set(arg_DEPENDS_ON "${arg_DEPENDS_ON};lua")
        set(arg_OUTPUT_NAME ${arg_NAME})
        set(arg_NAME "${arg_NAME}-lua-module")
        set(arg_CLEAR_PREFIX TRUE)
    endif()

    if ( arg_SHARED OR ENABLE_SHARED_LIBS )
        add_library( ${arg_NAME} SHARED ${arg_SOURCES} ${arg_HEADERS} )
    else()
        add_library( ${arg_NAME} STATIC ${arg_SOURCES} ${arg_HEADERS} )
    endif()

    # Handle copying and installing headers
    if ( arg_HEADERS )
        # Determine build and install location of headers
        set(headers_build_dir ${HEADER_INCLUDES_DIRECTORY})
        set(headers_install_dir ${CMAKE_INSTALL_PREFIX}/include)
        if (arg_HEADERS_OUTPUT_SUBDIR)
            if (IS_ABSOLUTE ${arg_HEADERS_OUTPUT_SUBDIR})
                message(FATAL_ERROR "blt_add_library must be called with a relative path for HEADERS_OUTPUT_SUBDIR")
            else()
                set(headers_build_dir ${headers_build_dir}/${arg_HEADERS_OUTPUT_SUBDIR})
                set(headers_install_dir ${headers_install_dir}/${arg_HEADERS_OUTPUT_SUBDIR})
            endif()
        endif()

        if ( ENABLE_COPY_HEADERS )
            blt_copy_headers_target( NAME        ${arg_NAME}
                                     HEADERS     ${arg_HEADERS}
                                     DESTINATION ${headers_build_dir})
        endif()

        install(FILES ${arg_HEADERS} DESTINATION ${headers_install_dir})
    endif()
    install(TARGETS ${arg_NAME} DESTINATION lib EXPORT ${arg_NAME}-targets)

    # Must tell fortran where to look for modules
    # CMAKE_Fortran_MODULE_DIRECTORY is the location of generated modules
    foreach (_file ${arg_SOURCES})
        get_source_file_property(_lang ${_file} LANGUAGE)
        if(_lang STREQUAL Fortran)
            set(_have_fortran TRUE)
        endif()
    endforeach()
    if(_have_fortran)
        include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
    endif()

    blt_setup_target( NAME ${arg_NAME}
                      DEPENDS_ON ${arg_DEPENDS_ON} )

    blt_setup_mpi_target( BUILD_TARGET ${arg_NAME} )

    if ( arg_OUTPUT_DIR )
        set_target_properties(${arg_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${arg_OUTPUT_DIR} )
    endif()

    if (arg_OUTPUT_NAME)
        set_target_properties(${arg_NAME} PROPERTIES
            OUTPUT_NAME ${arg_OUTPUT_NAME} )
    endif()

    if (arg_CLEAR_PREFIX)
        set_target_properties(${arg_NAME} PROPERTIES
            PREFIX "" )
    endif()

    blt_update_project_sources( TARGET_SOURCES ${arg_SOURCES} ${arg_HEADERS})

endmacro(blt_add_library)

##------------------------------------------------------------------------------
## blt_add_executable( NAME <name>
##                     SOURCES [source1 [source2 ...]]
##                     DEPENDS_ON [dep1 [dep2 ...]]
##                     OUTPUT_DIR [dir])
##
## Adds an executable target, called <name>.
##
## If given a DEPENDS_ON argument, it will add the necessary includes and 
## libraries if they are already registered with blt_register_library.  If
## not it will add them as a cmake target dependency.
##
## If DEPENDS_ON includes "openmp", the openmp compiler flags will be added 
## to the target and -DUSE_OPENMP will be added to the target's compiler 
## definitions.
##
## The OUTPUT_DIR is used to control the build output directory of this 
## executable. This is used to overwrite the default bin directory.
##
## If the first entry in SOURCES is a Fortran source file, the fortran linker 
## is used. (via setting the CMake target property LINKER_LANGUAGE to Fortran )
##
##------------------------------------------------------------------------------
macro(blt_add_executable)

    set(options )
    set(singleValueArgs NAME OUTPUT_DIR)
    set(multiValueArgs SOURCES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # sanity check
    if( "${arg_NAME}" STREQUAL "" )
        message(FATAL_ERROR "Must specify executable name with argument NAME <name>")
    endif()

    add_executable( ${arg_NAME} ${arg_SOURCES} )

    # CMake wants to load with C++ if any of the libraries are C++.
    # Force to load with Fortran if the first file is Fortran.
    list(GET arg_SOURCES 0 _first)
    get_source_file_property(_lang ${_first} LANGUAGE)
    if(_lang STREQUAL Fortran)
        set_target_properties( ${arg_NAME} PROPERTIES LINKER_LANGUAGE Fortran )
    endif()

    blt_setup_target(NAME ${arg_NAME}
                     DEPENDS_ON ${arg_DEPENDS_ON} )

    blt_setup_mpi_target( BUILD_TARGET ${arg_NAME} )

    if ( arg_OUTPUT_DIR )
        set_target_properties(${arg_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${arg_OUTPUT_DIR} )
    endif()

    blt_update_project_sources( TARGET_SOURCES ${arg_SOURCES} )

endmacro(blt_add_executable)


##------------------------------------------------------------------------------
## blt_add_test( NAME [name] COMMAND [command] NUM_PROCS [n] )
##
## Adds a cmake test to the project.
##
## NAME is used for the name that CTest reports with.
##
## COMMAND is the command line that will be used to run the test.  This will have
## the RUNTIME_OUTPUT_DIRECTORY prepended to it to fully qualify the path.
##
## NUM_PROCS indicates this is an MPI test and how many processors to use. The
## command line will use MPIEXEC and MPIXEC_NUMPROC_FLAG to create the mpi run line.
## These should be defined in your host-config specific to your platform.
##------------------------------------------------------------------------------
macro(blt_add_test)

    set(options )
    set(singleValueArgs NAME NUM_PROCS)
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

    # Handle mpi
    if ( ${arg_NUM_PROCS} )
        set(test_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_PROCS} ${test_command} )
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

      add_dependencies(run_benchmarks ${arg_NAME})
   endif()

endmacro(blt_add_benchmark)

##------------------------------------------------------------------------------
## blt_append_custom_compiler_flag( 
##                    FLAGS_VAR flagsVar     (required)
##                    DEFAULT   defaultFlag  (optional)
##                    GNU       gnuFlag      (optional)
##                    CLANG     clangFlag    (optional)
##                    INTEL     intelFlag    (optional)
##                    XL        xlFlag       (optional)
##                    MSVC      msvcFlag     (optional)
## )
##
## Appends compiler-specific flags to a given variable of flags
##
## If a custom flag is given for the current compiler, we use that,
## Otherwise, we will use the DEFAULT flag (if present)
##------------------------------------------------------------------------------
macro(blt_append_custom_compiler_flag)

   set(options)
   set(singleValueArgs FLAGS_VAR DEFAULT GNU CLANG INTEL XL MSVC)
   set(multiValueArgs)

   # Parse the arguments
   cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

   # Sanity check for required keywords
   if(NOT DEFINED arg_FLAGS_VAR)
      message( FATAL_ERROR "append_custom_compiler_flag macro requires FLAGS_VAR keyword and argument." )
   endif()

   # Set the desired flags based on the compiler family   
   if( DEFINED arg_CLANG AND COMPILER_FAMILY_IS_CLANG )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_CLANG} " )
   elseif( DEFINED arg_XL AND COMPILER_FAMILY_IS_XL )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_XL} " )
   elseif( DEFINED arg_INTEL AND COMPILER_FAMILY_IS_INTEL )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_INTEL} " )
   elseif( DEFINED arg_GNU AND COMPILER_FAMILY_IS_GNU )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_GNU} " )
   elseif( DEFINED arg_MSVC AND COMPILER_FAMILY_IS_MSVC )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_MSVC} " )
   elseif( DEFINED arg_DEFAULT )
      set (${arg_FLAGS_VAR} "${${arg_FLAGS_VAR}} ${arg_DEFAULT} ")
   endif()   

endmacro(blt_append_custom_compiler_flag)

