# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

include(${BLT_ROOT_DIR}/cmake/BLTPrivateMacros.cmake)

##------------------------------------------------------------------------------
## blt_list_append( TO <list> ELEMENTS [ <element>...] IF <bool> )
##
## Appends elements to a list if the specified bool evaluates to true.
##------------------------------------------------------------------------------
macro(blt_list_append)

    set(options)
    set(singleValueArgs TO IF)
    set(multiValueArgs ELEMENTS )

    # parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

     # Sanity checks
    if( NOT DEFINED arg_TO )
        message(FATAL_ERROR "blt_list_append() requires a TO <list> argument")
    endif()

    if ( NOT DEFINED arg_ELEMENTS )
         message(FATAL_ERROR "blt_list_append() requires ELEMENTS to be specified" )
    endif()

    # determine if we should add the elements to the list
    set(_shouldAdd FALSE )
    set(_listVar "${ARGN}")         # convert macro arguments to list variable
    if("IF" IN_LIST _listVar)
        set(_shouldAdd ${arg_IF})   # use IF condition, when present
    else()
        set(_shouldAdd TRUE)        # otherwise, always add the elements
    endif()

    # append if
    if ( ${_shouldAdd} )
        list( APPEND ${arg_TO} ${arg_ELEMENTS} )
    endif()

    unset(_shouldAdd)
    unset(_listVar)

endmacro(blt_list_append)


##------------------------------------------------------------------------------
## blt_list_remove_duplicates( TO <list> )
##
## Removes duplicate elements from the given TO list.
##------------------------------------------------------------------------------
macro(blt_list_remove_duplicates)

    set(options)
    set(singleValueArgs TO )
    set(multiValueArgs )

    # parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Sanity checks
    if( NOT DEFINED arg_TO )
        message(FATAL_ERROR "blt_list_append() requires a TO <list> argument")
    endif()

    if ( ${arg_TO} )
        list(REMOVE_DUPLICATES ${arg_TO} )
    endif()

endmacro(blt_list_remove_duplicates)


##------------------------------------------------------------------------------
## blt_add_target_definitions(TO     <target>
##                            SCOPE  <PUBLIC (Default)| INTERFACE | PRIVATE>
##                            TARGET_DEFINITIONS [FOO [BAR ...]])
##
## Adds pre-processor definitions to the given target.
##------------------------------------------------------------------------------
macro(blt_add_target_definitions)

    set(options)
    set(singleValueArgs TO SCOPE)
    set(multiValueArgs TARGET_DEFINITIONS)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # Sanity checks
    if(NOT TARGET ${arg_TO})
        message(FATAL_ERROR "Target ${arg_TO} passed to blt_add_target_definitions is not a valid cmake target")    
    endif()

    blt_determine_scope(TARGET ${arg_TO} SCOPE "${arg_SCOPE}" OUT _scope)

    # Only add the flag if it is not empty
    string(STRIP "${arg_TARGET_DEFINITIONS}" _strippedDefs)
    if(NOT "${_strippedDefs}" STREQUAL "")
        target_compile_definitions(${arg_TO} ${_scope} ${_strippedDefs})
    endif()

    unset(_scope)
    unset(_strippedDefs)

endmacro(blt_add_target_definitions)


##------------------------------------------------------------------------------
## blt_add_target_compile_flags(TO    <target>
##                              SCOPE  <PUBLIC (Default)| INTERFACE | PRIVATE>
##                              FLAGS [FOO [BAR ...]])
##
## Adds compiler flags to a target (library, executable or interface) by 
## appending to the target's existing flags.
##------------------------------------------------------------------------------
macro(blt_add_target_compile_flags)

    set(options)
    set(singleValuedArgs TO SCOPE)
    set(multiValuedArgs FLAGS)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    # Sanity checks
    if(NOT TARGET ${arg_TO})
        message(FATAL_ERROR "Target ${arg_TO} passed to blt_add_target_compile_flags is not a valid cmake target")    
    endif()

    blt_determine_scope(TARGET ${arg_TO} SCOPE "${arg_SCOPE}" OUT _scope)

    # Only add the flag if it is not empty
    string(STRIP "${arg_FLAGS}" _strippedFlags)
    if(NOT "${_strippedFlags}" STREQUAL "")
        target_compile_options(${arg_TO} ${_scope} ${_strippedFlags})
    endif()

    unset(_strippedFlags)
    unset(_scope)

endmacro(blt_add_target_compile_flags)


##------------------------------------------------------------------------------
## blt_set_target_folder(TARGET <target> FOLDER <folder>)
##
## Sets the folder property of cmake target <target> to <folder>.
##------------------------------------------------------------------------------
macro(blt_set_target_folder)

    set(options)
    set(singleValuedArgs TARGET FOLDER)
    set(multiValuedArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    # Sanity checks
    if(NOT DEFINED arg_TARGET)
        message(FATAL_ERROR "TARGET is a required parameter for blt_set_target_folder macro")
    endif()

    if(NOT TARGET ${arg_TARGET})
        message(FATAL_ERROR "Target ${arg_TARGET} passed to blt_set_target_folder is not a valid cmake target")
    endif()

    if(NOT DEFINED arg_FOLDER)
        message(FATAL_ERROR "FOLDER is a required parameter for blt_set_target_folder macro")
    endif()

    # Set the folder property for this target
    if(ENABLE_FOLDERS AND NOT "${arg_FOLDER}" STREQUAL "")
        set_property(TARGET ${arg_TARGET} PROPERTY FOLDER "${arg_FOLDER}")
    endif()

endmacro(blt_set_target_folder)


##------------------------------------------------------------------------------
## blt_add_target_link_flags (TO    <target>
##                            SCOPE <PUBLIC (Default)| INTERFACE | PRIVATE>
##                            FLAGS [FOO [BAR ...]])
##
## Adds linker flags to a target by appending to the target's existing flags.
##------------------------------------------------------------------------------
macro(blt_add_target_link_flags)

    set(options)
    set(singleValuedArgs TO SCOPE)
    set(multiValuedArgs FLAGS)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    set(_flags ${arg_FLAGS})
    # Convert rpath flag if linking with CUDA
    if (CUDA_LINK_WITH_NVCC)
        string(REPLACE "-Wl,-rpath," "-Xlinker -rpath -Xlinker "
                       _flags "${_flags}")
    endif()

    # Only add the flag if it is not empty
    if(NOT "${arg_FLAGS}" STREQUAL "")
        if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
            # In CMake 3.13+, LINK_FLAGS was converted to LINK_OPTIONS.
            # This now supports generator expressions and scoping but expects a list
            # not a string
            blt_determine_scope(TARGET ${arg_TO} SCOPE "${arg_SCOPE}" OUT _scope)

            # Note: "SHELL:"" causes the flags to be not de-duplicated and parsed with
            # separate_arguments
            if(NOT "${arg_FLAGS}" MATCHES SHELL:)
                target_link_options(${arg_TO} ${_scope} SHELL:${arg_FLAGS})
            else()
                target_link_options(${arg_TO} ${_scope} ${arg_FLAGS})
            endif()
        else()
            # In CMake <= 3.12, there is no target_link_flags or target_link_options command
            get_target_property(_link_flags ${arg_TO} LINK_FLAGS)
            if(NOT _link_flags)
                set(_link_flags "")
            endif()
            set(_link_flags "${_flags} ${_link_flags}")

            # Convert from a CMake ;-list to a string
            string (REPLACE ";" " " _link_flags_str "${_link_flags}")
            set_target_properties(${arg_TO}
                                  PROPERTIES LINK_FLAGS "${_link_flags_str}")
        endif()
    endif()

    unset(_flags)
    unset(_link_flags)
    unset(_link_flags_str)
    unset(_scope)

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
## Registers a library to the project to ease use in other BLT macro calls.
##------------------------------------------------------------------------------
macro(blt_register_library)

    set(singleValueArgs NAME OBJECT TREAT_INCLUDES_AS_SYSTEM)
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
        set(BLT_${uppercase_name}_DEPENDS_ON ${arg_DEPENDS_ON} CACHE STRING "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_DEPENDS_ON)
    endif()

    if( arg_INCLUDES )
        set(BLT_${uppercase_name}_INCLUDES ${arg_INCLUDES} CACHE STRING "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_INCLUDES)
    endif()

    if( ${arg_OBJECT} )
        set(BLT_${uppercase_name}_IS_OBJECT_LIBRARY TRUE CACHE BOOL "" FORCE)
    else()
        set(BLT_${uppercase_name}_IS_OBJECT_LIBRARY FALSE CACHE BOOL "" FORCE)
    endif()
    mark_as_advanced(BLT_${uppercase_name}_IS_OBJECT_LIBRARY)

    if( ${arg_TREAT_INCLUDES_AS_SYSTEM} )
        set(BLT_${uppercase_name}_TREAT_INCLUDES_AS_SYSTEM TRUE CACHE BOOL "" FORCE)
    else()
        set(BLT_${uppercase_name}_TREAT_INCLUDES_AS_SYSTEM FALSE CACHE BOOL "" FORCE)
    endif()
    mark_as_advanced(BLT_${uppercase_name}_TREAT_INCLUDES_AS_SYSTEM)

    if( ENABLE_FORTRAN AND arg_FORTRAN_MODULES )
        set(BLT_${uppercase_name}_FORTRAN_MODULES ${arg_INCLUDES} CACHE STRING "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_FORTRAN_MODULES)
    endif()

    if( arg_LIBRARIES )
        set(BLT_${uppercase_name}_LIBRARIES ${arg_LIBRARIES} CACHE STRING "" FORCE)
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
        set(BLT_${uppercase_name}_DEFINES ${arg_DEFINES} CACHE STRING "" FORCE)
        mark_as_advanced(BLT_${uppercase_name}_DEFINES)
    endif()

endmacro(blt_register_library)


##------------------------------------------------------------------------------
## blt_add_library( NAME         <libname>
##                  SOURCES      [source1 [source2 ...]]
##                  HEADERS      [header1 [header2 ...]]
##                  INCLUDES     [dir1 [dir2 ...]]
##                  DEFINES      [define1 [define2 ...]]
##                  DEPENDS_ON   [dep1 ...] 
##                  OUTPUT_NAME  [name]
##                  OUTPUT_DIR   [dir]
##                  SHARED       [TRUE | FALSE]
##                  OBJECT       [TRUE | FALSE]
##                  CLEAR_PREFIX [TRUE | FALSE]
##                  FOLDER       [name])
##
## Adds a library target, called <libname>, to be built from the given sources.
##------------------------------------------------------------------------------
macro(blt_add_library)

    set(options)
    set(singleValueArgs NAME OUTPUT_NAME OUTPUT_DIR HEADERS_OUTPUT_SUBDIR SHARED OBJECT CLEAR_PREFIX FOLDER)
    set(multiValueArgs SOURCES HEADERS INCLUDES DEFINES DEPENDS_ON)

    # parse the arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Sanity checks
    if( "${arg_NAME}" STREQUAL "" )
        message(FATAL_ERROR "blt_add_library() must be called with argument NAME <name>")
    endif()

    if (NOT arg_SOURCES AND NOT arg_HEADERS )
        message(FATAL_ERROR "blt_add_library(NAME ${arg_NAME} ...) called with no given sources or headers")
    endif()

    if (DEFINED arg_OBJECT AND arg_OBJECT)
        if (DEFINED arg_SHARED AND arg_SHARED)
                message(FATAL_ERROR "blt_add_library(NAME ${arg_NAME} ...) cannot be called with both OBJECT and SHARED set to TRUE.")
        endif()

        if (NOT arg_SOURCES)
            message(FATAL_ERROR "blt_add_library(NAME ${arg_NAME} ...) cannot create an object library with no sources.")
        endif()
    endif()

    if ( arg_SOURCES )
        # Determine type of library to build. STATIC by default and OBJECT takes
        # precedence over global BUILD_SHARED_LIBS variable.
        set(_build_shared_library ${BUILD_SHARED_LIBS})
        if( DEFINED arg_SHARED )
            set(_build_shared_library ${arg_SHARED})
        endif()

        if ( ${arg_OBJECT} )
            set(_lib_type "OBJECT")
            blt_register_library( NAME       ${arg_NAME}
                                  DEPENDS_ON ${arg_DEPENDS_ON}
                                  INCLUDES   ${arg_INCLUDES}
                                  OBJECT     TRUE
                                  DEFINES    ${arg_DEFINES} )
        elseif ( ${_build_shared_library} )
            set(_lib_type "SHARED")
        else()
            set(_lib_type "STATIC")
        endif()

        if (ENABLE_HIP)
            blt_add_hip_library(NAME         ${arg_NAME}
                                SOURCES      ${arg_SOURCES}
                                HEADERS      ${arg_HEADERS}
                                DEPENDS_ON   ${arg_DEPENDS_ON}
                                LIBRARY_TYPE ${_lib_type} )
        else()
            add_library( ${arg_NAME} ${_lib_type} ${arg_SOURCES} ${arg_HEADERS} )

            if (ENABLE_CUDA AND NOT ENABLE_CLANG_CUDA)
                blt_setup_cuda_target(
                    NAME         ${arg_NAME}
                    SOURCES      ${arg_SOURCES}
                    DEPENDS_ON   ${arg_DEPENDS_ON}
                    LIBRARY_TYPE ${_lib_type})
            endif()
        endif()
    else()
        #
        #  Header-only library support
        #
        foreach (_file ${arg_HEADERS})
            # Determine build location of headers
            get_filename_component(_absolute ${_file} ABSOLUTE)
            list(APPEND _build_headers ${_absolute})
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

    blt_setup_target( NAME       ${arg_NAME}
                      DEPENDS_ON ${arg_DEPENDS_ON} 
                      OBJECT     ${arg_OBJECT})

    if ( arg_INCLUDES )
        if (NOT arg_SOURCES )
            # Header only
            target_include_directories(${arg_NAME} INTERFACE ${arg_INCLUDES})
        else()
            target_include_directories(${arg_NAME} PUBLIC ${arg_INCLUDES})
        endif()
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

    if ( arg_SOURCES )
        # Don't clean header-only libraries because you would have to handle
        # the white-list of properties that are allowed
        blt_clean_target(TARGET ${arg_NAME})
    endif()

endmacro(blt_add_library)


##------------------------------------------------------------------------------
## blt_add_executable( NAME       <name>
##                     SOURCES    [source1 [source2 ...]]
##                     INCLUDES   [dir1 [dir2 ...]]
##                     DEFINES    [define1 [define2 ...]]
##                     DEPENDS_ON [dep1 [dep2 ...]]
##                     OUTPUT_DIR [dir]
##                     FOLDER     [name])
##
## Adds an executable target, called <name>, to be built from the given sources.
##------------------------------------------------------------------------------
macro(blt_add_executable)

    set(options )
    set(singleValueArgs NAME OUTPUT_DIR FOLDER)
    set(multiValueArgs SOURCES INCLUDES DEFINES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # Sanity checks
    if( "${arg_NAME}" STREQUAL "" )
        message(FATAL_ERROR "blt_add_executable() must be called with argument NAME <name>")
    endif()

    if (NOT arg_SOURCES )
        message(FATAL_ERROR "blt_add_executable(NAME ${arg_NAME} ...) given with no sources")
    endif()

    if (ENABLE_HIP)
        blt_add_hip_executable(NAME         ${arg_NAME}
                               SOURCES      ${arg_SOURCES}
                               DEPENDS_ON   ${arg_DEPENDS_ON})
    else()
        add_executable( ${arg_NAME} ${arg_SOURCES} )

        if (ENABLE_CUDA AND NOT ENABLE_CLANG_CUDA)
            blt_setup_cuda_target(
                NAME         ${arg_NAME}
                SOURCES      ${arg_SOURCES}
                DEPENDS_ON   ${arg_DEPENDS_ON})
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
       
    blt_setup_target(NAME       ${arg_NAME}
                     DEPENDS_ON ${arg_DEPENDS_ON} 
                     OBJECT     FALSE)

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

    blt_clean_target(TARGET ${arg_NAME})

endmacro(blt_add_executable)


##------------------------------------------------------------------------------
## blt_add_test( NAME           [name]
##               COMMAND        [command] 
##               NUM_MPI_TASKS  [n]
##               CONFIGURATIONS [config1 [config2...]])
##
## Adds a test to the project.
##------------------------------------------------------------------------------
macro(blt_add_test)

    set(options )
    set(singleValueArgs NAME NUM_MPI_TASKS)
    set(multiValueArgs COMMAND CONFIGURATIONS)

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
        # Handle CMake changing MPIEXEC to MPIEXEC_EXECUTABLE
        if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
            set(_mpiexec ${MPIEXEC_EXECUTABLE})
        else()
            set(_mpiexec ${MPIEXEC})
        endif()

        set(test_command ${_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_MPI_TASKS} ${BLT_MPI_COMMAND_APPEND} ${test_command} )
    endif()

    add_test(NAME           ${arg_NAME}
             COMMAND        ${test_command}
             CONFIGURATIONS ${arg_CONFIGURATIONS})

endmacro(blt_add_test)


##------------------------------------------------------------------------------
## blt_add_benchmark( NAME          [name] 
##                    COMMAND       [command]
##                    NUM_MPI_TASKS [n])
##
## Adds a benchmark to the project.
##------------------------------------------------------------------------------
macro(blt_add_benchmark)

    set(options)
    set(singleValueArgs NAME NUM_MPI_TASKS)      
    set(multiValueArgs COMMAND)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
     "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # The 'CONFIGURATIONS Benchmark' line excludes benchmarks 
    # from the general list of tests
    blt_add_test( NAME           ${arg_NAME}
                  COMMAND        ${arg_COMMAND}
                  NUM_MPI_TASKS  ${arg_NUM_MPI_TASKS}
                  CONFIGURATIONS Benchmark)

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
##                    PGI        pgiFlag        (optional))
##
## Appends compiler-specific flags to a given variable of flags
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
##                     NAMES      [libname1 [libname2 ...]]
##                     REQUIRED   [TRUE (default) | FALSE ]
##                     PATHS      [path1 [path2 ...]] )
##
## This command is used to find a list of libraries.
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
##                               LINK_POSTPEND [])
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

    # Sanity checks
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
    blt_list_remove_duplicates(TO interface_include_directories )
    blt_list_remove_duplicates(TO interface_system_include_directories )

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
## blt_print_target_properties(TARGET <target> )
##
## Prints out all properties of the given target.
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
        blt_list_remove_duplicates(TO _property_list)   

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
            if(prop MATCHES "^${_target_prefix}")
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
