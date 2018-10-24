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

include(CMakeParseArguments)

## Internal BLT CMake Macros


##-----------------------------------------------------------------------------
## blt_error_if_target_exists()
##
## Checks if target already exists in CMake project and errors out with given
## error_msg.
##-----------------------------------------------------------------------------
function(blt_error_if_target_exists target_name error_msg)
    if (TARGET ${target_name})
        message(FATAL_ERROR "${error_msg}Duplicate target name: ${target_name}")
    endif()
endfunction()


################################################################################
# blt_find_executable(NAME         <name of program to find>
#                     EXECUTABLES  [exe1 [exe2 ...]])
#
# This macro attempts to find the given executable via either a previously defined
# <UPPERCASE_NAME>_EXECUTABLE or using find_program with the given EXECUTABLES.
# if EXECUTABLES is left empty, then NAME is used.
#
# If successful the following variables will be defined:
# <UPPERCASE_NAME>_FOUND
# <UPPERCASE_NAME>_EXECUTABLE
################################################################################
macro(blt_find_executable)

    set(options)
    set(singleValueArgs NAME)
    set(multiValueArgs  EXECUTABLES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_find_executable' macro" )
    endif()

    string(TOUPPER ${arg_NAME} _ucname)

    message(STATUS "${arg_NAME} support is ${ENABLE_${_ucname}}")
    if (${ENABLE_${_ucname}})
        set(_exes ${arg_NAME})
        if (DEFINED arg_EXECUTABLES)
            set(_exes ${arg_EXECUTABLES})
        endif()

        if (${_ucname}_EXECUTABLE)
            if (NOT EXISTS ${${_ucname}_EXECUTABLE})
                message(FATAL_ERROR "User defined ${_ucname}_EXECUTABLE does not exist. Fix/unset variable or set ENABLE_${_ucname} to OFF.")
            endif()
        else()
            find_program(${_ucname}_EXECUTABLE
                         NAMES ${_exes}
                         DOC "Path to ${arg_NAME} executable")
        endif()

        # Handle REQUIRED and QUIET arguments
        # this will also set ${_ucname}_FOUND to true if ${_ucname}_EXECUTABLE exists
        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(${arg_NAME}
                                          "Failed to locate ${arg_NAME} executable"
                                          ${_ucname}_EXECUTABLE)
    endif()
endmacro(blt_find_executable)


##------------------------------------------------------------------------------
## blt_setup_target( NAME [name] DEPENDS_ON [dep1 ...] )
##------------------------------------------------------------------------------
macro(blt_setup_target)

    set(options)
    set(singleValueArgs NAME)
    set(multiValueArgs DEPENDS_ON)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_NAME )
        message( FATAL_ERROR "Must provide a NAME argument to the 'blt_setup_target' macro" )
    endif()

    # Expand dependency list
    set(_expanded_DEPENDS_ON ${arg_DEPENDS_ON})
    foreach( i RANGE 50 )
        foreach( dependency ${_expanded_DEPENDS_ON} )
            string(TOUPPER ${dependency} uppercase_dependency )

            if ( DEFINED BLT_${uppercase_dependency}_DEPENDS_ON )
                foreach(new_dependency ${BLT_${uppercase_dependency}_DEPENDS_ON})
                    if (NOT ${new_dependency} IN_LIST _expanded_DEPENDS_ON)
                        list(APPEND _expanded_DEPENDS_ON ${new_dependency})
                    endif()
                endforeach()
            endif()
        endforeach()
    endforeach()

    # Add dependency's information
    foreach( dependency ${_expanded_DEPENDS_ON} )
        string(TOUPPER ${dependency} uppercase_dependency )

        if ( DEFINED BLT_${uppercase_dependency}_INCLUDES )
            if ( BLT_${uppercase_dependency}_TREAT_INCLUDES_AS_SYSTEM )
                target_include_directories( ${arg_NAME} SYSTEM PUBLIC
                    ${BLT_${uppercase_dependency}_INCLUDES} )
            else()
                target_include_directories( ${arg_NAME} PUBLIC
                    ${BLT_${uppercase_dependency}_INCLUDES} )
            endif()
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_FORTRAN_MODULES )
            target_include_directories( ${arg_NAME} PUBLIC
                ${BLT_${uppercase_dependency}_FORTRAN_MODULES} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_LIBRARIES)
            # This prevents cmake from adding -l<library name> to the
            # command line for BLT registered libraries which are not
            # actual CMake targets
            if(NOT "${BLT_${uppercase_dependency}_LIBRARIES}"
                    STREQUAL "BLT_NO_LIBRARIES" )
                target_link_libraries( ${arg_NAME} PUBLIC
                    ${BLT_${uppercase_dependency}_LIBRARIES} )
            endif()
        else()
            target_link_libraries( ${arg_NAME} PUBLIC ${dependency} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_DEFINES )
            target_compile_definitions( ${arg_NAME} PUBLIC
                ${BLT_${uppercase_dependency}_DEFINES} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_COMPILE_FLAGS )
            blt_add_target_compile_flags(TO ${arg_NAME}
                                         FLAGS ${BLT_${uppercase_dependency}_COMPILE_FLAGS} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_LINK_FLAGS )
            blt_add_target_link_flags(TO ${arg_NAME}
                                      FLAGS ${BLT_${uppercase_dependency}_LINK_FLAGS} )
        endif()

    endforeach()

endmacro(blt_setup_target)

##------------------------------------------------------------------------------
## blt_setup_cuda_source_properties(BUILD_TARGET TARGET_SOURCES <sources>)
##------------------------------------------------------------------------------
macro(blt_setup_cuda_source_properties)

    set(options)
    set(singleValueArgs BUILD_TARGET)
    set(multiValueArgs TARGET_SOURCES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_BUILD_TARGET )
        message( FATAL_ERROR "Must provide a BUILD_TARGET argument to the 'blt_setup_cuda_source_properties' macro")
    endif()

    if ( NOT DEFINED arg_TARGET_SOURCES )
        message( FATAL_ERROR "Must provide TARGET_SOURCES to the 'blt_setup_cuda_source_properties' macro")
    endif()

    set(_cuda_sources)
    set(_non_cuda_sources)
    blt_split_source_list_by_language(SOURCES      ${arg_TARGET_SOURCES}
                                      C_LIST       _cuda_sources
                                      Fortran_LIST _non_cuda_sources)

    set_source_files_properties( ${_cuda_sources}
                                 PROPERTIES
                                 LANGUAGE CUDA)

    if (CUDA_SEPARABLE_COMPILATION)
        set_source_files_properties( ${_cuda_sources}
                                     PROPERTIES
                                     CUDA_SEPARABLE_COMPILATION ON)
    endif()

    #
    # for debugging, or if we add verbose BLT output
    #
    ##message(STATUS "target '${arg_BUILD_TARGET}' CUDA Sources: ${_cuda_sources}")
    ##message(STATUS "target '${arg_BUILD_TARGET}' non-CUDA Sources: ${_non_cuda_sources}")

endmacro(blt_setup_cuda_source_properties)


##------------------------------------------------------------------------------
## blt_split_source_list_by_language( SOURCES <sources>
##                                    C_LIST <list name>
##                                    Fortran_LIST <list name>)
##------------------------------------------------------------------------------
macro(blt_split_source_list_by_language)

    set(options)
    set(singleValueArgs C_LIST Fortran_LIST)
    set(multiValueArgs SOURCES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_SOURCES )
        message( FATAL_ERROR "Must provide a SOURCES argument to the 'blt_split_source_list_by_language' macro" )
    endif()

    # Generate source lists based on language
    foreach(_file ${arg_SOURCES})
        get_filename_component(_ext ${_file} EXT)
        string(TOLOWER ${_ext} _ext_lower)

        if(${_ext_lower} IN_LIST BLT_C_FILE_EXTS)
            if (DEFINED arg_C_LIST)
                list(APPEND ${arg_C_LIST} ${_file})
            endif()
        elseif(${_ext_lower} IN_LIST BLT_Fortran_FILE_EXTS)
            if (DEFINED arg_Fortran_LIST)
                list(APPEND ${arg_Fortran_LIST} ${_file})
            endif()
        else()
            message(FATAL_ERROR "blt_split_source_list_by_language given source file with unknown file extension. Add the missing extension to the corresponding list (BLT_C_FILE_EXTS or BLT_Fortran_FILE_EXTS).\n Unknown file: ${_file}")
        endif()
    endforeach()

endmacro(blt_split_source_list_by_language)


##------------------------------------------------------------------------------
## blt_update_project_sources( TARGET_SOURCES <sources> )
##------------------------------------------------------------------------------
macro(blt_update_project_sources)

    set(options)
    set(singleValueArgs)
    set(multiValueArgs TARGET_SOURCES)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_TARGET_SOURCES )
        message( FATAL_ERROR "Must provide target sources" )
    endif()

    ## append the target source to the all project sources
    foreach( src ${arg_TARGET_SOURCES} )
        if(IS_ABSOLUTE ${src})
            list(APPEND "${PROJECT_NAME}_ALL_SOURCES" "${src}")
        else()
            list(APPEND "${PROJECT_NAME}_ALL_SOURCES"
                "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        endif()
    endforeach()

    set( "${PROJECT_NAME}_ALL_SOURCES" "${${PROJECT_NAME}_ALL_SOURCES}"
        CACHE STRING "" FORCE )
    mark_as_advanced("${PROJECT_NAME}_ALL_SOURCES")

endmacro(blt_update_project_sources)

##------------------------------------------------------------------------------
## blt_filter_list( TO <list_var> REGEX <string> OPERATION <string> )
##
## This macro provides the same functionality as cmake's list(FILTER )
## which is only available in cmake-3.6+.
##
## The TO argument (required) is the name of a list variable.
## The REGEX argument (required) is a string containing a regex.
## The OPERATION argument (required) is a string that defines the macro's operation.
## Supported values are "include" and "exclude"
##
## The filter is applied to the input list, which is modified in place.
##------------------------------------------------------------------------------
macro(blt_filter_list)

    set(options )
    set(singleValueArgs TO REGEX OPERATION)
    set(multiValueArgs )

    # Parse arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}" 
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if( NOT DEFINED arg_TO )
        message(FATAL_ERROR "blt_filter_list macro requires a TO <list> argument")
    endif()

    if( NOT DEFINED arg_REGEX )
        message(FATAL_ERROR "blt_filter_list macro requires a REGEX <string> argument")
    endif()

    # Ensure OPERATION argument is provided with value "include" or "exclude"
    set(_exclude)
    if( NOT DEFINED arg_OPERATION )
        message(FATAL_ERROR "blt_filter_list macro requires a OPERATION <string> argument")
    elseif(NOT arg_OPERATION MATCHES "^(include|exclude)$")
        message(FATAL_ERROR "blt_filter_list macro's OPERATION argument must be either 'include' or 'exclude'")
    else()
        if(${arg_OPERATION} MATCHES "exclude")
            set(_exclude TRUE)
        else()
            set(_exclude FALSE)
        endif()
    endif()

    # Filter the list
    set(_resultList)
    foreach(elem ${${arg_TO}})
        if(elem MATCHES ${arg_REGEX})
            if(NOT ${_exclude})
                list(APPEND _resultList ${elem})
            endif()
        else()
            if(${_exclude})
                list(APPEND _resultList ${elem})
            endif()
        endif()
    endforeach()

    # Copy result back to input list variable
    set(${arg_TO} ${_resultList})

    unset(_exclude)
    unset(_resultList)
endmacro(blt_filter_list)

