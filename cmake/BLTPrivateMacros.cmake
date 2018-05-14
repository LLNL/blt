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


##------------------------------------------------------------------------------
## blt_copy_headers_target( NAME [name] HEADERS [hdr1 ...] DESTINATION [destination] )
##
## Adds a custom "copy_headers" target for the given project
##
## Adds a custom target, blt_copy_headers_[NAME], for the given project. 
## The role of this target is to copy the given list of headers, [HEADERS], to 
## the destination directory [DESTINATION].
##
## This macro is used to copy the header of each component in to the build
## space, under an "includes" directory.
##------------------------------------------------------------------------------
macro(blt_copy_headers_target)

    set(options)
    set(singleValueArgs NAME DESTINATION)
    set(multiValueArgs HEADERS)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}" 
                        "${multiValueArgs}" ${ARGN} )
                        
    # Make all headers paths absolute
    set(temp_list "")
    foreach(header ${arg_HEADERS})
        list(APPEND temp_list ${CMAKE_CURRENT_LIST_DIR}/${header})
    endforeach()
    set(arg_HEADERS ${temp_list})

    add_custom_target(blt_copy_headers_${arg_NAME}
        COMMAND ${CMAKE_COMMAND}
                 -DOUTPUT_DIRECTORY=${arg_DESTINATION}
                 -DLIBHEADERS="${arg_HEADERS}"
                 -P ${BLT_ROOT_DIR}/cmake/copy_headers.cmake

        DEPENDS
            ${arg_HEADERS}

        WORKING_DIRECTORY
            ${PROJECT_SOURCE_DIR}

        COMMENT
            "copy headers ${arg_NAME}"
        )

endmacro(blt_copy_headers_target)

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

    # Add it's own copy headers target
    if (ENABLE_COPY_HEADERS AND TARGET "blt_copy_headers_${arg_NAME}")
        add_dependencies( ${arg_NAME} "blt_copy_headers_${arg_NAME}")
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

        if (ENABLE_COPY_HEADERS AND TARGET "blt_copy_headers_${dependency}")
            add_dependencies( ${arg_NAME} "blt_copy_headers_${dependency}")
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


    foreach (_file ${arg_TARGET_SOURCES})
      get_source_file_property(_lang ${_file} LANGUAGE)

      if (${_lang} MATCHES Fortran OR ${_lang} MATCHES C)
            set(_non_cuda_sources ${_non_cuda_sources} ${_file})
        else()
            set(_cuda_sources ${_cuda_sources} ${_file})
        endif()
    endforeach()

    set_source_files_properties( ${_cuda_sources}
                                 PROPERTIES
                                 CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    set_source_files_properties( ${_non_cuda_sources}
                                 PROPERTIES
                                 CUDA_SOURCE_PROPERTY_FORMAT False)

    #
    # for debugging, or if we add verbose BLT output
    #
    ##message(STATUS "target '${arg_BUILD_TARGET}' CUDA Sources: ${_cuda_sources}")
    ##message(STATUS "target '${arg_BUILD_TARGET}' non-CUDA Sources: ${_non_cuda_sources}")

endmacro(blt_setup_cuda_source_properties)


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

