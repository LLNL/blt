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
        message( FATAL_ERROR "Must provide a NAME argument to the macro" )
    endif()

    # Add it's own copy headers target
    if (ENABLE_COPY_HEADERS AND TARGET "blt_copy_headers_${arg_NAME}")
        add_dependencies( ${arg_NAME} "blt_copy_headers_${arg_NAME}")
    endif()

    # Add dependency's information
    foreach( dependency ${arg_DEPENDS_ON} )
        string(TOUPPER ${dependency} uppercase_dependency )

        if ( DEFINED BLT_${uppercase_dependency}_INCLUDES )
            target_include_directories( ${arg_NAME} PRIVATE
                ${BLT_${uppercase_dependency}_INCLUDES} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_FORTRAN_MODULES )
            target_include_directories( ${arg_NAME} PRIVATE
                ${BLT_${uppercase_dependency}_FORTRAN_MODULES} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_LIBRARIES)
            if(NOT "${BLT_${uppercase_dependency}_LIBRARIES}"
                    STREQUAL "BLT_NO_LIBRARIES" )
                target_link_libraries( ${arg_NAME}
                    ${BLT_${uppercase_dependency}_LIBRARIES} )
            endif()
        else()
            target_link_libraries( ${arg_NAME} ${dependency} )
        endif()

        if ( DEFINED BLT_${uppercase_dependency}_DEFINES )
            target_compile_definitions( ${arg_NAME} PRIVATE
                ${BLT_${uppercase_dependency}_DEFINES} )
        endif()

        if (ENABLE_COPY_HEADERS AND TARGET "blt_copy_headers_${dependency}")
            add_dependencies( ${arg_NAME} "blt_copy_headers_${dependency}")
        endif()

    endforeach()

endmacro(blt_setup_target)


##------------------------------------------------------------------------------
## blt_setup_mpi_target( BUILD_TARGET <target> )
##------------------------------------------------------------------------------
macro(blt_setup_mpi_target)

    set(options)
    set(singleValueArgs BUILD_TARGET)
    set(multiValueArgs)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}" 
                            "${multiValueArgs}" ${ARGN} )
                            
    # Check arguments
    if ( NOT DEFINED arg_BUILD_TARGET )
        message( FATAL_ERROR "Must provide a BUILD_TARGET argument to the macro" )
    endif()

    if ( ${ENABLE_MPI} )
        blt_add_target_definitions( TO ${arg_BUILD_TARGET} TARGET_DEFINITIONS USE_MPI )

        target_include_directories( ${arg_BUILD_TARGET} 
                                    PUBLIC ${MPI_C_INCLUDE_PATH} )

        target_include_directories( ${arg_BUILD_TARGET} 
                                    PUBLIC ${MPI_CXX_INCLUDE_PATH} )

        target_include_directories( ${arg_BUILD_TARGET} 
                                    PUBLIC ${MPI_Fortran_INCLUDE_PATH} )

        if ( NOT "${MPI_C_COMPILE_FLAGS}" STREQUAL "")
            set_target_properties( ${arg_BUILD_TARGET} 
                PROPERTIES COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS} )
        endif()

        if ( NOT "${MPI_C_LINK_FLAGS}" STREQUAL "")
            set_target_properties( ${arg_BUILD_TARGET} 
                PROPERTIES LINK_FLAGS ${MPI_C_LINK_FLAGS} )
        endif()

        if ( NOT "${MPI_Fortran_LINK_FLAGS}" STREQUAL "" )
            set_target_properties( ${arg_BUILD_TARGET} 
                PROPERTIES LINK_FLAGS ${MPI_Fortran_LINK_FLAGS} )
        endif()

        target_link_libraries( ${arg_BUILD_TARGET} ${MPI_C_LIBRARIES} )
        target_link_libraries( ${arg_BUILD_TARGET} ${MPI_CXX_LIBRARIES} )
        target_link_libraries( ${arg_BUILD_TARGET} ${MPI_Fortran_LIBRARIES} )
    endif()

endmacro(blt_setup_mpi_target)


##------------------------------------------------------------------------------
## blt_setup_openmp_target( TARGET <target> USE_OPENMP <bool> )
##------------------------------------------------------------------------------
macro(blt_setup_openmp_target)

    set(options)
    set(singleValueArgs BUILD_TARGET USE_OPENMP)
    set(multiValueArgs)

    # Parse the arguments
    cmake_parse_arguments(arg "${options}" "${singleValueArgs}" 
                            "${multiValueArgs}" ${ARGN} )

    # Check arguments
    if ( NOT DEFINED arg_BUILD_TARGET )
        message ( FATAL_ERROR "Must provide a BUILD_TARGET argument to the macro")
    endif()

    if ( NOT DEFINED arg_USE_OPENMP )
        message( FATAL_ERROR "Must provide an OpenMP boolean flag")
    endif()

    if ( ${arg_USE_OPENMP} AND NOT ${ENABLE_OPENMP} )
        message( FATAL_ERROR "Building an OpenMP library, but OpenMP is disabled!")
    endif()

    if ( ${arg_USE_OPENMP} )
        blt_add_target_definitions( TO ${arg_BUILD_TARGET}
                                TARGET_DEFINITIONS USE_OPENMP )
        set_target_properties( ${arg_BUILD_TARGET}
                               PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS} )
        set_target_properties( ${arg_BUILD_TARGET}
                               PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS} )
    endif()

endmacro(blt_setup_openmp_target)

##------------------------------------------------------------------------------
## update_project_sources( TARGET_SOURCES <souces> )
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

endmacro(blt_update_project_sources)
