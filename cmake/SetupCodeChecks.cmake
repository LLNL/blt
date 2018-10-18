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

###############################################################################
# Targets related to source code checks (formatting, static analysis, etc)
###############################################################################

add_custom_target(check)
add_custom_target(style)

if(UNCRUSTIFY_FOUND)
    # targets for verifying formatting
    add_custom_target(uncrustify_check)
    add_dependencies(check uncrustify_check)

    # targets for modifying formatting
    add_custom_target(uncrustify_style)
    add_dependencies(style uncrustify_style)
endif()

if(ASTYLE_FOUND)
    # targets for verifying formatting
    add_custom_target(astyle_check)
    add_dependencies(check astyle_check)

    # targets for modifying formatting
    add_custom_target(astyle_style)
    add_dependencies(style astyle_style)
endif()

if(CPPCHECK_FOUND)
    add_custom_target(cppcheck_check)
    add_dependencies(check cppcheck_check)
endif()

# Code check targets should only be run on demand
foreach(target 
        check uncrustify_check astyle_check cppcheck_check
        style uncrustify_style astyle_style )
    if(TARGET ${target})
        set_property(TARGET ${target} PROPERTY EXCLUDE_FROM_ALL TRUE)
        set_property(TARGET ${target} PROPERTY EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    endif()
endforeach()


##------------------------------------------------------------------------------
## blt_add_code_checks( PREFIX              <Base name used for created targets>
##                      SOURCES             [source1 [source2 ...]]
##                      UNCRUSTIFY_CFG_FILE <path to uncrustify config file>
##                      ASTYLE_CFG_FILE     <path to astyle config file>)
##
## This macro adds all enabled code check targets for the given SOURCES. It
## filters based on file extensions.
##
## PREFIX is used in the creation of all the underlying targets. For example:
## <PREFIX>_uncrustify_check.
##
## Sources are filtered based on file extensions for use in these code checks.  If you need
## additional file extensions defined add them to BLT_C_FILE_EXTS and BLT_Fortran_FILE_EXTS.
##
## This macro supports formatting with either Uncrustify or AStyle (but not both) using
## the following parameters:
##
## * UNCRUSTIFY_CFG_FILE is the configuration file for Uncrustify. If UNCRUSTIFY_EXECUTABLE
##   is defined, found, and UNCRUSTIFY_CFG_FILE is provided it will create both check and
##   style function for the given C/C++ files.
##
## * ASTYLE_CFG_FILE is the configuration file for AStyle. If ASTYLE_EXECUTABLE
##   is defined, found, and ASTYLE_CFG_FILE is provided it will create both check and
##   style function for the given C/C++ files.
##
##------------------------------------------------------------------------------

macro(blt_add_code_checks)

    set(options )
    set(singleValueArgs PREFIX UNCRUSTIFY_CFG_FILE ASTYLE_CFG_FILE)
    set(multiValueArgs SOURCES)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED arg_PREFIX)
        message(FATAL_ERROR "blt_add_code_checks requires the parameter PREFIX.")
    endif()

    if (NOT DEFINED arg_SOURCES)
        message(FATAL_ERROR "blt_add_code_checks requires the parameter SOURCES.")
    endif()

    # Make the sources relative to the bin directory
    set(_rel_sources)
    foreach(_file ${arg_SOURCES})
        # Get full path
        if(IS_ABSOLUTE ${_file})
            set(_full_path ${_file})
        else()
            set(_full_path ${CMAKE_CURRENT_SOURCE_DIR}/${_file})
        endif()

        file(RELATIVE_PATH _rel_path ${CMAKE_BINARY_DIR} ${_full_path})
        list(APPEND _rel_sources ${_rel_path})
    endforeach()   

    # Generate source lists based on language
    set(_c_sources)
    set(_f_sources)
    blt_split_source_list_by_language(SOURCES      ${_rel_sources}
                                      C_LIST       _c_sources
                                      Fortran_LIST _f_sources)

    # Check that at most one formatting config file was supplied
    if (DEFINED arg_UNCRUSTIFY_CFG_FILE AND DEFINED arg_ASTYLE_CFG_FILE)
        message(FATAL_ERROR 
          "blt_add_code_checks macro does not support multiple "
          "style config parameters within the same invocation. "
          "Both UNCRUSTIFY_CFG_FILE and ASTYLE_CFG_FILE were supplied.")
    endif()

    # Add code checks
    set(_error_msg "blt_add_code_checks tried to create an already existing target with given PREFIX: ${arg_PREFIX}. ")
    if (UNCRUSTIFY_FOUND AND DEFINED arg_UNCRUSTIFY_CFG_FILE)
        set(_check_target_name ${arg_PREFIX}_uncrustify_check)
        blt_error_if_target_exists(${_check_target_name} ${_error_msg})
        set(_style_target_name ${arg_PREFIX}_uncrustify_style)
        blt_error_if_target_exists(${_style_target_name} ${_error_msg})

        blt_add_uncrustify_target( NAME              ${_check_target_name}
                                   MODIFY_FILES      FALSE
                                   CFG_FILE          ${arg_UNCRUSTIFY_CFG_FILE} 
                                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                                   SRC_FILES         ${_c_sources} )

        blt_add_uncrustify_target( NAME              ${_style_target_name}
                                   MODIFY_FILES      TRUE
                                   CFG_FILE          ${arg_UNCRUSTIFY_CFG_FILE} 
                                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                                   SRC_FILES         ${_c_sources} )
    endif()

    if (ASTYLE_FOUND AND DEFINED arg_ASTYLE_CFG_FILE)
        set(_check_target_name ${arg_PREFIX}_astyle_check)
        blt_error_if_target_exists(${_check_target_name} ${_error_msg})
        set(_style_target_name ${arg_PREFIX}_astyle_style)
        blt_error_if_target_exists(${_style_target_name} ${_error_msg})

        blt_add_astyle_target( NAME              ${_check_target_name}
                               MODIFY_FILES      FALSE
                               CFG_FILE          ${arg_ASTYLE_CFG_FILE} 
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               SRC_FILES         ${_c_sources} )

        blt_add_astyle_target( NAME              ${_style_target_name}
                               MODIFY_FILES      TRUE
                               CFG_FILE          ${arg_ASTYLE_CFG_FILE} 
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                               SRC_FILES         ${_c_sources} )
    endif()

    if (CPPCHECK_FOUND)
        set(_cppcheck_target_name ${arg_PREFIX}_cppcheck_check)
        blt_error_if_target_exists(${_cppcheck_target_name} ${_error_msg})

        blt_add_cppcheck_target( NAME              ${_cppcheck_target_name}
                                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                                 SRC_FILES         ${_c_sources})
    endif()

endmacro(blt_add_code_checks)


##-----------------------------------------------------------------------------
## blt_add_cppcheck_target( NAME                <Created Target Name>
##                          WORKING_DIRECTORY   <Working Directory>
##                          PREPEND_FLAGS       <additional flags for cppcheck>
##                          APPEND_FLAGS        <additional flags for cppcheck>
##                          COMMENT             <Additional Comment for Target Invocation>
##                          SRC_FILES           [FILE1 [FILE2 ...]] )
##
## Creates a new target with the given NAME for running cppcheck over the given SRC_FILES
##
## PREPEND_FLAGS are additional flags given to added to the front of the uncrustify flags.
##
## APPEND_FLAGS are additional flags given to added to the end of the uncrustify flags.
##
## COMMENT is prepended to the commented outputted by CMake.
##
## WORKING_DIRECTORY is the directory that uncrustify will be ran.  It defaults to the directory
## where this macro is called.
##
## SRC_FILES is a list of source files that cppcheck will be run on.
##-----------------------------------------------------------------------------
macro(blt_add_cppcheck_target)

    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs NAME COMMENT WORKING_DIRECTORY)
    set(multiValueArgs SRC_FILES PREPEND_FLAGS APPEND_FLAGS)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Check required parameters
    if(NOT DEFINED arg_NAME)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a NAME parameter")
    endif()

    if(NOT DEFINED arg_SRC_FILES)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a SRC_FILES parameter")
    endif()

    if(DEFINED arg_WORKING_DIRECTORY)
        set(_wd ${arg_WORKING_DIRECTORY})
    else()
        set(_wd ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    add_custom_target(${arg_NAME}
            COMMAND ${CPPCHECK_EXECUTABLE} ${arg_PREPEND_FLAGS} ${arg_SRC_FILES} ${arg_APPEND_FLAGS}
            WORKING_DIRECTORY ${_wd}
            COMMENT "${arg_COMMENT}Running cppcheck source code static analysis checks.")

    # hook our new target into the proper dependency chain
    add_dependencies(cppcheck_check ${arg_NAME})

    # Code check targets should only be run on demand
    set_property(TARGET ${arg_NAME} PROPERTY EXCLUDE_FROM_ALL TRUE)
    set_property(TARGET ${arg_NAME} PROPERTY EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endmacro(blt_add_cppcheck_target)


##------------------------------------------------------------------------------
## blt_add_uncrustify_target( NAME              <Created Target Name>
##                            MODIFY_FILES      [TRUE | FALSE (default)]
##                            CFG_FILE          <Uncrustify Configuration File> 
##                            PREPEND_FLAGS     <Additional Flags to Uncrustify>
##                            APPEND_FLAGS      <Additional Flags to Uncrustify>
##                            COMMENT           <Additional Comment for Target Invocation>
##                            WORKING_DIRECTORY <Working Directory>
##                            SRC_FILES         [FILE1 [FILE2 ...]] )
##
## Creates a new target with the given NAME for running uncrustify over the given SRC_FILES.
##
## MODIFY_FILES, if set to TRUE, modifies the files in place and adds the created target to
## the style target.  Otherwise the files are not modified and the created target is added
## to the check target.
## Note: Setting MODIFY_FILES to FALSE is only supported in Uncrustify v0.61 or greater.
##
## CFG_FILE defines the uncrustify settings.
##
## PREPEND_FLAGS are additional flags added to the front of the uncrustify flags.
##
## APPEND_FLAGS are additional flags added to the end of the uncrustify flags.
##
## COMMENT is prepended to CMake's output for this target.
##
## WORKING_DIRECTORY is the directory in which uncrustify will be run.  It defaults 
## to the directory where this macro is called.
##
## SRC_FILES is a list of source files to style/check with uncrustify.
##
##------------------------------------------------------------------------------
macro(blt_add_uncrustify_target)
    
    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs NAME MODIFY_FILES CFG_FILE COMMENT WORKING_DIRECTORY)
    set(multiValueArgs SRC_FILES PREPEND_FLAGS APPEND_FLAGS)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Check/Set required parameters
    if(NOT DEFINED arg_NAME)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a NAME parameter")
    endif()

    if(NOT DEFINED arg_CFG_FILE)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a CFG_FILE parameter")
    endif()

    if(NOT DEFINED arg_SRC_FILES)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a SRC_FILES parameter")
    endif()

    if(NOT DEFINED arg_MODIFY_FILES)
        set(arg_MODIFY_FILES FALSE)
    endif()

    if(DEFINED arg_WORKING_DIRECTORY)
        set(_wd ${arg_WORKING_DIRECTORY})
    else()
        set(_wd ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    set(_generate_target TRUE)

    if(${arg_MODIFY_FILES})
        set(MODIFY_FILES_FLAG "--no-backup")
    else()
        set(MODIFY_FILES_FLAG "--check")

        # Check the version -- output is of the form "uncrustify X.Y.Z"
        execute_process(
            COMMAND ${UNCRUSTIFY_EXECUTABLE} --version
            OUTPUT_VARIABLE _version_str
            OUTPUT_STRIP_TRAILING_WHITESPACE )
        string(REGEX MATCH "([0-9]+(\\.)?)+(_[a-zA-Z])?$" _uncrustify_version ${_version_str})
        
        # Skip 'check' target if version is not high enough 
        if(_uncrustify_version VERSION_LESS 0.61)
            set(_generate_target FALSE)
            message(WARNING "blt_add_uncrustify_target requires uncrustify v0.61 or greater "
                            " for style check targets. "
                            " Current uncrustify executable: '${UNCRUSTIFY_EXECUTABLE}' "
                            " Current uncrustify version is: ${_uncrustify_version}."    )
        endif()        
    endif()

    if(_generate_target)
        add_custom_target(${arg_NAME}
                COMMAND ${UNCRUSTIFY_EXECUTABLE} ${arg_PREPEND_FLAGS}
                    -c ${arg_CFG_FILE} ${MODIFY_FILES_FLAG} ${arg_SRC_FILES} ${arg_APPEND_FLAGS}
                WORKING_DIRECTORY ${_wd} 
                COMMENT "${arg_COMMENT}Running uncrustify source code formatting checks.")
            
        # hook our new target into the proper dependency chain
        if(${arg_MODIFY_FILES})
            add_dependencies(uncrustify_style ${arg_NAME})
        else()
            add_dependencies(uncrustify_check ${arg_NAME})
        endif()

        # Code formatting targets should only be run on demand
        set_property(TARGET ${arg_NAME} PROPERTY EXCLUDE_FROM_ALL TRUE)
        set_property(TARGET ${arg_NAME} PROPERTY EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    endif()

endmacro(blt_add_uncrustify_target)

##------------------------------------------------------------------------------
## blt_add_astyle_target( NAME              <Created Target Name>
##                        MODIFY_FILES      [TRUE | FALSE (default)]
##                        CFG_FILE          <AStyle Configuration File> 
##                        PREPEND_FLAGS     <Additional Flags to AStyle>
##                        APPEND_FLAGS      <Additional Flags to AStyle>
##                        COMMENT           <Additional Comment for Target Invocation>
##                        WORKING_DIRECTORY <Working Directory>
##                        SRC_FILES         [FILE1 [FILE2 ...]] )
##
## Creates a new target with the given NAME for running astyle over the given SRC_FILES.
##
## MODIFY_FILES, if set to TRUE, modifies the files in place and adds the created target to
## the style target. Otherwise the files are not modified and the created target is added
## to the check target.  
## Note: Setting MODIFY_FILES to FALSE is only supported in AStyle v2.05 or greater.
##
## CFG_FILE defines the astyle settings.
##
## PREPEND_FLAGS are additional flags added to the front of the astyle flags.
##
## APPEND_FLAGS are additional flags added to the end of the astyle flags.
##
## COMMENT is prepended to CMake's output for this target.
##
## WORKING_DIRECTORY is the directory in which astyle will be run. It defaults to 
## the directory where this macro is called.
##
## SRC_FILES is a list of source files to style/check with astyle.
##
##------------------------------------------------------------------------------
macro(blt_add_astyle_target)

    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs NAME MODIFY_FILES CFG_FILE COMMENT WORKING_DIRECTORY)
    set(multiValueArgs SRC_FILES PREPEND_FLAGS APPEND_FLAGS)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Check/Set required parameters
    if(NOT DEFINED arg_NAME)
        message(FATAL_ERROR "blt_add_astyle_target requires a NAME parameter")
    endif()

    if(NOT DEFINED arg_CFG_FILE)
        message(FATAL_ERROR "blt_add_astyle_target requires a CFG_FILE parameter")
    endif()

    if(NOT DEFINED arg_SRC_FILES)
        message(FATAL_ERROR "blt_add_astyle_target requires a SRC_FILES parameter")
    endif()

    if(NOT DEFINED arg_MODIFY_FILES)
        set(arg_MODIFY_FILES FALSE)
    endif()

    if(DEFINED arg_WORKING_DIRECTORY)
        set(_wd ${arg_WORKING_DIRECTORY})
    else()
        set(_wd ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    set(_generate_target TRUE)

    if(${arg_MODIFY_FILES})
        set(MODIFY_FILES_FLAG --suffix=none)
    else()
        set(MODIFY_FILES_FLAG --dry-run)

        # Check the version -- output is of the form "Artistic Style Version X.Y.Z"
        execute_process(
            COMMAND ${ASTYLE_EXECUTABLE} --version
            OUTPUT_VARIABLE _version_str
            OUTPUT_STRIP_TRAILING_WHITESPACE )
        string(REGEX MATCH "([0-9]+(\\.)?)+$" _astyle_version ${_version_str})
        
        # Skip 'check' target if version is not high enough 
        if(_astyle_version VERSION_LESS 2.05)
            set(_generate_target FALSE)
            message(WARNING "blt_add_astyle_target requires AStyle v2.05 or greater "
                            " for style check targets. "
                            " Current AStyle executable: '${ASTYLE_EXECUTABLE}' "
                            " Current AStyle version is: ${_astyle_version}."    )
        endif()
    endif()

    if(_generate_target)

        # AStyle doesn't report failure when there are files that require formatting.
        # Fix this with a wrapper script that parses the output.
        set(wrapped_astyle_script ${CMAKE_CURRENT_BINARY_DIR}/WrapAstyle_${arg_NAME}.cmake)

        configure_file(
            ${BLT_ROOT_DIR}/cmake/WrapAstyle.cmake.in
            ${wrapped_astyle_script}
            @ONLY )

        add_custom_target(
            ${arg_NAME}
            COMMAND ${CMAKE_COMMAND} -P ${wrapped_astyle_script}
            WORKING_DIRECTORY ${_wd}
            COMMENT "${arg_COMMENT}Running AStyle source code formatting checks.")
        
        # Hook our new target into the proper dependency chain
        if(${arg_MODIFY_FILES})
            add_dependencies(astyle_style ${arg_NAME})
        else()
            add_dependencies(astyle_check ${arg_NAME})
        endif()

        # Code formatting targets should only be run on demand
        set_property(TARGET ${arg_NAME} PROPERTY EXCLUDE_FROM_ALL TRUE)
        set_property(TARGET ${arg_NAME} PROPERTY EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    endif()
endmacro(blt_add_astyle_target)
