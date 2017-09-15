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

###############################################################################
# Targets related to source code checks (formatting, static analysis, etc)
###############################################################################

add_custom_target(check)

if(UNCRUSTIFY_FOUND)
    # targets for verifying formatting
    add_custom_target(uncrustify_check)
    add_dependencies(check uncrustify_check)
    # targets for forcing formatting
    add_custom_target(uncrustify_inplace)

endif()

##------------------------------------------------------------------------------
## - Macro that helps add all code check targets
##
## blt_add_code_check_targets(CFG_FILE <uncrusify_configuration_file>)
##
##------------------------------------------------------------------------------
macro(blt_add_code_check_targets )

    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs CFG_FILE)
    set(multiValueArgs)
    
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(UNCRUSTIFY_FOUND)
        # Only run uncrustify on C and C++ files
        # Note, we can later extend this by passing in a list of valid types to the macro
        set(_fileTypes ".cpp" ".hpp" ".cxx" ".hxx" ".cc" ".c" ".h")
    
        # generate the filtered list of source files
        set(_filt_sources)
        foreach(_file ${${PROJECT_NAME}_ALL_SOURCES})
          get_filename_component(_ext ${_file} EXT)
          list(FIND _fileTypes "${_ext}" _index)
      
          if(_index GREATER -1)
             file(RELATIVE_PATH _relpath ${CMAKE_CURRENT_BINARY_DIR} ${_file})
             list(APPEND _filt_sources ${_relpath})
          endif()
        endforeach()

        blt_add_uncrustify_check(CFG_FILE ${arg_CFG_FILE}   SRC_FILES ${_filt_sources})
        blt_add_uncrustify_inplace(CFG_FILE ${arg_CFG_FILE} SRC_FILES ${_filt_sources})
    endif()

endmacro(blt_add_code_check_targets)
    

##------------------------------------------------------------------------------
## - Macro for invoking uncrustify to check code formatting
##
## blt_add_uncrustify_check( CFG_FILE <uncrusify_configuration_file> 
##                           FLAGS <additional_flags_to_uncrustify>
##                           COMMENT <additional_comment_for_target_invocation>
##                           WORKING_DIRECTORY <working_directory>
##                           REDIRECT <redirection_commands>
##                           SRC_FILES <list_of_src_files_to_uncrustify> )
##
##------------------------------------------------------------------------------
macro(blt_add_uncrustify_check)
        
    message(STATUS "Creating uncrustify check target: uncrustify_check_${PROJECT_NAME}")

    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs CFG_FILE COMMENT WORKING_DIRECTORY)
    set(multiValueArgs SRC_FILES FLAGS REDIRECT)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(${arg_WORKING_DIRECTORY})
        set(_wd ${arg_WORKING_DIRECTORY})
    else()
        set(_wd ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    add_custom_target("uncrustify_check_${PROJECT_NAME}"
            ${UNCRUSTIFY_EXECUTABLE} ${arg_FLAGS}
            -c ${arg_CFG_FILE} --check ${arg_SRC_FILES} ${arg_REDIRECT}
             WORKING_DIRECTORY ${_wd} 
             COMMENT "${arg_COMMENT}Running uncrustify source code formatting checks.")
        
    # hook our new target into the check dependency chain
    add_dependencies(uncrustify_check "uncrustify_check_${PROJECT_NAME}")

endmacro(blt_add_uncrustify_check)


##------------------------------------------------------------------------------
## - Macro for invoking uncrustify to apply formatting inplace
##
## blt_add_uncrustify_inplace(CFG_FILE <uncrusify_configuration_file> 
##                            FLAGS <additional_flags_to_uncrustify>
##                            COMMENT <additional_comment_for_target_invocation>
##                            WORKING_DIRECTORY <working_directory>
##                            REDIRECT <redirection_commands>
##                            SRC_FILES <list_of_src_files_to_uncrustify> )
##
##------------------------------------------------------------------------------
macro(blt_add_uncrustify_inplace)
    
    message(STATUS "Creating uncrustify inplace target: uncrustify_inplace_${PROJECT_NAME}")

    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs CFG_FILE COMMENT WORKING_DIRECTORY)
    set(multiValueArgs SRC_FILES FLAGS REDIRECT)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(${arg_WORKING_DIRECTORY})
        set(_wd ${arg_WORKING_DIRECTORY})
    else()
        set(_wd ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    add_custom_target("uncrustify_inplace_${PROJECT_NAME}"
            ${UNCRUSTIFY_EXECUTABLE} ${arg_FLAGS}
            -c ${arg_CFG_FILE} --no-backup ${arg_SRC_FILES} ${arg_REDIRECT}
             WORKING_DIRECTORY ${_wd} 
             COMMENT "${arg_COMMENT}Running uncrustify to apply code formatting settings.")
        
    # hook our new target into the uncrustify_inplace dependency chain
    add_dependencies(uncrustify_inplace "uncrustify_inplace_${PROJECT_NAME}")

endmacro(blt_add_uncrustify_inplace)
