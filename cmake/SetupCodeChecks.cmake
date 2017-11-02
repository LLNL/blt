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
add_custom_target(style)

if(UNCRUSTIFY_FOUND)
    # targets for verifying formatting
    add_custom_target(uncrustify_check)
    add_dependencies(check uncrustify_check)

    # targets for modifying formatting
    add_custom_target(uncrustify_style)
    add_dependencies(style uncrustify_style)
endif()


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
##
## CFG_FILE defines the uncrustify settings.
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
##------------------------------------------------------------------------------
macro(blt_add_uncrustify_target)
    
    ## parse the arguments to the macro
    set(options)
    set(singleValueArgs NAME CFG_FILE COMMENT WORKING_DIRECTORY)
    set(multiValueArgs SRC_FILES PREPEND_FLAGS APPEND_FLAGS)

    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Check required parameters
    if(NOT DEFINED arg_NAME)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a NAME parameter")
    endif()

    if(NOT DEFINED arg_CFG_FILE)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a CFG_FILE parameter")
    endif()

    if(NOT DEFINED arg_SRC_FILES)
        message(FATAL_ERROR "blt_add_uncrustify_target requires a SRC_FILES parameter")
    endif()

    if(${arg_WORKING_DIRECTORY})
        set(_wd ${arg_WORKING_DIRECTORY})
    else()
        set(_wd ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(NOT DEFINED arg_MODIFY_FILES OR NOT ${arg_MODIFY_FILES})
        set(MODIFY_FILES_FLAG "--check")
    else()
        set(MODIFY_FILES_FLAG "--no-backup")
    endif()

    add_custom_target(${arg_NAME}
            COMMAND ${UNCRUSTIFY_EXECUTABLE} ${arg_PREPEND_FLAGS}
                -c ${arg_CFG_FILE} ${MODIFY_FILES_FLAG} ${arg_SRC_FILES} ${arg_APPEND_FLAGS}
            WORKING_DIRECTORY ${_wd} 
            COMMENT "${arg_COMMENT}Running uncrustify source code formatting checks.")
        
    if(NOT DEFINED arg_MODIFY_FILES OR NOT ${arg_MODIFY_FILES})
        # hook our new target into the check dependency chain
        add_dependencies(uncrustify_check ${arg_NAME})
    else()
        add_dependencies(uncrustify_style ${arg_NAME})
    endif()

endmacro(blt_add_uncrustify_target)
