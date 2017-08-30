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

##------------------------------------------------------------------------------
## blt_git( SOURCE_DIR <dir>
##          GIT_COMMAND <command>
##          OUTPUT_VARIABLE <out>
##          RETURN_CODE <rc>
##          [QUIET] )
##
## Runs the supplied git command on the given Git repository.
##
## This macro runs the user-supplied Git command, given by GIT_COMMAND, on the
## given Git repository corresponding to SOURCE_DIR. The supplied GIT_COMMAND
## is just a string consisting of the Git command and its arguments. The
## resulting output is returned to the supplied CMake variable provided by
## the OUTPUT_VARIABLE argument.
##
## A return code for the Git command is returned to the caller via the CMake
## variable provided with the RETURN_CODE argument. A non-zero return code
## indicates that an error has occured.
##
## Note, this macro assumes FindGit() was invoked and was successful. It relies
## on the following variables set by FindGit():
##   - Git_FOUND flag that indicates if git is found
##   - GIT_EXECUTABLE points to the Git binary
##
## If Git_FOUND is "false" this macro will throw a FATAL_ERROR message.
##
## Usage Example:
##
##    blt_git( SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
##             GIT_COMMAND describe --tags master
##             OUTPUT_VARIABLE axom_tag
##             RETURN_CODE rc
##             )
##
##    if (NOT ${rc} EQUAL 0)
##      message( FATAL_ERROR "blt_git failed!" )
##    endif()
##
##------------------------------------------------------------------------------
macro(blt_git)

    set(options)
    set(singleValueArgs SOURCE_DIR OUTPUT_VARIABLE RETURN_CODE)
    set(multiValueArgs GIT_COMMAND )

    ## parse macro arguments
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    ## ensure required arguments are supplied
    if ( NOT DEFINED arg_SOURCE_DIR )
      message(FATAL_ERROR "SOURCE_DIR is a required argument to blt_git()")
    endif()

    if ( NOT DEFINED arg_GIT_COMMAND )
      message(FATAL_ERROR "GIT_COMMAND is a required argument to blt_git()")
    endif()

    if ( NOT DEFINED arg_OUTPUT_VARIABLE )
      message(FATAL_ERROR "OUTPUT_VARIABLE is a required argument to blt_git()")
    endif()

    if ( NOT DEFINED arg_RETURN_CODE )
      message(FATAL_ERROR "RETURN_CODE is a required argument to blt_git()")
    endif()

    ## check arguments
    if (GIT_FOUND)

      ## assemble the Git command
      set(git_cmd "${GIT_EXECUTABLE}" "${arg_GIT_COMMAND}" )

      ## run it
      execute_process( COMMAND
                         ${git_cmd}
                       WORKING_DIRECTORY
                         "${arg_SOURCE_DIR}"
                       RESULT_VARIABLE
                         ${arg_RETURN_CODE}
                       OUTPUT_VARIABLE
                         ${arg_OUTPUT_VARIABLE}
                       ERROR_QUIET
                       OUTPUT_STRIP_TRAILING_WHITESPACE
                       ERROR_STRIP_TRAILING_WHITESPACE
                       )

    else( )
       message( FATAL_ERROR "Git is not found. Git is required for blt_git()")
    endif()

endmacro(blt_git)

##------------------------------------------------------------------------------
## blt_is_git_repo( OUTPUT_STATE <state>
##                  [SOURCE_DIR <dir>]
##                  )
##
## Checks if we are working with a valid Git repository.
##
## This macro checks if the corresponding source directory is a valid Git repo.
## Nominally, the corresponding source directory that is used is set to
## ${CMAKE_CURRENT_SOURCE_DIR}. A different source directory may be optionally
## specified using the SOURCE_DIR argument.
##
## The resulting state is stored in the CMake variable specified by the caller
## using the OUTPUT_STATE parameter.
##
## Usage Example:
##
##    blt_is_git_repo( OUTTPUT_STATE is_git_repo )
##    if ( ${is_git_repo} )
##      message(STATUS "Pointing to a valid Git repo!")
##    else()
##      message(STATUS "Not a Git repo!")
##    endif()
##
##------------------------------------------------------------------------------
macro(blt_is_git_repo)

    set(options)
    set(singleValueArgs OUTPUT_STATE SOURCE_DIR )
    set(multiValueArgs)

    ## parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    ## ensure required variables are supplied
    if ( NOT DEFINED arg_OUTPUT_STATE )
       message(FATAL_ERROR "OUTPUT_STATE is a required argument to blt_is_git_repo")
    endif()

    ## check if SOURCE_DIR was supplied
    if ( NOT DEFINED arg_SOURCE_DIR )
      set(git_dir ${CMAKE_CURRENT_SOURCE_DIR})
    else()
      set(git_dir ${arg_SOURCE_DIR})
    endif()

    blt_git( SOURCE_DIR ${git_dir}
             GIT_COMMAND rev-parse --show-toplevel
             OUTPUT_VARIABLE tmp
             RETURN_CODE rc
             )

    if ( NOT ${rc} EQUAL 0 )
       ## rev-parse failed, this is not a git repo
       set( ${arg_OUTPUT_STATE} FALSE )
    else()
       set( ${arg_OUTPUT_STATE} TRUE )
    endif()

endmacro(blt_is_git_repo)

##------------------------------------------------------------------------------
## blt_git_tag( OUTPUT_TAG <tag>
##              RETURN_CODE <rc>
##              [SOURCE_DIR <dir>]
##              [ON_BRANCH <branch>]
##              )
##
## Returns the latest tag on a corresponding Git repository.
##
## This macro gets the latest tag from a Git repository that can be specified
## via the SOURCE_DIR argument. If SOURCE_DIR is not supplied, the macro will
## use ${CMAKE_CURRENT_SOURCE_DIR}. By default the macro will return the latest
## tag on the branch that is currently checked out. A particular branch may be
## specified using the ON_BRANCH option.
##
## The tag is stored in the CMake variable specified by the caller using the
## the OUTPUT_TAG parameter.
##
## A return code for the Git command is returned to the caller via the CMake
## variable provided with the RETURN_CODE argument. A non-zero return code
## indicates that an error has occured.
##
## Usage Example:
##
##    blt_git_tag( OUTPUT_TAG tag RETURN_CODE rc ON_BRANCH master )
##
##    if ( NOT ${rc} EQUAL 0 )
##      message( FATAL_ERROR "blt_git_tag failed!" )
##    endif()
##
##    message( STATUS "tag=${tag}" )
##------------------------------------------------------------------------------
macro(blt_git_tag)

    set(options)
    set(singleValueArgs SOURCE_DIR ON_BRANCH OUTPUT_TAG RETURN_CODE )
    set(multiValueArgs)

    ## parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    ## ensure required arguments are supplied
    if ( NOT DEFINED arg_OUTPUT_TAG )
       message(FATAL_ERROR "OUTPUT_TAG is a required argument to blt_git_tag")
    endif()

    if ( NOT DEFINED arg_RETURN_CODE )
       message(FATAL_ERROR "RETURN_CODE is a required argument to blt_git_tag")
    endif()

    ## git command to execute
    if ( NOT DEFINED arg_ON_BRANCH )
      set(git_cmd describe --tags )
    else()
      set(git_cmd describe --tags ${arg_ON_BRANCH} )
    endif()

    ## set working directory
    if ( NOT DEFINED arg_SOURCE_DIR} )
      set(git_dir ${CMAKE_CURRENT_SOURCE_DIR})
    else()
      set(git_dir ${arg_SOURCE_DIR})
    endif()

    blt_git( SOURCE_DIR ${git_dir}
             GIT_COMMAND ${git_cmd}
             OUTPUT_VARIABLE ${arg_OUTPUT_TAG}
             RETURN_CODE ${arg_RETURN_CODE}
             )

endmacro(blt_git_tag)

##------------------------------------------------------------------------------
## blt_git_branch( BRANCH_NAME <branch>
##                 RETURN_CODE <rc>
##                 [SOURCE_DIR <dir>]
##                 )
##
## Returns the name of the active branch in the checkout space.
##
## This macro gets the name of the current active branch in the checkout space
## that can be specified using the SOURCE_DIR argument. If SOURCE_DIR is not
## supplied by the caller, this macro will point to the checkout space
## corresponding to ${CMAKE_CURRENT_SOURCE_DIR}.
##
## A return code for the Git command is returned to the caller via the CMake
## variable provided with the RETURN_CODE argument. A non-zero return code
## indicates that an error has occured.
##
## Usage Example:
##
##    blt_git_branch( BRANCH_NAME active_branch RETURN_CODE rc )
##
##    if ( NOT ${rc} EQUAL 0 )
##      message( FATAL_ERROR "blt_git_tag failed!" )
##    endif()
##
##    message( STATUS "active_branch=${active_branch}" )
##------------------------------------------------------------------------------
macro(blt_git_branch)

    set(options)
    set(singleValueArgs SOURCE_DIR BRANCH_NAME RETURN_CODE)
    set(multiValueArgs)

    ## parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    ## ensure required arguments are supplied
    if ( NOT DEFINED arg_BRANCH_NAME )
       message(FATAL_ERROR "BRANCH_NAME is a required argument to blt_git_branch" )
    endif()

    if ( NOT DEFINED arg_RETURN_CODE )
       message(FATAL_ERROR "RETURN_CODE is a required argument to blt_git_branch")
    endif()

    ## set set working directory
    if ( NOT DEFINED arg_SOURCE_DIR )
       set(git_dir ${CMAKE_CURRENT_SOURCE_DIR})
    else()
       set(git_dir ${arg_SOURCE_DIR})
    endif()

    blt_git( SOURCE_DIR ${git_dir}
             GIT_COMMAND rev-parse --abbrev-ref HEAD
             OUTPUT_VARIABLE ${arg_BRANCH_NAME}
             RETURN_CODE ${arg_RETURN_CODE}
             )

endmacro(blt_git_branch)

##------------------------------------------------------------------------------
## blt_git_hashcode( HASHCODE <hc>
##                   RETURN_CODE <rc>
##                   [SOURCE_DIR <dir>]
##                   [ON_BRANCH <branch>]
##                   )
##
## Returns the SHA-1 hashcode at the tip of a branch.
##
## This macro returns the SHA-1 hashcode at the tip of a branch that may be
## specified with the ON_BRANCH argument. If the ON_BRANCH argument is not
## supplied, the macro will return the SHA-1 hash at the tip of the current
## branch. In addition, the caller may specify the target Git repository using
## the SOURCE_DIR argument. Otherwise, if SOURCE_DIR is not specified, the
## macro will use ${CMAKE_CURRENT_SOURCE_DIR}.
##
## A return code for the Git command is returned to the caller via the CMake
## variable provided with the RETURN_CODE argument. A non-zero return code
## indicates that an error has occured.
##
## Usage Example:
##
##    blt_git_hashcode( HASHCODE sha1 RETURN_CODE rc )
##    if ( NOT ${rc} EQUAL 0 )
##      message( FATAL_ERROR "blt_git_hashcode failed!" )
##    endif()
##
##    message( STATUS "sha1=${sha1}" )
##------------------------------------------------------------------------------
macro(blt_git_hashcode)

    set(options)
    set(singleValueArgs SOURCE_DIR HASHCODE ON_BRANCH RETURN_CODE)
    set(multiValueArgs)

    ## parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    ## ensure required arguments are supplied
    if ( NOT DEFINED arg_HASHCODE )
       message(FATAL_ERROR "HASHCODE is a required argument to blt_git_hashcode" )
    endif()

    if ( NOT DEFINED arg_RETURN_CODE )
       message(FATAL_ERROR "RETURN_CODE is a required argument to blt_git_hashcode" )
    endif()

    ## set working directory
    if ( NOT DEFINED arg_SOURCE_DIR )
      set(git_dir ${CMAKE_CURRENT_SOURCE_DIR})
    else()
      set(git_dir ${arg_SOURCE_DIR})
    endif()

    ## set target ref
    if ( NOT DEFINED arg_ON_BRANCH )
      set(git_cmd rev-parse --short HEAD )
    else()
      set(git_cmd rev-parse --short ${arg_ON_BRANCH} )
    endif()

     blt_git( SOURCE_DIR ${git_dir}
              GIT_COMMAND ${git_cmd}
              OUTPUT_VARIABLE ${arg_HASHCODE}
              RETURN_CODE ${arg_RETURN_CODE}
              )

endmacro(blt_git_hashcode)
