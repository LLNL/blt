# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

# TODO: temporary hack
set(CONDUIT_DIR ${arg_PATH})

# first Check for CONDUIT_DIR

if(NOT CONDUIT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
endif()

if(NOT EXISTS ${CONDUIT_DIR}/lib/cmake/conduit.cmake)
    MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file (${CONDUIT_DIR}/lib/cmake/conduit.cmake)")
endif()

include(${CONDUIT_DIR}/lib/cmake/conduit.cmake)

set(CONDUIT_INCLUDE_DIRS ${CONDUIT_DIR}/include/conduit)
set(CONDUIT_LIBRARIES conduit conduit_relay)

#
# Display used CMake variables
#
message(STATUS "Conduit Include Dirs: ${CONDUIT_INCLUDE_DIRS}")
message(STATUS "Conduit Libraries:    ${CONDUIT_LIBRARIES}")

# Ensure it was actually found
set(CONDUIT_FOUND TRUE CACHE BOOL "")
foreach(_path IN LISTS CONDUIT_INCLUDE_DIRS)
    if(NOT EXISTS ${_path})
        set(CONDUIT_FOUND FALSE CACHE BOOL "" FORCE)
    endif()
endforeach()
message(STATUS "Conduit Found: ${CONDUIT_FOUND}")

blt_register_library( NAME      conduit
                      INCLUDES  ${CONDUIT_INCLUDE_DIRS}
                      LIBRARIES ${CONDUIT_LIBRARIES}
                      TREAT_INCLUDES_AS_SYSTEM ON)

