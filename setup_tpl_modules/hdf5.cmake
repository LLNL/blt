# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

# TODO: temporary hack
set(HDF5_DIR ${arg_PATH})

# first Check for HDF5_DIR
if(NOT HDF5_DIR)
    MESSAGE(FATAL_ERROR "HDF5 support needs explicit HDF5_DIR")
endif()

# find the absolute path w/ symlinks resolved of the passed HDF5_DIR, 
# since sanity checks later need to compare against the real path
get_filename_component(HDF5_DIR_REAL "${HDF5_DIR}" REALPATH)
message(STATUS "Looking for HDF5 at: " ${HDF5_DIR_REAL})

# CMake's FindHDF5 module uses the HDF5_ROOT env var
set(HDF5_ROOT ${HDF5_DIR_REAL})

if(NOT WIN32)
    set(ENV{HDF5_ROOT} ${HDF5_ROOT}/bin)
    # Use CMake's FindHDF5 module, which uses hdf5's compiler wrappers to extract
    # all the info about the hdf5 install
    include(FindHDF5)
else()
    # CMake's FindHDF5 module is buggy on windows and will put the dll
    # in HDF5_LIBRARY.  Instead, use the 'CONFIG' signature of find_package
    # with appropriate hints for where cmake can find hdf5-config.cmake.
    find_package(HDF5 CONFIG 
                 REQUIRED
                 HINTS ${HDF5_DIR}/cmake/hdf5 
                       ${HDF5_DIR}/lib/cmake/hdf5
                       ${HDF5_DIR}/share/cmake/hdf5)
endif()

# FindHDF5/find_package sets HDF5_DIR to it's installed CMake info if it exists
# we want to keep HDF5_DIR as the root dir of the install to be 
# consistent with other packages

# find the absolute path w/ symlinks resolved of the passed HDF5_DIR, 
# since sanity checks later need to compare against the real path
get_filename_component(HDF5_DIR_REAL "${HDF5_ROOT}" REALPATH)

set(HDF5_DIR ${HDF5_DIR_REAL} CACHE PATH "" FORCE)
message(STATUS "HDF5_DIR_REAL=${HDF5_DIR_REAL}")
#
# Sanity check to alert us if some how we found an hdf5 instance
# in an unexpected location.  
#
message(STATUS "Checking that found HDF5_INCLUDE_DIRS are in HDF5_DIR")

#
# HDF5_INCLUDE_DIRS may also include paths to external lib headers 
# (such as szip), so we check that *at least one* of the includes
# listed in HDF5_INCLUDE_DIRS exists in the HDF5_DIR specified. 
#

# HDF5_INCLUDE_DIR is deprecated, but there are still some cases
# where HDF5_INCLUDE_DIR is set, but HDF5_INCLUDE_DIRS is not
if(NOT HDF5_INCLUDE_DIRS)
    if(HDF5_INCLUDE_DIR)
        set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "FindHDF5 did not provide HDF5_INCLUDE_DIRS or HDF5_INCLUDE_DIR.")
    endif()
endif()

message(STATUS "HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS}")
set(check_hdf5_inc_dir_ok 0)
foreach(IDIR ${HDF5_INCLUDE_DIRS})
    
    # get real path of the include dir 
    # w/ abs and symlinks resolved
    get_filename_component(IDIR_REAL "${IDIR}" REALPATH)
    # check if idir_real is a substring of hdf5_dir

    if("${IDIR_REAL}" MATCHES "${HDF5_DIR}")
        message(STATUS " ${IDIR_REAL} includes HDF5_DIR (${HDF5_DIR})")
        set(check_hdf5_inc_dir_ok 1)
    endif()
endforeach()

if(NOT check_hdf5_inc_dir_ok)
    message(FATAL_ERROR " ${HDF5_INCLUDE_DIRS} does not include HDF5_DIR")
endif()

#
# filter HDF5_LIBRARIES to remove hdf5_hl if it exists
# we don't use hdf5_hl, but if we link with it will become
# a transitive dependency
#
set(HDF5_HL_LIB FALSE)
foreach(LIB ${HDF5_LIBRARIES})
    if("${LIB}" MATCHES "hdf5_hl")
        set(HDF5_HL_LIB ${LIB})
    endif()
endforeach()

if(HDF5_HL_LIB)
    message(STATUS "Removing hdf5_hl from HDF5_LIBRARIES")
    list(REMOVE_ITEM HDF5_LIBRARIES ${HDF5_HL_LIB})
endif()


#
# Display used HDF5 CMake variables
#
message(STATUS "HDF5 Include Dirs: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 Libraries:    ${HDF5_LIBRARIES}")

# Ensure it was actually found
set(HDF5_FOUND TRUE CACHE BOOL "")
foreach(_path IN LISTS HDF5_INCLUDE_DIRS HDF5_LIBRARIES)
    if(NOT EXISTS ${_path})
        set(HDF5_FOUND FALSE CACHE BOOL "" FORCE)
    endif()
endforeach()
message(STATUS "HDF5 Found: ${HDF5_FOUND}")

blt_register_library( NAME      hdf5
                      INCLUDES  ${HDF5_INCLUDE_DIRS}
                      LIBRARIES ${HDF5_LIBRARIES}
                      TREAT_INCLUDES_AS_SYSTEM ON)
