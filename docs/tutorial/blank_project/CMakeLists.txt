###############################################################################
#
# BLT Tutorial Example: Blank Project.
#
###############################################################################

cmake_minimum_required(VERSION 3.3)

project( blank )

###############################################################################
# Setup BLT
###############################################################################
# TODO: Sensible explanation of this.  
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "blt")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

