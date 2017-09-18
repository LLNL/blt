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
# Set BLT_SOURCE_DIR to default location, if not set by user  
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "blt")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

