# This file is intended to be included in the *-config.cmake files of
# any project using a third-party library.  The macro 
# `blt_install_tpl_setups(DESTINATION <dir>)`  installs this file
# into the destination specified by the argument <dir>.

# Consider an example project foo, built using BLT, that depends on OpenMP.  
# This file is used by projects that link to foo, to create a target for OpenMP
# with valid config flags. 
include("${CMAKE_CURRENT_LIST_DIR}/BLTOptions.cmake")
# BLTInstallableMacros provides helper macros for setting up and creating
# third-party library targets.
include("${CMAKE_CURRENT_LIST_DIR}/BLTInstallableMacros.cmake")

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/SetupMPI.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/SetupMPI.cmake")
endif()

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/SetupOpenMP.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/SetupOpenMP.cmake")
endif()

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/SetupCUDA.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/SetupCUDA.cmake")
endif()

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/SetupHIP.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/SetupHIP.cmake")
endif()
