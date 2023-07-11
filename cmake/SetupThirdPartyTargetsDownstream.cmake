# This file is intended to be included in the *-config.cmake files of
# any project using a third-party library.  The macro 
# `blt_install_tpl_setups(DESTINATION <dir>)`  installs this file
# into the destination specified by the argument <dir>.

# Consider an example project foo, built using BLT, that depends on OpenMP.  
# This file is used by projects that link to foo, to create a target for OpenMP
# with valid config flags. 
include("${CMAKE_CURRENT_LIST_DIR}/BLTOptions.cmake")
# Despite the inclusion of BLTMacros and BLTPrivateMacros, these macros are not
# guaranteed to work as expected in arbitrary downstream projects.  Instead,
# they are included here so that SetupMPI, SetupOpenMP, SetupCUDA, and SetupHIP 
# can function as expected. 
include("${CMAKE_CURRENT_LIST_DIR}/BLTPrivateMacros.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/BLTMacros.cmake")

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
