#######################################################
# add a target to generate documentation with Doxygen
#######################################################

if(DOXYGEN_FOUND)
    add_doxygen_target( calc_pi_doxygen )
endif()

