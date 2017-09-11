#######################################################
# add a target to generate documentation with sphinx
#######################################################

if(SPHINX_FOUND)
    add_sphinx_target( calc_pi_sphinx )
endif()
