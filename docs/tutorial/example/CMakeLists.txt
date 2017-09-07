#
#
#

cmake_minimum_required(VERSION 3.3)

project( pi_playground )

include(blt/SetupBLT.cmake)

#
# First example (adding an executable)
#
blt_add_executable( NAME calc_pi_example_1
                    SOURCES calc_pi_example_1.cpp )



#
# Second example (adding an executable)
#

blt_add_library( NAME calc_pi_lib
                 HEADERS calc_pi_serial_lib.hpp
                 SOURCES calc_pi_serial_lib.cpp )


blt_add_executable( NAME calc_pi_example_2
                    SOURCES calc_pi_example_2.cpp 
                    DEPENDS_ON calc_pi_lib)
