 
################################
cmake_minimum_required(VERSION 3.11)

project(combine_static_library_test LANGUAGES C CXX)

###############################################################################
# Setup BLT
###############################################################################
# Set BLT_SOURCE_DIR to default location, if not set by user 
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../../..")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)


blt_add_library( NAME Foo1
                 SOURCES Foo1.cpp
                 INCLUDES Foo1.hpp
               )
           
blt_add_library( NAME Foo23
                 SOURCES Foo2.cpp Foo3.cpp
                 INCLUDES Foo2.hpp Foo3.hpp
               )
           
blt_combine_static_libraries( NAME FooStatic
                       SOURCE_LIBS Foo1 Foo23
                       LIB_TYPE STATIC
                       LINK_PREPEND ""
                       LINK_POSTPEND ""
                     )
                 
blt_combine_static_libraries( NAME FooShared
                       SOURCE_LIBS Foo1 Foo23
                       LIB_TYPE SHARED
                       LINK_PREPEND ""
                       LINK_POSTPEND ""
                     )

blt_add_executable( NAME FooStatic.x
                    SOURCES main.cpp
                    DEPENDS_ON FooStatic
                  )
              
blt_add_executable( NAME FooShared.x
                    SOURCES main.cpp
                    DEPENDS_ON FooShared
                  )