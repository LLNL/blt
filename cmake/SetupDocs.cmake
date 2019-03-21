# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

add_custom_target(docs)

if(DOXYGEN_FOUND)
    add_custom_target(doxygen_docs)
    add_dependencies(docs doxygen_docs)
endif()

if(SPHINX_FOUND)
    add_custom_target(sphinx_docs)
    add_dependencies(docs sphinx_docs)
endif()


##------------------------------------------------------------------------------
## - Macro for invoking doxygen to generate docs
##
##  blt_add_doxygen_target(doxygen_target_name)
##
##  Expects to find a Doxyfile.in in the directory the macro is called in. 
##  
##  This macro sets up the doxygen paths so that the doc builds happen 
##  out of source. For a make install, this will place the resulting docs in 
##  docs/doxygen/`doxygen_target_name`
##
##------------------------------------------------------------------------------
macro(blt_add_doxygen_target doxygen_target_name)

    # add a target to generate API documentation with Doxygen
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(${doxygen_target_name}
                     ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     COMMENT "Generating API documentation with Doxygen for ${doxygen_target_name} target" VERBATIM)

    add_dependencies(doxygen_docs ${doxygen_target_name})

    install(CODE "execute_process(COMMAND ${CMAKE_BUILD_TOOL} ${doxygen_target_name} WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")")

    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html" 
            DESTINATION docs/doxygen/${doxygen_target_name} OPTIONAL)

endmacro(blt_add_doxygen_target)


##------------------------------------------------------------------------------
## - Macro for invoking sphinx to generate docs
##
##  blt_add_sphinx_target(sphinx_target_name)
##
##  Expects to find a conf.py or conf.py.in in the directory the macro is 
##  called in. 
##  
##
##  If conf.py is found, it is directly used as input to sphinx.
##
##  If conf.py.in is found, this macro uses CMake's configure_file() command
##  to generate a conf.py, which is then used as input to sphinx.
##
##  This macro sets up the sphinx paths so that the doc builds happen 
##  out of source. For a make install, this will place the resulting docs in 
##  docs/sphinx/`sphinx_target_name`
##
##------------------------------------------------------------------------------
macro(blt_add_sphinx_target sphinx_target_name )

    # configured documentation tools and intermediate build results
    set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

    # Sphinx cache with pickled ReST documents
    set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

    # HTML output directory
    set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

    # support both direct use of a conf.py file and a cmake-configured
    # sphinx input file (conf.py.in). The cmake-configured input file is
    # preferred when both exist.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
                       "${SPHINX_BUILD_DIR}/conf.py"
                       @ONLY)

        add_custom_target(${sphinx_target_name}
                          ${SPHINX_EXECUTABLE}
                          -q -b html
                          #-W disable warn on error for now, while our sphinx env is still in flux
                          -c "${SPHINX_BUILD_DIR}"
                          -d "${SPHINX_CACHE_DIR}"
                          "${CMAKE_CURRENT_SOURCE_DIR}"
                          "${SPHINX_HTML_DIR}"
                          COMMENT "Building HTML documentation with Sphinx for ${sphinx_target_name} target"
                          DEPENDS ${deps})
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conf.py")
        add_custom_target(${sphinx_target_name}
                          ${SPHINX_EXECUTABLE}
                          -q -b html
                          #-W disable warn on error for now, while our sphinx env is still in flux
                          -d "${SPHINX_CACHE_DIR}"
                          "${CMAKE_CURRENT_SOURCE_DIR}"
                          "${SPHINX_HTML_DIR}"
                          COMMENT "Building HTML documentation with Sphinx for ${sphinx_target_name} target"
                          DEPENDS ${deps})
    else()
        message(FATAL_ERROR "Failed to find sphinx 'conf.py' or 'conf.py.in' in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
        
    # hook our new target into the docs dependency chain
    add_dependencies(sphinx_docs ${sphinx_target_name})

    ######
    # This snippet makes sure if we do a make install w/o the optional "docs"
    # target built, it will be built during the install process.
    ######

    install(CODE "execute_process(COMMAND ${CMAKE_BUILD_TOOL} ${sphinx_target_name} WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")")

    install(DIRECTORY "${SPHINX_HTML_DIR}" 
            DESTINATION "docs/sphinx/${sphinx_target_name}" OPTIONAL)

endmacro(blt_add_sphinx_target)
