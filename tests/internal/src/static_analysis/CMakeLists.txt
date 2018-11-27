#####################################
# Test clang-query static analysis
# Clang Query needs to be enabled, with appropriate Flags set up for building and running
# on your current GPU architecture. 
#####################################

set(BLT_CLANG_QUERY_CHECKER_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/static_analysis_passes)
blt_add_executable( NAME well_analyzed_executable
                    SOURCES well_analyzed_source.cpp
)
blt_add_code_checks(PREFIX dog
  SOURCES well_analyzed_source.cpp)
