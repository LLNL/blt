project(dibs)
cmake_minimum_required(VERSION 2.8)

set(SPACK_ROOT "" CACHE STRING "Location of the Spack root directory")
set(SPACK_SHARE ${SPACK_ROOT}/share/spack)
set(SPACK_BINARY ${SPACK_ROOT}/bin/spack)

# Set up scripts for installing Spack packages and querying their environment
configure_file(spack_env_parser.in ${CMAKE_CURRENT_BINARY_DIR}/spack_env_parser)
configure_file(spack_to_raw.in ${CMAKE_CURRENT_BINARY_DIR}/spack_to_raw)

# If we don't have them, install environment-modules
execute_process(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/spack_to_raw environment-modules)

# Find out where environment-modules are installed
execute_process(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/spack_env_parser environment-modules PREFIX OUTPUT_VARIABLE environment_module_dir)

# Now we know where the raw module command is
set(MODULE_CMD ${environment_module_dir}/Modules/bin/modulecmd)

set(MODULE_FILE_NAME .envrc CACH SRING "Name of the file to emit containing environment variables for the build")

# Configure the rest of the Dibs system
configure_file(dibs.in ${CMAKE_CURRENT_BINARY_DIR}/dibs @ONLY)
configure_file(spack_loader.in ${CMAKE_CURRENT_BINARY_DIR}/spack_loader)
configure_file(clean_modules.in ${CMAKE_CURRENT_BINARY_DIR}/clean_modules)
configure_file(dibs.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/dibs.cmake @ONLY)

# Install
install(PROGRAMS module_parser module_parser.py DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/dibs.cmake DESTINATION cmake)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/spack_to_raw DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/spack_loader DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/dibs DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/spack_env_parser DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clean_modules DESTINATION bin)
install(PROGRAMS parse_environment_names.py DESTINATION bin)
install(PROGRAMS parse_environment_values.py DESTINATION bin)
