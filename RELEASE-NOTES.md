# BLT Software Release Notes

Notes describing significant changes in each BLT release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

The project release numbers follow [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- Added support for C++17. Note: Neither XL or CMake's CUDA_STANDARD does not support
  C++17 (A BLT fatal error will occur).
- Added ability to override all MPI variables: BLT_MPI_COMPILE_FLAGS,
  BLT_MPI_INCLUDES, BLT_MPI_LIBRARIES, and BLT_MPI_LINK_FLAGS

### Changed

- BLT_CXX_STD is no longer defined to "c++11" by default. If undefined, BLT will
  not try and add any C++ standard flags.
- Handle FindMPI variable MPIEXEC changed to MPIEXEC_EXECUTABLE in CMake 3.10+.
  This now works regardless of which the user defines or what CMake returns.
- Handle CMake target property LINK_FLAGS changed to LINK_OPTIONS in CMake 3.13+.
  blt_add_target_link_flags() handles this under the covers and converts the 
  users strings to a list (3.13+) or list to a string (<3.13).  New property supports
  generator expressions so thats a plus.
- Improved how all BLT MPI information is being merged together and reported to users.
- Increased number of ranks in `blt_mpi_smoke` test to catch regression.

## 0.2.0 - Release date 2019-02-15

### Added
- Release notes...
- Explicitly check for CMake 3.8+ for required CMake features (Not fatal error)
- Object library support through blt_add_library(... OBJECT TRUE ...)
- Now reporting BLT version through CMake cache variable BLT_VERSION
- Output CMake version and executable used during CMake step
- Clang-query support now added (Thanks David Poliakoff)

### Removed

### Deprecated

### Changed
- Object libraries no longer call target_link_libraries() but will pass inherited information
  because why would anyone ever want to install/export a bunch of object files.
- Remove duplicates of select target properties at the end of blt_add_library and
  blt_add_executable. (include directories and compile defines)

### Fixed
- Incorrect use of cuda vs cuda_runtime targets
- Improved tutorial documentation
- Incorrect use of Fortran flags with CUDA (Thanks Robert Blake)
- Handle correctly CMake version differences with CUDA host compiler variables
  (Thanks Randy Settgast)
- Handle uncrustify (version 0.68) command line option changes (--no-backup)

### Known Bugs


