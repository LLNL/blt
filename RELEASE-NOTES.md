# BLT Software Release Notes

Notes describing significant changes in each BLT release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

The project release numbers follow [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- BLT_CXX_STD is no longer defined to "c++11" by default. If undefined, BLT will
  not try and add any C++ standard flags.

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
- Object libraries are no longer marked PUBLIC but will pass inherited information
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


