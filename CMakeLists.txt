cmake_minimum_required(VERSION 3.24)
project(tensorstore_dll
    VERSION 1.0.0
    DESCRIPTION "TensorStore DLL wrapper for Zarr operations"
    LANGUAGES CXX
)

# Disable protobuf generation
set(TENSORSTORE_ENABLE_PROTO OFF CACHE BOOL "Disable protobuf in TensorStore")
set(TENSORSTORE_BUILD_PROTOBUF OFF CACHE BOOL "Disable building protobuf")

# Force Visual Studio settings
set(CMAKE_GENERATOR "Visual Studio 16 2019" CACHE INTERNAL "" FORCE)
set(CMAKE_GENERATOR_PLATFORM "x64" CACHE INTERNAL "" FORCE)

# Global configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows-specific options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)
option(STATIC_RUNTIME "Link against static runtime libraries" ON)

# Configure static runtime if requested
if(STATIC_RUNTIME)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endif()

# Set output directories to match Visual Studio defaults
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/Debug")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/Release")

# Disable specific MSVC warnings globally
add_compile_options(
    /wd4251  # class needs to have dll-interface
    /wd4275  # non dll-interface class used as base
)

# Add external dependencies
add_subdirectory(external/tensorstore EXCLUDE_FROM_ALL)

# Configure version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/tensorstore_dll/version.h
    @ONLY
)

# Add main library
add_subdirectory(src)

# Optionally add examples and tests
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Export and installation settings
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tensorstore_dll-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)