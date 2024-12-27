cmake_minimum_required(VERSION 3.24)

# Disable compiler extensions before project declaration
set(CMAKE_CXX_EXTENSIONS OFF)

project(tensorstore_dll
    VERSION 1.0.0
    DESCRIPTION "TensorStore DLL wrapper for Zarr operations"
    LANGUAGES CXX
)

# Build settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use static runtime
foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
endforeach()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Fetch dependencies
include(FetchContent)

# Fetch Microsoft GSL
FetchContent_Declare(
    GSL
    GIT_REPOSITORY https://github.com/microsoft/GSL.git
    GIT_TAG v4.0.0
)
set(GSL_TEST OFF CACHE BOOL "")
FetchContent_MakeAvailable(GSL)

# Get GSL source directory
FetchContent_GetProperties(GSL SOURCE_DIR GSL_SOURCE_DIR)

# Fetch nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    URL_HASH SHA256=8c4b26bf4b422252e13f332bc5e388ec0ab5c3443d24399acb675e68278d341f
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# Fetch Abseil
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20230125.3
)
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(absl)

# Create the DLL library
add_library(tensorstore_dll SHARED
    src/tensorstore_dll.cpp
    src/error_handling.cpp
)

# Include directories
target_include_directories(tensorstore_dll
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/tensorstore
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include/half
	${GSL_SOURCE_DIR}/include  # Add GSL include directory
)

# Link dependencies
target_link_libraries(tensorstore_dll
    PRIVATE
        Microsoft.GSL::GSL
        absl::base
        absl::strings
        absl::status
        absl::statusor
        nlohmann_json::nlohmann_json
)

# Compile definitions
target_compile_definitions(tensorstore_dll
    PRIVATE
        TENSORSTORE_DLL_EXPORTS
        _WIN32_WINNT=0x0601  # Target Windows 7 or later
        NOMINMAX             # Disable min/max macros
        WIN32_LEAN_AND_MEAN
        TENSORSTORE_NAMESPACE=tensorstore  # Define namespace
        GLOG_NO_ABBREVIATED_SEVERITIES     # Fix Windows.h macro conflicts
        TENSORSTORE_USE_GSL_SPAN=1        # Use GSL span
)

# Compiler options for MSVC
if(MSVC)
    target_compile_options(tensorstore_dll
        PRIVATE
            /W3             # Lower warning level
            /wd4146        # unary minus operator applied to unsigned type
            /wd4251        # class needs to have dll-interface
            /wd4275        # non dll-interface class used as base
            /wd4244        # conversion warnings
            /wd4267        # conversion warnings
            /wd4099        # type name first seen using 'class' now seen using 'struct'
            /wd4324        # structure was padded due to alignment specifier
            /Zc:__cplusplus # Report correct __cplusplus value
            /Zc:preprocessor # Enable standards-conforming preprocessor
            /Zc:externConstexpr # Enable extern constexpr variables
            /Zc:throwingNew  # Assume new throws
            /permissive-   # Standards conformance
            /EHsc          # Enable C++ EH
    )
endif()

# Installation
install(TARGETS tensorstore_dll
    EXPORT tensorstore_dll-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

# Export targets
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tensorstore_dll-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)