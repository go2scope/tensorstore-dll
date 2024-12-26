include(FetchContent)

# Declare GoogleTest dependency
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0  # Use a specific version tag
)

# Configure GoogleTest options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # Use shared (DLL) run-time lib even when Google Test is built as static lib
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)  # Don't build GMock
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)  # Don't install GTest

# Make GoogleTest available
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Create test executable
add_executable(test_basic
    test_basic.cpp
)

# Link against GTest and your library
target_link_libraries(test_basic
    PRIVATE
        tensorstore_dll
        GTest::gtest
        GTest::gtest_main
)

# Set test properties
set_target_properties(test_basic PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# Add the test to CTest
add_test(
    NAME test_basic
    COMMAND test_basic
)