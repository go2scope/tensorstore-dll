# Define library sources
set(TENSORSTORE_DLL_SOURCES
    tensorstore_dll.cpp
)

set(TENSORSTORE_DLL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/tensorstore_dll/tensorstore_dll.h
)

# Define the library target
add_library(tensorstore_dll ${TENSORSTORE_DLL_SOURCES} ${TENSORSTORE_DLL_HEADERS})
add_library(tensorstore_dll::tensorstore_dll ALIAS tensorstore_dll)

# Set target properties
set_target_properties(tensorstore_dll PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME tensorstore_dll
    DEBUG_POSTFIX "d"
    # VS-specific properties
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$(Configuration)"
    VS_JUST_MY_CODE_DEBUGGING ON
)

# Configure include directories
target_include_directories(tensorstore_dll
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Configure compile definitions
target_compile_definitions(tensorstore_dll
    PRIVATE
        TENSORSTORE_DLL_EXPORTS
        _WIN32_WINNT=0x0601  # Target Windows 7 or later
        NOMINMAX             # Disable min/max macros from Windows.h
        WIN32_LEAN_AND_MEAN  # Exclude rare Windows APIs
        $<$<CONFIG:Debug>:_DEBUG>
)

# Configure Visual Studio specific compile options
target_compile_options(tensorstore_dll
    PRIVATE
        /W4          # Warning level 4
        /WX          # Treat warnings as errors
        /Zi          # Generate debug info
        /MP          # Multi-processor compilation
        /EHsc        # Standard C++ exception handling
        /permissive- # Strict C++ conformance
        /Zc:__cplusplus # Report correct __cplusplus value
        $<$<CONFIG:Release>:/GL>   # Whole program optimization
        $<$<CONFIG:Release>:/Gy>   # Function-level linking
        $<$<CONFIG:Release>:/Oi>   # Generate intrinsic functions
        $<$<CONFIG:Release>:/Ot>   # Favor fast code
        $<$<CONFIG:Release>:/O2>   # Maximum optimization
        $<$<CONFIG:Debug>:/Od>     # Disable optimization
        $<$<CONFIG:Debug>:/RTC1>   # Runtime error checks
)

# Configure linker options
target_link_options(tensorstore_dll
    PRIVATE
        /DEBUG             # Generate debug info
        /INCREMENTAL:NO   # Disable incremental linking
        /OPT:REF          # Remove unreferenced functions/data
        /OPT:ICF          # Enable COMDAT folding
        $<$<CONFIG:Release>:/LTCG> # Link-time code generation
)

# Link dependencies
target_link_libraries(tensorstore_dll
    PRIVATE
        tensorstore::tensorstore
)

# Installation rules
install(TARGETS tensorstore_dll
    EXPORT tensorstore_dll-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tensorstore_dll
    DESTINATION include
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/tensorstore_dll/version.h
    DESTINATION include/tensorstore_dll
)