function(add_example NAME)
    add_executable(${NAME} ${NAME}.cpp)
    
    target_link_libraries(${NAME} 
        PRIVATE tensorstore_dll::tensorstore_dll
    )
    
    set_target_properties(${NAME} PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$(Configuration)"
        VS_JUST_MY_CODE_DEBUGGING ON
    )
    
    # Copy DLL to example output directory post-build
    add_custom_command(TARGET ${NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tensorstore_dll>
            $<TARGET_FILE_DIR:${NAME}>
    )
endfunction()

add_example(basic_usage)
add_example(metadata_example)
add_example(chunking_example)