add_executable(example "main.cpp")

target_link_libraries(example
    PRIVATE
    Simd::Simd
)

if(WIN32)
    add_custom_target(simd_copy_dlls ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_RUNTIME_DLLS:example>"
        "$<TARGET_FILE_DIR:example>"
        COMMAND_EXPAND_LISTS

        DEPENDS $<TARGET_RUNTIME_DLLS:example>

        VERBATIM
        COMMENT "Syncing runtime DLLs..."
    )

    add_dependencies(example simd_copy_dlls)
endif()
