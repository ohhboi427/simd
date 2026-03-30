add_executable(example "main.cpp")

target_link_libraries(example
    PRIVATE
    Simd::Simd
)

target_compile_options(example
    PRIVATE
    -msse4.2
)
