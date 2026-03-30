include(GenerateExportHeader)

add_library(simd)
add_library(Simd::Simd ALIAS simd)

generate_export_header(simd)

target_sources(simd
    PRIVATE
    "simd.cpp"

    PUBLIC
    FILE_SET HEADERS
        BASE_DIRS "../include" FILES
        "../include/simd/simd.hpp"
        "../include/simd/traits.hpp"
        "../include/simd/impl/simd_standard.hpp"
        "../include/simd/impl/simd_sse42.hpp"

    FILE_SET generated_headers TYPE HEADERS
        BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}" FILES
        "${CMAKE_CURRENT_BINARY_DIR}/simd_export.h"
)

target_compile_features(simd
    PUBLIC
    cxx_std_23
)

target_compile_options(simd
    PUBLIC
    -mavx512vl
)
