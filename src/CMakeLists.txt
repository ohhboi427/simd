add_library(simd INTERFACE)
add_library(Simd::Simd ALIAS simd)

target_sources(simd
    INTERFACE
    FILE_SET HEADERS
        BASE_DIRS "../include" FILES
        "../include/simd/simd.hpp"
        "../include/simd/macros.hpp"
        "../include/simd/traits.hpp"
        "../include/simd/type.hpp"
        "../include/simd/cast.hpp"
        "../include/simd/math.hpp"
        "../include/simd/ops.hpp"
        "../include/simd/impl/simd_standard.hpp"
        "../include/simd/impl/simd_sse42.hpp"
        "../include/simd/impl/simd_avx2.hpp"
)

target_compile_features(simd
    INTERFACE
    cxx_std_23
)
