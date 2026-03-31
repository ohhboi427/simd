#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "type.hpp"
#include "impl/simd_standard.hpp"

#include "cast.hpp"
#include "math.hpp"
#include "ops.hpp"

#if defined(__SSE4_2__) || defined(__AVX2__)
#   include "impl/simd_sse42.hpp"
#endif

#if defined(__AVX2__)
#   include "impl/simd_avx2.hpp"
#endif

namespace simd {
#if defined(__AVX2__)
    using native_isa = isa::avx2;
#elif defined(__SSE4_2__)
    using native_isa = isa::sse42;
#else
    using native_isa = isa::standard;
#endif

    template<typename T, typename A = isa::default_abi<native_isa>::type>
    using native_simd = simd<T, A, native_isa>;
}
