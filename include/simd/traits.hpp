#pragma once

#include <cstddef>

namespace simd {
    namespace abi {
        template<std::size_t Bits>
        struct fixed;

        using scalar = fixed<32U>;
        using m128   = fixed<128U>;
        using m256   = fixed<256U>;
        using m512   = fixed<512U>;
    }

    namespace isa {
        struct standard;

        template<typename I>
        struct prior_isa;

        template<typename I>
        struct default_abi;
    }

    template<typename T, typename A, typename I>
    struct simd_traits;

    template<typename T, typename A, typename I>
    using prior_simd_traits = simd_traits<T, A, typename isa::prior_isa<I>::type>;

    template<typename T, typename A, typename I>
    struct simd_traits : prior_simd_traits<T, A, I> {};
}
