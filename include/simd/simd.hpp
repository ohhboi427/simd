#pragma once

#include "simd_export.h"

#include "traits.hpp"
#include "impl/simd_standard.hpp"

namespace simd {
    SIMD_EXPORT auto hello_world() -> void;

    template<typename T, typename A = abi::scalar, typename I = isa::standard>
    struct simd {
        using traits = simd_traits<T, A, I>;
        using type   = typename traits::type;

        type data;
    };

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator+(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::add(a.data, b.data) };
    }
}
