#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "type.hpp"

namespace simd {
    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto trunc(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::trunc(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto round(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::round(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto floor(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::floor(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto ceil(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::ceil(x.data) };
    }
}
