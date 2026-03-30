#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "type.hpp"

namespace simd {
    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto abs(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::abs(x.data) };
    }

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

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto fract(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return x - floor(x);
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto mod(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return a - b * trunc(a / b);
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto min(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::min(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto max(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::max(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto clamp(
        const simd<T, A, I> x,
        const simd<T, A, I> a,
        const simd<T, A, I> b
    ) noexcept -> simd<T, A, I> {
        return min(max(x, a), b);
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto sqrt(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::sqrt(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto rsqrt(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::rsqrt(x.data) };
    }
}
