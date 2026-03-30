#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "type.hpp"

namespace simd {
    template<typename U, typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto simd_cast(const simd<T, A, I> x) noexcept -> simd<U, A, I> {
        return { simd_traits<T, A, I>::cast(x.data) };
    }

    template<typename U, typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto simd_cvt(const simd<T, A, I> x) noexcept -> simd<U, A, I> {
        return { simd_traits<T, A, I>::cvt(x.data) };
    }
}
