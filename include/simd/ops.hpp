#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "type.hpp"

namespace simd {
    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto fmadd(
        const simd<T, A, I> a,
        const simd<T, A, I> b,
        const simd<T, A, I> c
    ) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::fmadd(a.data, b.data, c.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto fnmadd(
        const simd<T, A, I> a,
        const simd<T, A, I> b,
        const simd<T, A, I> c
    ) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::fnmadd(a.data, b.data, c.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto fmsub(
        const simd<T, A, I> a,
        const simd<T, A, I> b,
        const simd<T, A, I> c
    ) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::fmsub(a.data, b.data, c.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto fnmsub(
        const simd<T, A, I> a,
        const simd<T, A, I> b,
        const simd<T, A, I> c
    ) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::fnmsub(a.data, b.data, c.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator-(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::neg(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator+(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::add(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator-(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::sub(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator*(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::mul(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator/(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::div(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator~(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::inv(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator&(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::conj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator|(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::disj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator^(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::exor(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator<<(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::lshift(a.data, count) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator>>(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::rshift(a.data, count) };
    }
}
