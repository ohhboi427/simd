#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "type.hpp"

namespace simd {
    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto neg(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::neg(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto add(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::add(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto sub(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::sub(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto mul(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::mul(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto div(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::div(a.data, b.data) };
    }

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
    [[nodiscard]] SIMD_INLINE auto inv(const simd<T, A, I> x) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::inv(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto conj(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::conj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto conjinv(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::conjinv(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto disj(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::disj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto exor(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::exor(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto lshift(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::lshift(a.data, count) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto rshift(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::rshift(a.data, count) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto rshift2(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::rshift2(a.data, count) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto eq(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::eq(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto neq(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::neq(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto gt(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::gt(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto ge(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::ge(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto lt(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::lt(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto le(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::le(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto blend(
        const simd<T, A, I> a,
        const simd<T, A, I> b,
        const simd<T, A, I> mask
    ) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::blend(a.data, b.data, mask.data) };
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

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator==(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::eq(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator!=(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::neq(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator>(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::gt(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator>=(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::ge(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator<(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::lt(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator<=(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::le(a.data, b.data) };
    }
}
